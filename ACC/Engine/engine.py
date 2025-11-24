import logging
import os
import random
import traceback
from typing import Optional, List
from ACC.Engine.scenario import Scenario

import carla
from ACC.Engine.duo_classes import DuoActor, DuoClient, DuoWorld
import time
import gc

class Engine():

    def __init__(self, args, scenario : Optional[Scenario] = None):

        self.args = args
        self.scenario = scenario

        # client stuff
        self.host = args.host
        self.real_port = args.real_port
        self.mirror_port = args.mirror_port

        # TM
        self.mirror_traffic_manager_port = args.tm_mirror_port

        # world settings

        temp_map_name = scenario.map if scenario is not None else args.map



        self.map_name = temp_map_name
        self.delta_seconds = scenario.delta_seconds if scenario is not None else args.delta_seconds

        # scenario
        self.duo_client: Optional[DuoClient] = None
        self.duo_world: Optional[DuoWorld] = None
        self.tm_mirror: Optional[carla.TrafficManager] = None
        self.blueprint_library: Optional[carla.BlueprintLibrary] = None
        self.blueprints_vehicles: Optional[carla.BlueprintLibrary] = None
        self.spawn_points: Optional[List[carla.Transform]] = None
        self.spectator: Optional[carla.Actor] = None

        # Actrors
        self.num_npcs = scenario.number_of_npc if scenario is not None else args.num_npcs
        self.ego: Optional[DuoActor] = None
        self.lead: Optional[DuoActor] = None
        self.npcs: List[DuoActor] = []

        self.scenario = scenario

    def _load_world_safely(self, client: carla.Client, map_name: str, server_name: str):
        """
        Loads a map with patience, verification, and memory cleanup.
        """
        logging.info(f"[{server_name}] Preparing to load map: {map_name}...")
        client.set_timeout(5 * 60)

        if map_name.startswith("CUSTOM_"):
            file_name = map_name + ".xodr"

            if not os.path.exists(file_name):
                raise FileNotFoundError(f"[{server_name}] Could not find OpenDRIVE file: {file_name}")

            logging.info(f"[{server_name}] Reading OpenDRIVE file: {file_name}...")
            with open(file_name, 'r') as f:
                xodr_content = f.read()

            logging.info(f"[{server_name}] Generating procedural world (No 3D Models)...")

            client.generate_opendrive_world(
                xodr_content,
                carla.OpendriveGenerationParameters(
                    vertex_distance=2.0,
                    max_road_length=500.0,
                    wall_height=1.0,  # Invisible walls to keep car on track
                    additional_width=0.6,  # Wider lanes at junctions
                    smooth_junctions=True,
                    enable_mesh_visibility=True
                )
            )

        # 2. CHECK CURRENT STATE
        # If the map is already loaded, don't reload it (saves 5 seconds)

        else:
            try:
                current_map = client.get_world().get_map().name
                if current_map.endswith(map_name):
                    logging.info(f"[{server_name}] Map {map_name} is already loaded. Skipping reload.")
                    return client.get_world()
            except RuntimeError:
                pass  # World might not exist yet, that's fine

            gc.collect()

            # 4. LOAD THE WORLD
            logging.info(f"[{server_name}] Sending Load Command...")
            client.load_world(map_name)

        # 5. VERIFICATION LOOP
        world = None
        for i in range(10):
            try:
                world = client.get_world()

                if not map_name.startswith("CUSTOM_"):
                    if world.get_map().name.rsplit('/', 1)[1] == map_name:
                        logging.info(f"[{server_name}] Map verified successfully.")
                        break

                else:
                    if world.get_map() is not None:
                        logging.info(f"[{server_name}] Custom map generated successfully.")
                        break

            except RuntimeError:
                pass

            logging.info(f"[{server_name}] Waiting for map to settle... ({i + 1}/10)")
            time.sleep(1.0)

        if not world:
            raise RuntimeError(f"[{server_name}] Failed to load map {map_name} after waiting.")

        # 6. WARM UP TICKS
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_seconds
        world.apply_settings(settings)

        for _ in range(5):
            world.tick()

        return world

    def connect_to_worlds(self):
        client_real = None
        client_mirror = None
        world_real = None
        world_mirror = None

        try:




            logging.info(f"Connecting to REAL CARLA server at {self.host}:{self.real_port}")
            client_real = carla.Client(self.host, self.real_port)

            map_name = self.map_name if self.map_name != "random" else random.choice(["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07"])


            logging.info(f"Connection to REAL server successful. Loading map: {map_name}...")

            #world_real = client_real.load_world(map_name)
            world_real = self._load_world_safely(client_real, map_name, "REAL")
            logging.info(f"Successfully loaded REAL world. Map: {world_real.get_map().name}")

            logging.info(f"Connecting to MIRROR CARLA server at {self.host}:{self.mirror_port}")
            client_mirror = carla.Client(self.host, self.mirror_port)
            logging.info(f"Connection to MIRROR server successful. Loading map {map_name}...")

            #world_mirror = client_mirror.load_world(map_name)
            world_mirror = self._load_world_safely(client_mirror, map_name, "MIRROR")


            if world_mirror.get_map().name != world_real.get_map().name:
                raise RuntimeError(
                    f"Map mismatch! Real world: {world_real.get_map().name}, Mirror world: {world_mirror.get_map().name}")
            logging.info(f"Successfully loaded MIRROR world. Map: {world_mirror.get_map().name}")

            self.duo_client = DuoClient(client_real, client_mirror)
            self.duo_world = DuoWorld(world_real, world_mirror)
            self.duo_world.tick()
            logging.info("DuoClient and DuoWorld created successfully.")

        except Exception as e:
            logging.error(f"Failed to connect to CARLA worlds: {e}")
            traceback.print_exc()
            if client_real and not world_real: client_real = None
            if client_mirror and not world_mirror: client_mirror = None
            self.duo_client = DuoClient(client_real, client_mirror) if (client_real or client_mirror) else None
            self.duo_world = DuoWorld(world_real, world_mirror) if (world_real and world_mirror) else None
            self.cleanup()
            raise

    def spawn_actor_sync(self, world: carla.World, blueprint: carla.BlueprintLibrary, transform: carla.Transform):
        actor = world.try_spawn_actor(blueprint, transform)

        if actor is None:
            print(f"Warning: Failed to spawn actor {blueprint.id} at {transform.location}")
            return None

        for _ in range(5):  # Try a few times
            world.tick()
            if world.get_actor(actor.id) is not None:
                return actor

        print(f"Warning: Actor {actor.id} did not appear in world after spawning.")
        actor.destroy()  # Clean up to be sure
        return None

    def spawn_actor_pair(self, blueprint: carla.BlueprintLibrary, transform: carla.Transform) -> Optional[DuoActor]:
        """
        creates a type of actors in the Mirror and Real world
        """
        if not self.duo_world or not self.duo_client:
            logging.error("Worlds or clients not initialized. Cannot spawn actor pair.")
            return None

        world_real = self.duo_world.get_real_world()
        world_mirror = self.duo_world.get_mirror_world()

        # Spawn Real Actor
        real_actor = self.spawn_actor_sync(world_real, blueprint, transform)
        if not real_actor:
            logging.error(f"Failed to spawn REAL actor {blueprint.id}")
            return None

        # Spawn Mirror Actor
        mirror_actor = self.spawn_actor_sync(world_mirror, blueprint, transform)
        if not mirror_actor:
            logging.error(f"Failed to spawn MIRROR actor {blueprint.id}")
            try:
                logging.warning(f"Destroying real actor {real_actor.id} because mirror spawn failed.")
                if real_actor.is_alive: real_actor.destroy()
                world_real.tick()  # Tick after destroy
            except Exception as e:
                logging.error(f"Exception during cleanup of real actor {real_actor.id}: {e}")
            return None

        logging.info(
            f"Successfully spawned actor pair: Real ID {real_actor.id}, Mirror ID {mirror_actor.id} ({blueprint.id})")
        return DuoActor(real_actor, mirror_actor)

    def set_scenario(self, scenario : Scenario):
        self.delta_seconds = scenario.delta_seconds
        self.map_name = scenario.map
        self.num_npcs = scenario.number_of_npc
        self.scenario = scenario


    def setup(self):
        """
        setup the world
        """
        if not self.duo_world or not self.duo_client:
            logging.error("Cannot run setup. Worlds not connected.")
            return False

        try:
            # Apply world settings
            logging.info(f"Putting both worlds in sync mode with delta_seconds={self.delta_seconds}")
            settings_sync = carla.WorldSettings(
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,

            )
            self.duo_world.set_both_worlds_settings(settings_sync)
            self.duo_world.tick()
            logging.info("Sync mode activated.")

            # Setup Traffic Manager (Mirror World)
            logging.info(f"Setting up Traffic Manager in Mirror world on port {self.mirror_traffic_manager_port}")
            self.tm_mirror : carla.TrafficManager = self.duo_client.mirror.get_trafficmanager(self.mirror_traffic_manager_port)
            self.tm_mirror.set_synchronous_mode(True)
            self.duo_world.get_mirror_world().tick()
            logging.info("Mirror TM set to synchronous mode.")


            # Get Blueprints and Spawn Points
            logging.info("Getting blueprints and spawn points...")
            world_real = self.duo_world.get_real_world()
            self.blueprint_library = world_real.get_blueprint_library()
            self.blueprints_vehicles = self.blueprint_library.filter('vehicle.*.*')
            self.spawn_points = world_real.get_map().get_spawn_points()
            if not self.spawn_points:
                logging.info("Map has no spawn points! Generating waypoint...")
                waypoint = world_real.get_map().get_waypoint_xodr(road_id=1, lane_id=-1, s=5.0)
                if waypoint:
                    transform = waypoint.transform
                    transform.location.z += 2.0
                    self.spawn_points = [transform]
                else:
                    logging.warning("WARNING: XODR lookup failed. Using hardcoded coordinates.")
                    self.spawn_points = [carla.Transform(carla.Location(x=0, y=0, z=2.0), carla.Rotation(yaw=0.0))]


            logging.info(f"Found {len(self.spawn_points)} spawn points.")
            # Use a copy for spawning to avoid modifying the original list if needed later. Was annoying to find
            available_spawn_points = list(self.spawn_points)

            # 4. Spawn Actors
            logging.info("Spawning actor pairs...")

            # --- EGO ---
            if not available_spawn_points: raise RuntimeError("Spawn points list is empty.")
            ego_spawn_point_index = int(self.args.spawn_point) if self.args.spawn_point != "random" else random.randrange(len(available_spawn_points))
            ego_spawn_point = available_spawn_points.pop(ego_spawn_point_index)

            filter = self.scenario.ego_car_bp_name if self.scenario is not None else 'vehicle.tesla.model3'
            if filter == "random":
                filter = 'vehicle.*.*'

            ego_bp = random.choice(self.blueprints_vehicles.filter(filter))
            self.ego = self.spawn_actor_pair(ego_bp, ego_spawn_point)
            if not self.ego: raise RuntimeError("Failed to spawn EGO pair.")
            logging.info(f"Spawned EGO pair: Real ID {self.ego.real.id}, Mirror ID {self.ego.mirror.id}")

            self.tm_mirror.auto_lane_change(self.ego.mirror, False) #turn off lane changes

            # --- LEAD ---
            if self.scenario.lead_car_bp_name != "":
                lead_transform = carla.Transform(
                    ego_spawn_point.location + ego_spawn_point.get_forward_vector() * 15.0,
                    ego_spawn_point.rotation
                )

                filter = self.scenario.ego_car_bp_name if self.scenario is not None else 'vehicle.mitsubishi.fusorosa'
                if filter == "random":
                    filter = 'vehicle.*.*'

                lead_bp = random.choice(self.blueprints_vehicles.filter(filter))
                self.lead = self.spawn_actor_pair(lead_bp, lead_transform)
                if not self.lead: raise RuntimeError("Failed to spawn LEAD pair.")
                logging.info(f"Spawned LEAD pair: Real ID {self.lead.real.id}, Mirror ID {self.lead.mirror.id}")

            # --- NPCs ---
            logging.info(f"Attempting to spawn {self.num_npcs} NPC pairs...")
            npc_spawn_count = 0
            random.shuffle(available_spawn_points)
            for i in range(self.num_npcs):
                if not available_spawn_points:
                    logging.warning("Ran out of unique spawn points for NPCs.")
                    break
                spawn_point_npc = available_spawn_points.pop()
                bp_npc = random.choice(self.blueprints_vehicles)
                npc_pair = self.spawn_actor_pair(bp_npc, spawn_point_npc)
                if npc_pair:
                    self.npcs.append(npc_pair)
                    npc_spawn_count += 1
                else:
                    logging.warning(f"Failed to spawn NPC pair {i + 1}/{self.num_npcs}")
            logging.info(f"Successfully spawned {npc_spawn_count} NPC pairs.")
            if npc_spawn_count == 0 and self.num_npcs > 0:
                logging.warning("No NPCs were spawned.")

            # Configure Mirror Actors (Autopilot/Physics)
            logging.info("Configuring mirror actors (autopilot/physics)...")
            actors_to_configure = []
            if self.ego: actors_to_configure.append(self.ego)
            if self.lead: actors_to_configure.append(self.lead)
            actors_to_configure.extend(self.npcs)

            for actor_pair in actors_to_configure:
                if actor_pair:
                    actor_pair.set_mirror_autopilot(True, self.mirror_traffic_manager_port)
                    actor_pair.set_mirror_physics(True)

            self.duo_world.get_mirror_world().tick()
            logging.info("Mirror actors configured.")

            # Spectator setup
            self.spectator = self.duo_world.get_real_world().get_spectator()

            self.duo_world.tick()
            logging.info("Setup complete.")
            return True

        except Exception as e:
            logging.error(f"An error occurred during setup: {e}")
            traceback.print_exc()
            return False

    def synchronization_real_npc_with_mirror_npcs(self):
        if not self.duo_world:
            logging.error("Cannot synchronize, DuoWorld not initialized.")
            return

        actors_to_sync: List[Optional[DuoActor]] = []
        if self.lead: actors_to_sync.append(self.lead)
        actors_to_sync.extend(self.npcs)

        sync_count = 0
        fail_count = 0
        for actor_pair in actors_to_sync:
            if actor_pair and actor_pair.real and actor_pair.mirror and actor_pair.is_alive():
                mirror_transform = actor_pair.get_mirror_transform()
                if mirror_transform:
                    try:
                        actor_pair.real.set_transform(mirror_transform)
                        sync_count += 1
                    except Exception as e:
                        logging.warning(f"Failed to apply sync transform to real actor {actor_pair.real.id}: {e}")
                        fail_count += 1

    def synchronization_mirror_ego_with_real_ego(self):
        if self.ego.is_alive():
            final_real_ego_transform = self.ego.get_real_transform()
            if final_real_ego_transform and self.ego.mirror and self.ego.mirror.is_alive:
                try:
                    self.ego.mirror.set_transform(final_real_ego_transform)
                except Exception as e:
                    logging.error(f"Error syncing mirror ego {self.ego.mirror.id} with real transform: {e}")

    def update_spectator(self):
        if self.spectator and self.ego and self.ego.is_alive():
            ego_transform_spec = self.ego.get_real_transform()
            if ego_transform_spec:
                spectator_location = ego_transform_spec.location - 10 * ego_transform_spec.get_forward_vector() + carla.Location(
                    z=5)
                spectator_rotation = carla.Rotation(pitch=-15, yaw=ego_transform_spec.rotation.yaw, roll=0)
                try:
                    self.spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
                except Exception as e:
                    logging.warning(f"Failed to update spectator transform: {e}")

    def cleanup(self):
        logging.info('Initiating cleanup...')

        # Store actors to destroy
        actors_to_destroy: List[Optional[DuoActor]] = []
        if self.ego: actors_to_destroy.append(self.ego)
        if self.lead: actors_to_destroy.append(self.lead)
        actors_to_destroy.extend(self.npcs)

        # Restore Async Settings
        settings_async = carla.WorldSettings(
            synchronous_mode=False,
            fixed_delta_seconds=0.0
        )
        if self.duo_world:
            try:
                logging.info("Restoring world settings to asynchronous...")
                self.duo_world.set_both_worlds_settings(settings_async)
                self.duo_world.tick()
            except Exception as e:
                logging.error(f"Error restoring world settings: {e}")

        # Destroy Actors
        logging.info(f"Destroying {len(actors_to_destroy)} actor pairs...")
        destroyed_count = 0
        for actor_pair in actors_to_destroy:
            if actor_pair:
                actor_pair.destroy()
                destroyed_count += 1
        logging.info(f"Destroy method called for {destroyed_count} actor pairs.")

        if self.duo_world:
            self.duo_world.tick()



        # Clear
        self.ego = None
        self.lead = None
        self.npcs = []

        self.tm_mirror = None
        self.spectator = None
        self.duo_world = None
        self.duo_client = None

        logging.info('Cleanup process finished.')
