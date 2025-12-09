import logging
import math
import os
import random
import traceback
from typing import Optional, List, Tuple
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

    def _load_world_safely(self, client: carla.Client, map_name: str, server_name: str) -> Tuple[carla.World, bool]:
        """
        Loads a map with patience, verification, and memory cleanup.
        """
        logging.info(f"[{server_name}] Preparing to load map: {map_name}...")
        client.set_timeout(5 * 60)

        try:
            current_world = client.get_world()
            current_map_name = current_world.get_map().name

            if "OpenDriveMap" in current_map_name and map_name.startswith("CUSTOM_"):
                logging.info(f"[{server_name}] Custom map already loaded. Skipping regeneration.")
                logging.info(f"[{server_name}] Applying Synchronous Settings...")

                settings = current_world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.delta_seconds

                current_world.apply_settings(settings)


                return current_world, True

            if map_name in current_map_name:
                logging.info(f"[{server_name}] Map {map_name} is already loaded. Skipping reload.")

                settings = current_world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.delta_seconds

                current_world.apply_settings(settings)

                return current_world, False
        except RuntimeError:
            pass



        is_custom : bool = map_name.startswith("CUSTOM_")
        if is_custom:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            acc_dir = os.path.dirname(current_dir)
            file_name = os.path.join(acc_dir, "Agents", map_name + ".xodr")


            if not os.path.exists(file_name):
                raise FileNotFoundError(f"[{server_name}] Could not find OpenDRIVE file: {file_name}")

            logging.info(f"[{server_name}] Reading OpenDRIVE file: {file_name}...")
            with open(file_name, 'r') as f:
                xodr_content = f.read()

            logging.info(f"[{server_name}] Generating procedural world (No 3D Models)...")

            client.generate_opendrive_world(
                xodr_content,
                carla.OpendriveGenerationParameters(
                    vertex_distance=0.2,
                    max_road_length=500.0,
                    wall_height=1.0,  # Invisible walls to keep car on track
                    additional_width=0.6,  # Wider lanes at junctions
                    smooth_junctions=True,
                    enable_mesh_visibility=True
                )

            )
            time.sleep(2.0)

        # 2. CHECK CURRENT STATE
        # If the map is already loaded, don't reload it (saves 5 seconds)

        else:
            try:
                current_map = client.get_world().get_map().name
                if current_map.endswith(map_name):
                    logging.info(f"[{server_name}] Map {map_name} is already loaded. Skipping reload.")
                    return client.get_world(), False
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
        logging.info(f"[{server_name}] Applying Synchronous Settings...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_seconds

        world.apply_settings(settings)


        world.tick()

        return world, is_custom

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
            world_real, real_world_is_custom = self._load_world_safely(client_real, map_name, "REAL")
            logging.info(f"Successfully loaded REAL world. Map: {world_real.get_map().name}")

            logging.info(f"Connecting to MIRROR CARLA server at {self.host}:{self.mirror_port}")
            client_mirror = carla.Client(self.host, self.mirror_port)
            logging.info(f"Connection to MIRROR server successful. Loading map {map_name}...")

            #world_mirror = client_mirror.load_world(map_name)
            world_mirror, mirror_world_is_custom = self._load_world_safely(client_mirror, map_name, "MIRROR")


            if world_mirror.get_map().name != world_real.get_map().name:
                raise RuntimeError(
                    f"Map mismatch! Real world: {world_real.get_map().name}, Mirror world: {world_mirror.get_map().name}")
            logging.info(f"Successfully loaded MIRROR world. Map: {world_mirror.get_map().name}")

            self.duo_client = DuoClient(client_real, client_mirror)
            self.duo_world = DuoWorld(world_real, world_mirror)

            self.duo_world.tick()

            if real_world_is_custom and mirror_world_is_custom:
                green_time, yellow_time, red_time = self.reset_traffic_lights(self.duo_world.get_real_world())
                self.reset_traffic_lights(self.duo_world.get_mirror_world(), green_time, yellow_time, red_time)

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
                max_substep_delta_time=0.01,
                max_substeps=10
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
            self.tm_mirror.set_respawn_dormant_vehicles(False)


            # --- LEAD ---
            lead_is_not_skipped = True
            if self.args.do_train:
                lead_is_not_skipped = random.random() > 0.25


            if self.scenario.lead_car_bp_name != "" and lead_is_not_skipped:
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

            self.duo_world.tick()
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
        if self.ego and self.ego.is_alive():
            final_real_ego_transform = self.ego.get_real_transform()
            real_velocity = self.ego.get_velocity()
            real_angular_vel = self.ego.get_angular_velocity()

            if final_real_ego_transform and self.ego.mirror and self.ego.mirror.is_alive:
                try:
                    self.ego.mirror.set_transform(final_real_ego_transform)
                    self.ego.set_mirror_velocity(real_velocity)
                    self.ego.set_mirror_angular_velocity(real_angular_vel)
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

    def repair_actor_pair(self, duo_actor: DuoActor) -> bool:
        """
        Attempts to repair a DuoActor where one actor has died but the other survives.
        Uses the surviving actor's transform.
        """

        if not self.duo_world or not self.duo_client:
            logging.error("Cannot repair actor pair. Worlds not initialized.")
            return False

        real_alive = duo_actor.real is not None and duo_actor.real.is_alive
        mirror_alive = duo_actor.mirror is not None and duo_actor.mirror.is_alive

        # Both alive? Nothing to repair.
        if real_alive and mirror_alive:
            logging.debug("Both actors are alive. No repair needed.")
            return True

        # Both dead? Can't salvage this one.
        if not real_alive and not mirror_alive:
            logging.error("Both actors are dead. Cannot repair without a survivor.")
            return False

        # Determine survivor and target world
        if real_alive:
            survivor = duo_actor.real
            target_world = self.duo_world.get_mirror_world()
            target_side = "MIRROR"
        else:
            survivor = duo_actor.mirror
            target_world = self.duo_world.get_real_world()
            target_side = "REAL"

        # Grab transform from the living actor
        try:
            repair_transform = survivor.get_transform()
            repair_transform.location.z += 0.55

            survivor_velocity = survivor.get_velocity()
            survivor_ang_velocity = survivor.get_angular_velocity()
        except Exception as e:
            logging.error(f"Failed to get transform from surviving actor: {e}")
            return False

        # Resolve blueprint
        try:
            type_id = survivor.type_id
            blueprint = self.blueprint_library.find(type_id)
        except Exception as e:
            logging.error(f"Failed to infer blueprint from survivor: {e}")
            return False

        logging.debug(f"Attempting to respawn {target_side} actor using {blueprint.id} at {repair_transform.location}")

        # Spawn the replacement
        new_actor = self.spawn_actor_sync(target_world, blueprint, repair_transform)
        if not new_actor:
            logging.error(f"Failed to spawn replacement {target_side} actor.")
            return False

        new_actor.set_target_velocity(survivor_velocity)
        new_actor.set_target_angular_velocity(survivor_ang_velocity)

        # Patch up the DuoActor
        if target_side == "MIRROR":
            duo_actor.mirror = new_actor
        else:
            duo_actor.real = new_actor

        logging.debug(f"Successfully repaired {target_side} actor. New ID: {new_actor.id}")
        return True

    def reset_traffic_lights(self, world, green_time=0.0, yellow_time=0.0, red_time=0.0):
        """Force all traffic lights to cycle"""
        actors = world.get_actors().filter('traffic.traffic_light')
        for light in actors:
            try:
                if green_time == 0.0:
                   green_time = random.randint(4, 35)
                   if random.random() <= 0.25:
                        green_time = 200

                if yellow_time == 0.0:
                    yellow_time = random.randint(3, 5)
                    if yellow_time == 200:
                        red_time = 1
                if red_time == 0.0:
                    red_time = random.randint(4, 35)
                    if green_time == 200:
                        red_time = 1

                light.set_state(carla.TrafficLightState.Green)
                light.set_green_time(green_time)
                light.set_yellow_time(yellow_time)
                light.set_red_time(red_time)

                return green_time, yellow_time, red_time
            except:
                pass

    def revive_ego_pair(self) -> bool:
        """
        Specialized repair function for the ego vehicle.
        Handles ego-specific configuration after resurrection (e.g., TM settings).
        """
        if not self.ego:
            logging.error("No ego actor exists to revive.")
            return False

        was_mirror_dead = self.ego.mirror is None or not self.ego.mirror.is_alive

        success = self.repair_actor_pair(self.ego)

        if success and was_mirror_dead:
            # Reconfigure the mirror ego with TM settings
            try:
                if self.tm_mirror and self.ego.mirror:
                    self.ego.set_mirror_autopilot(True, self.mirror_traffic_manager_port)
                    self.tm_mirror.auto_lane_change(self.ego.mirror, False)
                    self.ego.set_mirror_physics(True)
                    self.duo_world.get_mirror_world().tick()
                    logging.debug("Ego mirror actor reconfigured with Traffic Manager settings.")
            except Exception as e:
                logging.warning(f"Failed to reconfigure ego mirror TM settings: {e}")

        return success

    def revive_lead_pair(self, speed_limit = 0) -> bool:
        """
        Specialized repair function for the lead vehicle.
        Handles lead-specific configuration after resurrection (e.g., TM settings).
        """
        if not self.lead:
            logging.error("No lead actor exists to revive.")
            return False

        was_mirror_dead = self.lead.mirror is None or not self.lead.mirror.is_alive

        success = self.repair_actor_pair(self.lead)

        if success and was_mirror_dead:
            # Reconfigure the mirror lead with TM settings
            try:
                if self.tm_mirror and self.lead.mirror:
                    self.lead.set_mirror_autopilot(True, self.mirror_traffic_manager_port)
                    self.lead.set_mirror_physics(True)
                    self.duo_world.get_mirror_world().tick()

                    if speed_limit > 0:
                        self.tm_mirror.set_desired_speed(self.lead.mirror, speed_limit * 3.6)

                    logging.debug("Lead mirror actor reconfigured with Traffic Manager settings.")
            except Exception as e:
                logging.warning(f"Failed to reconfigure lead mirror TM settings: {e}")

        return success

    def soft_reset(self) -> bool:
        """
        Soft reset: Respawn actors without full world disconnect.
        This avoids the session assertion failures from rapid reconnection.

        Returns:
            bool: True if reset successful, False if full reset needed
        """
        logging.info("Performing soft reset (respawn actors only)...")

        try:
            # Destroy existing actors but keep world connection
            actors_to_destroy = []
            if self.ego:
                actors_to_destroy.append(self.ego)
            if self.lead:
                actors_to_destroy.append(self.lead)
            actors_to_destroy.extend(self.npcs)

            for actor_pair in actors_to_destroy:
                if actor_pair:
                    try:
                        actor_pair.destroy()
                    except Exception as e:
                        logging.warning(f"Error destroying actor during soft reset: {e}")

            self.ego = None
            self.lead = None
            self.npcs = []

            # Tick to process destructions
            self.duo_world.tick()
            time.sleep(0.2)

            # Reset traffic lights
            green_time, yellow_time, red_time = self.reset_traffic_lights(self.duo_world.get_real_world())
            self.reset_traffic_lights(self.duo_world.get_mirror_world(), green_time, yellow_time, red_time)

            # 4. Re-run actor spawning (reuse existing spawn logic)
            # Get spawn points
            world_real = self.duo_world.get_real_world()
            self.spawn_points = world_real.get_map().get_spawn_points()

            if not self.spawn_points:
                waypoint = world_real.get_map().get_waypoint_xodr(road_id=1, lane_id=-1, s=5.0)
                if waypoint:
                    transform = waypoint.transform
                    transform.location.z += 0.55
                    self.spawn_points = [transform]
                else:
                    self.spawn_points = [carla.Transform(carla.Location(x=0, y=0, z=0.55), carla.Rotation(yaw=0.0))]

            available_spawn_points = list(self.spawn_points)

            # Spawn EGO
            ego_spawn_point_index = int(
                self.args.spawn_point) if self.args.spawn_point != "random" else random.randrange(
                len(available_spawn_points))
            ego_spawn_point = available_spawn_points.pop(ego_spawn_point_index % len(available_spawn_points))

            filter_ego = self.scenario.ego_car_bp_name if self.scenario else 'vehicle.tesla.model3'
            if filter_ego == "random":
                filter_ego = 'vehicle.*.*'

            ego_bp = random.choice(self.blueprints_vehicles.filter(filter_ego))
            self.ego = self.spawn_actor_pair(ego_bp, ego_spawn_point)

            if not self.ego:
                logging.error("Soft reset failed: Could not spawn EGO")
                return False

            logging.info(f"Spawned EGO pair: Real ID {self.ego.real.id}, Mirror ID {self.ego.mirror.id}")

            # Spawn LEAD if configured
            if self.scenario and self.scenario.lead_car_bp_name:
                if available_spawn_points:
                    lead_spawn_point = available_spawn_points.pop(0)
                else:
                    lead_spawn_point = ego_spawn_point
                    lead_spawn_point.location.x += 25  # Offset from ego

                lead_bp = random.choice(self.blueprints_vehicles.filter(self.scenario.lead_car_bp_name))
                self.lead = self.spawn_actor_pair(lead_bp, lead_spawn_point)

                if self.lead:
                    logging.info(f"Spawned LEAD pair: Real ID {self.lead.real.id}, Mirror ID {self.lead.mirror.id}")

            # 5. Configure mirror actors with TM
            actors_to_configure = []
            if self.ego:
                self.ego.set_mirror_autopilot(True, self.mirror_traffic_manager_port)
                self.ego.set_mirror_physics(True)
                self.tm_mirror.auto_lane_change(self.ego.mirror, False)

            if self.lead:
                self.lead.set_mirror_autopilot(True, self.mirror_traffic_manager_port)
                self.lead.set_mirror_physics(True)





            # 6. Final tick
            self.duo_world.tick()

            logging.info("Soft reset complete.")
            return True

        except Exception as e:
            logging.error(f"Soft reset failed with error: {e}")
            traceback.print_exc()
            return False


    def cleanup(self, delete_all=False):
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

        all_traffic_actors : carla.ActorList = self.duo_world.get_real_world().get_actors()


        if delete_all:
            for actor in all_traffic_actors:
                if actor.is_alive:
                    actor.destroy()

            all_traffic_actors = self.duo_world.get_mirror_world().get_actors()
            for actor in all_traffic_actors:
                if actor.is_alive:
                    actor.destroy()

        time.sleep(1)

        tm = self.duo_client.mirror.get_trafficmanager()
        tm.shut_down()

        time.sleep(1)

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
