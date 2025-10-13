import carla
import argparse
import queue
import random
from typing import List, Optional
from implementations import CarlaStateSensor, SimpleAccAgent, PygameUI


def main_loop(args):
    client: carla.Client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world: Optional[carla.World] = None

    actor_list: List[carla.Actor]  = []
    sensor_list: List[carla.Sensor]  = []
    ui: Optional[PygameUI] = None

    try:
        # World Settings
        world: carla.World = client.load_world('Town04')
        settings: carla.WorldSettings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Traffic Manger
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # Blueprints
        blueprint_library: carla.BlueprintLibrary = world.get_blueprint_library()
        blueprints_vehicles: carla.BlueprintLibrary = blueprint_library.filter('vehicle.*.*')
        spawn_points: List[carla.Transform] = world.get_map().get_spawn_points()

        # Spawn NPCs
        for i in range(0, 50):
            world.try_spawn_actor(random.choice(blueprints_vehicles), random.choice(spawn_points))

        world.tick() #tick is needed for the actors to spawn so can use world.get_actors()

        for vehicle in world.get_actors().filter('*vehicle*'):
            vehicle.set_autopilot(True, 8000)
            actor_list.append(vehicle)

        # Spawn Ego
        ego_spawn_point: carla.Transform = spawn_points[100]
        ego_vehicle: carla.Vehicle = world.spawn_actor(random.choice(blueprints_vehicles), ego_spawn_point)
        actor_list.append(ego_vehicle)

        # Spawn Lead TODO: spawn is wrong
        lead_transform = carla.Transform(
            ego_spawn_point.location + carla.Location(y=20),
            ego_spawn_point.rotation
        )

        lead_vehicle: carla.Vehicle = world.spawn_actor(blueprints_vehicles.filter("*.mitsubishi.fusorosa")[0], lead_transform)
        lead_vehicle.set_autopilot(True, 8000)
        actor_list.append(lead_vehicle)

        #Set spectator to Ego
        spectator: carla.Actor = world.get_spectator()
        camera_bp: carla.ActorBlueprint = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))

        transform: carla.Transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera: carla.Sensor = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle)
        sensor_list.append(camera)

        image_queue: queue.Queue[carla.Image] = queue.Queue()
        camera.listen(image_queue.put)

        #TODO: delete later
        ego_vehicle.set_autopilot(True, 8000)
        while True:
            world.tick()
            # Move spectator to follow ego vehicle
            ego_transform = ego_vehicle.get_transform()
            spectator_transform = spectator.get_transform()
            spectator_location = ego_transform.location - 10 * ego_transform.get_forward_vector() + carla.Location(z=5)

            spectator.set_transform(carla.Transform(spectator_location, spectator_transform.rotation))


            image: carla.Image = image_queue.get()

            # TODO: uncomment later, change controls and stuff
            #control = ego_vehicle.get_control()
            #ego_vehicle.apply_control(control)


    finally:
        print('Cleaning up')
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        if ui:
            ui.destroy()

        client.apply_batch([carla.command.DestroyActor(s) for s in sensor_list])
        client.apply_batch([carla.command.DestroyActor(a) for a in actor_list])
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA ACC Infrastructure Test')
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    parser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    parser.add_argument('--width', default=1280, type=int, help='Camera image width')
    parser.add_argument('--height', default=720, type=int, help='Camera image height')
    args = parser.parse_args()

    main_loop(args)
