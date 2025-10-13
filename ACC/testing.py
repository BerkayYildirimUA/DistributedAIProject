import glob
import os
import sys
import random
import time

from yolo_experiment.carla_validate_yolo import spectator

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# --- Setup CARLA agents path ---
# Add the PythonAPI folder to the system path
try:
    # pythonapi_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pythonapi_path = '/opt/carla-simulator/PythonAPI'  # TODO: change this to your path
    sys.path.append(pythonapi_path)
except IndexError:
    pass

import carla
from agents.navigation.basic_agent import BasicAgent


def main():
    """Main function"""
    actor_list = []
    client = None

    try:
        # 1. Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # 2. Spawn a vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')  # Find a vehicle blueprint

        # Choose a random spawn point from the map
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("Error: No spawn points found in the current map.")
            return
        spawn_point = random.choice(spawn_points)

        # Spawn the vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f'Spawned vehicle: {vehicle.type_id} at {spawn_point.location}')

        # 3. Create an agent to control the vehicle
        # The BasicAgent drives to a target speed and follows waypoints
        agent = BasicAgent(vehicle, target_speed=30)  # Target speed in km/h

        # 4. Set a random destination
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)
        print(f'Agent is driving to destination: {destination}')

        spectator = world.get_spectator()
        transform = spectator.get_transform()
        transform.location = spawn_points
        spectator.set_transform(carla.Transform())

        

        # 5. Main loop to run the simulation
        while True:
            world.wait_for_tick()

            if agent.done():
                print("The agent has reached its destination.")
                break

            control = agent.run_step()
            vehicle.apply_control(control)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 6. Clean up: destroy the spawned actors
        if client and actor_list:
            print('Destroying actors...')
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            print('Actors destroyed.')


if __name__ == '__main__':
    main()