import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import random

client = carla.Client('localhost', 2000)
world = client.get_world()

client.load_world('Town05')

spectator = world.get_spectator()

transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation

# Set the spectator with an empty transform
spectator.set_transform(carla.Transform())

vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

spawn_points = world.get_map().get_spawn_points()
for i in range(0,500):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))


ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
