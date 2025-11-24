import carla
from typing import Optional


class DuoWorld:

    def __init__(self, real_world: carla.World, mirror_world: carla.World):
        self.__real_world = real_world
        self.__mirror_world = mirror_world

    def set_mirror_world_settings(self, settings: carla.WorldSettings):
        self.__mirror_world.apply_settings(settings)

    def set_real_world_settings(self, settings: carla.WorldSettings):
        self.__real_world.apply_settings(settings)

    def set_both_worlds_settings(self, settings: carla.WorldSettings):
        self.set_mirror_world_settings(settings)
        self.set_real_world_settings(settings)

    def get_real_world(self) -> carla.World:
        return self.__real_world

    def get_mirror_world(self) -> carla.World:
        return self.__mirror_world

    def tick(self):
        return self.__mirror_world.tick(), self.__real_world.tick()

class DuoActor:

    def __init__(self, real: carla.Actor, mirror: carla.Actor):
        self.real = real
        self.mirror = mirror

    def set_mirror_transform(self, transform: carla.Transform):
        self.mirror.set_transform(transform)

    def set_real_transform(self, transform: carla.Transform):
        self.real.set_transform(transform)

    def set_both_transform(self, transform: carla.Transform):
        self.set_mirror_transform(transform)
        self.set_real_transform(transform)

    def get_mirror_transform(self) -> Optional[carla.Transform]:
        return self.mirror.get_transform()

    def get_real_transform(self) -> Optional[carla.Transform]:
        return self.real.get_transform()

    def get_mirror_control(self) -> Optional[carla.VehicleControl]:
        return self.mirror.get_control()

    def apply_mirror_control(self, control: carla.VehicleControl):
        self.mirror.apply_control(control)

    def apply_real_control(self, control: carla.VehicleControl):
        self.real.apply_control(control)

    def set_mirror_autopilot(self, enable: bool, tm_port: int):
        self.mirror.set_autopilot(enable, tm_port)

    def set_real_autopilot(self, enable: bool, tm_port: int):
        self.real.set_autopilot(enable, tm_port)

    def set_mirror_physics(self, enable: bool):
        self.mirror.set_simulate_physics(enable)

    def set_real_physics(self, enable: bool):
        self.real.set_simulate_physics(enable)

    def get_velocity(self):
        return self.real.get_velocity()

    def get_angular_velocity(self):
        return self.real.get_angular_velocity()

    def set_mirror_velocity(self, velocity: carla.Vector3D):
        self.mirror.set_target_velocity(velocity)

    def set_mirror_angular_velocity(self, angular_velocity: carla.Vector3D):
        self.mirror.set_target_angular_velocity(angular_velocity)

    def destroy(self):
        if self.real:
            try:
                if self.real.is_alive:
                    self.real.destroy()
            except RuntimeError:
                pass  # It's already dead, who cares.
            self.real = None

        if self.mirror:
            try:
                if self.mirror.is_alive:
                    self.mirror.destroy()
            except RuntimeError:
                pass  # It's already dead.
            self.mirror = None



    def is_alive(self) -> bool:
        real_alive = self.real is not None and self.real.is_alive
        mirror_alive = self.mirror is not None and self.mirror.is_alive
        return real_alive and mirror_alive

class DuoClient:
    def __init__(self, client_real: carla.Client, client_mirror: carla.Client):
        self.real = client_real
        self.mirror = client_mirror
