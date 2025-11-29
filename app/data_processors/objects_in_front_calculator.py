# gt_front_counter.py
import math
import carla
from typing import Dict, Optional, Tuple


class ObjectsInFrontCalculator:
    """
    Counts ground-truth objects (vehicles, pedestrians, traffic lights, ...)
    in front of the ego vehicle within a given distance.

    "In front" = positive projection along ego forward vector.
    Distance = Euclidean distance in world coordinates.
    """

    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        max_distance: float = 20.0
    ) -> None:
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.max_distance = max_distance

    def _is_in_front_and_within_range(
        self,
        ego_transform: carla.Transform,
        actor_location: carla.Location
    ) -> bool:
        ego_loc = ego_transform.location
        forward = ego_transform.get_forward_vector()

        dx = actor_location.x - ego_loc.x
        dy = actor_location.y - ego_loc.y
        dz = actor_location.z - ego_loc.z

        # Projection along ego forward
        front_proj = dx * forward.x + dy * forward.y + dz * forward.z
        if front_proj <= 0.0:
            return False  # behind or exactly lateral

        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq > self.max_distance * self.max_distance:
            return False

        return True

    def count_objects_in_front(self) -> Dict[str, int]:
        """
        Returns a dict like:
        {
            'vehicles': v,
            'pedestrians': p,
            'traffic_lights': tl,
            'total': t,
        }
        """
        ego_transform = self.ego_vehicle.get_transform()
        actors = self.world.get_actors()

        # You can refine filters depending on your CARLA version/blueprints
        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.pedestrian.*')
        traffic_lights = actors.filter('traffic.traffic_light*')

        def in_front(actor):
            # Don’t count ego itself
            if actor.id == self.ego_vehicle.id:
                return False
            return self._is_in_front_and_within_range(
                ego_transform, actor.get_location()
            )

        num_vehicles = sum(1 for v in vehicles if in_front(v))
        num_pedestrians = sum(1 for w in pedestrians if in_front(w))
        num_tlights = sum(1 for t in traffic_lights if in_front(t))

        total = num_vehicles + num_pedestrians + num_tlights

        return {
            "vehicles": num_vehicles,
            "pedestrians": num_pedestrians,
            "traffic_lights": num_tlights,
            "total": total,
        }

    def get_lead_vehicle_in_lane(self, max_distance: float = 60.0,) -> Tuple[Optional[carla.Vehicle], Optional[float]]:
        """
        Returns the closest vehicle in front of ego in the same lane, within max_distance.

        If no such vehicle exists, returns (None, None).

        "Same lane" is defined by same road_id and lane_id of the waypoint
        with lane_type=Driving.
        """
        carla_map = self.world.get_map()

        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()

        ego_wp = carla_map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if ego_wp is None:
            return None, None  # ego is off-road somehow

        vehicles = self.world.get_actors().filter("vehicle.*")

        best_vehicle: Optional[carla.Vehicle] = None
        best_distance: float = max_distance

        for v in vehicles:
            if v.id == self.ego_vehicle.id:
                continue

            v_loc = v.get_location()
            v_wp = carla_map.get_waypoint(
                v_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if v_wp is None:
                continue

            # Same road and lane → we're in the same lane in the map sense
            if v_wp.road_id != ego_wp.road_id or v_wp.lane_id != ego_wp.lane_id:
                continue

            # Check if it's ahead of ego (projection along ego forward vector)
            dx = v_loc.x - ego_loc.x
            dy = v_loc.y - ego_loc.y
            dz = v_loc.z - ego_loc.z
            proj = dx * ego_fwd.x + dy * ego_fwd.y + dz * ego_fwd.z

            if proj <= 0.0:
                # Behind or exactly lateral
                continue

            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < best_distance:
                best_distance = dist
                best_vehicle = v

        if best_vehicle is None:
            return None, None

        return best_vehicle, best_distance
