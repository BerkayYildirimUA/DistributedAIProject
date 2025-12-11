# gt_front_counter.py
import math
import carla
from typing import Dict, Optional, Tuple
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import constants

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
        max_distance: float = constants.MAX_LEAD_ACTOR_DISTANCE
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

        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.pedestrian.*')
        traffic_lights = actors.filter('traffic.traffic_light*')
        speed_signs = actors.filter('traffic.speed_limit.*')

        def in_front(actor):
            # Don’t count ego itself
            if actor.id == self.ego_vehicle.id:
                return False
            return self._is_in_front_and_wigthin_range(
                ego_transform, actor.get_location()
            )

        num_vehicles = sum(1 for v in vehicles if in_front(v))
        num_pedestrians = sum(1 for w in pedestrians if in_front(w))
        num_tlights = sum(1 for t in traffic_lights if in_front(t))
        num_ssigns = sum(1 for s in speed_signs if in_front(s))

        total = num_vehicles + num_pedestrians + num_tlights

        return {
            "vehicles": num_vehicles,
            "pedestrians": num_pedestrians,
            "traffic_lights": num_tlights,
            "speed_signs": num_ssigns,
            "total": total,
        }

    def get_lead_actor_in_lane(
            self,
            max_distance: float = constants.MAX_LEAD_ACTOR_DISTANCE,
    ) -> Tuple[Optional[carla.Actor], Optional[float]]:
        """
        Returns the closest obstacle (vehicle or pedestrian) in front of ego
        in the same lane, within max_distance.

        "Obstacle" = any actor in ['vehicle.*', 'walker.pedestrian.*'].
        "Same lane" is defined by same road_id and lane_id of the waypoint
        with lane_type=Driving.

        If no such obstacle exists, returns (None, None).
        """
        carla_map = self.world.get_map()

        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()

        ego_wp = carla_map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            return None, None  # ego is off-road somehow

        actors = self.world.get_actors()
        vehicles = actors.filter("vehicle.*")
        pedestrians = actors.filter("walker.pedestrian.*")

        # Combine both into a single iterable of obstacles
        obstacles = list(vehicles) + list(pedestrians)

        best_actor: Optional[carla.Actor] = None
        best_distance: float = max_distance

        for a in obstacles:
            if a.id == self.ego_vehicle.id:
                continue

            a_loc = a.get_location()
            a_wp = carla_map.get_waypoint(
                a_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if a_wp is None:
                continue

            # Same road and lane → same lane in the map sense
            if a_wp.road_id != ego_wp.road_id or a_wp.lane_id != ego_wp.lane_id:
                continue

            dx = a_loc.x - ego_loc.x
            dy = a_loc.y - ego_loc.y
            dz = a_loc.z - ego_loc.z

            proj = dx * ego_fwd.x + dy * ego_fwd.y + dz * ego_fwd.z
            if proj <= 0.0:
                # Behind or exactly lateral
                continue

            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < best_distance:
                best_distance = dist
                best_actor = a

        if best_actor is None:
            return None, None

        return best_actor, best_distance

