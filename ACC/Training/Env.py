from __future__ import annotations

import logging
import math
import random
import time
from typing import SupportsFloat, Any, Dict
import carla
import gymnasium as gym
import numpy as np
import wandb
from gymnasium.core import RenderFrame

from ACC.Engine.engine import Engine, SingletonLightState
from ACC.Utils.abstractions import ActionsEnum, LightColors
from ACC.Utils.abstractions import VehicleState
from ACC.Utils.Sensors import CarlaWorldStateSensor
from ACC.Engine.scenario import Scenario
from gymnasium import spaces
import traceback

from mushroom_rl.core import MDPInfo

from typing import Optional

def vehicle_state_to_array(state: VehicleState) -> np.ndarray:


    #normilze
    norm_speed = state.speed_ms / 35
    speed_ratio = state.speed_ms / (state.speed_limit_ms + 1e-5)
    norm_limit = state.speed_limit_ms / 35

    norm_distance = np.clip(state.lead_distance_m / 250.0, 0.0, 1.0)

    norm_safe_dist = state.safe_following_distance_m / 150.0

    norm_light = float(state.light_color.value) / 2.0
    norm_light_dist = float(state.light_dist_m) / 250

    norm_light_speed =np.clip(state.light_speed_ms * 3.6 / 30, -3, 3)
    norm_speed_lead = np.clip(state.relative_speed_ms * 3.6 / 30, -3, 3)

    norm_acc_ego = np.clip(state.g_force_ego / 5, -1.5, 1.5)

    norm_crash = np.clip(state.crash_intensity / 200000, 0, 3)

    obs = np.array([
        norm_speed,
        norm_limit,
        speed_ratio,
        norm_distance,
        norm_safe_dist,
        norm_crash,
        norm_light,
        norm_light_dist,
        norm_light_speed,
        norm_speed_lead,
        norm_acc_ego
    ], dtype=np.float32)

    return obs


def array_to_action(action: np.ndarray) -> Dict[ActionsEnum, float]:
    val = float(action[0])

    throttle = 0.0
    brake = 0.0

    if val >= -0.5:
        throttle = (val + 0.5) / 1.5
    else:
        brake = abs(val + 0.5) / 0.5

    return {
        ActionsEnum.throttle: throttle,
        ActionsEnum.brake: brake,
    }


class CarlaEnv(gym.Env[VehicleState, Dict[ActionsEnum, float]]):

    def __init__(self, args, scene):
        super().__init__()
        self.eng_args = None
        self.eng_scene: Optional[Scenario] = None


        self.engine = Engine(args, scene)
        self.engine.connect_to_worlds()
        if not self.engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")


        self.sensor_real = CarlaWorldStateSensor(self.engine.ego.real, self.engine.duo_world.get_real_world())

        self.eng_args = args
        self.eng_scene: Optional[Scenario] = scene

        # Complete observation space
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0,  # speed
                 0.0,  # speed_limit
                 0.0,  # speed_ratio
                 0.0,  # distance to lead
                 0.0,  # safe_following_distance
                 0,  # crash power
                 0.0,  # light_color (0, 1, 2)
                 0.0,  #light distance
                 -3, # norm_light_speed
                 -3,  # norm_speed_lead
                 -1.5], # norm_acc_ego
                dtype=np.float32
            ),
            high=np.array(
                [1.5,  # speed
                 1.5,  # speed_limit
                 1.5,  # speed ratio
                 1.0,  # distances (max 250m)
                 1.0,  # safe_following_distance
                 3,  # crash power
                 1.0, # light_color
                 1.0, # light_dist (max 250m)
                 3, # norm_light_speed
                 3, # norm_speed_lead
                 1.5], # norm_acc_ego
                dtype=np.float32
            ),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32,
        )

        #self.__g_force_calculator = GForceCalculator(self.engine.delta_seconds)

        # MDPInfo for Mushroom-RL compatibility
        self._mdp_info = MDPInfo(
            observation_space=self.observation_space,
            action_space=self.action_space,
            gamma=0.99
            ,horizon=int(self.eng_args.horizon)
        )

        self.lead_speed_limit = 0
        self.light_check_done = False

    @property
    def info(self):
        return self._mdp_info

    def set_rewards(self, reward_crash=True, reward_geforce=True, reward_speed_limit=True, reward_safe_distance=True, reward_light=True):

        if self.eng_scene is not None:
            self.eng_scene.rewards["reward_crash"] = reward_crash
            self.eng_scene.rewards["reward_geforce"] = reward_geforce
            self.eng_scene.rewards["reward_speed_limit"] = reward_speed_limit
            self.eng_scene.rewards["reward_safe_distance"] = reward_safe_distance
            self.eng_scene.rewards["reward_light"] = reward_light





    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> \
            tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)


        current_speed_limit = 0.0
        counter = 0.0
        if self.sensor_real is not None:
            current_speed_limit = self.sensor_real.speed_limit
            counter = self.sensor_real.counter
            try:
                self.sensor_real.cleanup()
            except Exception as e:
                logging.warning(f"Sensor cleanup error: {e}")

            self.sensor_real = None

        reset_success = False
        if self.engine is not None:
            try:
                reset_success = self.engine.soft_reset()
            except Exception as e:
                logging.warning(f"Soft reset failed: {e}")
                reset_success = False

        if not reset_success:
            logging.warning("Soft reset failed, performing full reset...")

            if self.engine is not None:
                self.eng_args = self.engine.args
                self.eng_scene = self.engine.scenario
                try:
                    self.engine.cleanup()
                except Exception as e:
                    logging.warning(f"Engine cleanup error: {e}")
                self.engine = None

            time.sleep(3)

            self.engine = Engine(self.eng_args,self.eng_scene)
            self.engine.connect_to_worlds()
            self.engine.duo_world.tick()

            if not self.engine.setup():
                raise RuntimeError("Engine setup failed. Exiting.")

        self.sensor_real = CarlaWorldStateSensor(
            self.engine.ego.real,
            self.engine.duo_world.get_real_world()
        )
        self.sensor_real.counter = counter

        if self.eng_args.random_speed_limit:
            self.sensor_real.override_speed_limit = True
            if current_speed_limit != 0.0:
                self.sensor_real.speed_limit = current_speed_limit

        state = self.sensor_real.get_state()
        state.steering_dir = 0.0
        obs = vehicle_state_to_array(state)
        info: Dict[str, Any] = {}

        return obs, info

    def close(self):
        if self.sensor_real is not None:
            try:
                self.sensor_real.cleanup()
            except Exception as e:
                logging.error(f"Error cleaning up sensor in close: {e}")
            self.sensor_real = None

        if self.engine is not None:
            try:
                self.engine.cleanup(True)
            except Exception as e:
                logging.error(f"Error cleaning up engine in close: {e}")
            self.engine = None

        try:
            super().close()
        except Exception:
            pass


    def _reward(self, state : VehicleState) -> tuple[float, Dict]:

        #self.engine.ego.real
        #self.__g_force_calculator.update_speed(state.speed)
        #g_force = self.__g_force_calculator.get_latest_g_force()
        # TODO change light stop dist to 25m

        rewards_dict = self.eng_scene.rewards
        #print(rewards_dict)
        use_crash = rewards_dict.get("reward_crash", True)
        use_geforce = rewards_dict.get("reward_geforce", False)
        use_speed = rewards_dict.get("reward_speed_limit", True)
        use_dist = rewards_dict.get("reward_safe_distance", True)
        use_light = rewards_dict.get("reward_light", True)

        W_SPEED = 2.5
        W_DIST = 1.2
        W_COMFORT = 0.8
        W_LIGHT = 1.3

        P_CRASH_BASE = -20.0

        r_crash = 0.0
        r_speed = 0.0
        r_dist = 0.0
        r_comfort = 0.0
        r_light = 0.0

        DANGER_ZONE_START = 6
        DANGER_ZONE_END = 2.5

        ############### get distance to light or lead ###############
        if state.light_color in [LightColors.red, LightColors.orange]:
            min_obstacle_dist = min(state.lead_distance_m, state.light_dist_m)
        else:
            min_obstacle_dist = state.lead_distance_m

        ############### CRASH ###############
        if use_crash:
            is_red_violation = state.light_dist_m < 0.01 and (state.light_color in [LightColors.red])
            if (min_obstacle_dist < DANGER_ZONE_START and state.speed_ms > 1) or (min_obstacle_dist < DANGER_ZONE_END) or is_red_violation:

                penetration = max(0, (DANGER_ZONE_START - min_obstacle_dist)) / DANGER_ZONE_START

                r_crash = -2.0 * penetration * 4
                if state.crash_intensity > 0.0 or is_red_violation:
                    intensity_penalty = min(state.crash_intensity / 50000, 1.0) if state.crash_intensity > 0 else 1.0
                    r_crash = P_CRASH_BASE - intensity_penalty
                    if is_red_violation:
                        logging.info(f"Red Light! ({r_crash})")
                    else:
                        logging.info(f"Car Crashed! ({r_crash})")

            elif min_obstacle_dist < DANGER_ZONE_START and state.speed_ms <= 2:
                r_crash = -0.5 * state.speed_ms




        ############### G-FORCE ###############
        g_force_ego = state.g_force_ego
        if use_geforce:
            if g_force_ego is not None: # https://www.sciencedirect.com/science/article/pii/S0003687022002046?via%3Dihub
                if min_obstacle_dist < DANGER_ZONE_START:
                    if state.g_force_ego > 0:
                        r_comfort = -1.0 * (g_force_ego / 0.12) ** 2
                    else:
                        r_comfort = -1.0 * (g_force_ego / 0.25) ** 2
                else:
                    r_comfort = -1.0 * (g_force_ego / 0.12) ** 2
                r_comfort = max(r_comfort, -1.5)

        ############### TARGET SPEED CALCULATION ###############
        target_speed_ms = state.speed_limit_ms

        # Adjust target for red/orange lights

        if use_light and state.light_color in [LightColors.red, LightColors.orange]:
            if min_obstacle_dist < 20:
                dist_to_stop = max(0, min_obstacle_dist - DANGER_ZONE_START)
                safe_approach_speed = math.sqrt(2 * 0.56 * dist_to_stop)
                target_speed_ms = min(target_speed_ms, safe_approach_speed)


        # Adjust target for lead vehicle
        if use_dist and state.lead_distance_m is not None and state.lead_distance_m < 150:
            if state.lead_distance_m < state.safe_following_distance_m * 1.5:
                lead_speed = state.speed_ms + state.relative_speed_ms
                target_speed_ms = min(target_speed_ms, lead_speed)

        if min_obstacle_dist <= DANGER_ZONE_START:
            target_speed_ms = 0.0
        else:
            target_speed_ms = max(target_speed_ms, 2) # don't stop util you're at a good place to stop

        target_speed_ms = max(0.0, target_speed_ms)

        ############### SPEED ###############
        diff_kmh = 0
        if use_speed:
            diff_kmh = (state.speed_ms - target_speed_ms) * 3.6

            diff_kmh = min(max(diff_kmh, -150), 150)
            if diff_kmh > 0:
                r_speed = 1.75 * math.exp(-0.5 * (diff_kmh ** 2))- 0.25 - 0.01 * diff_kmh
            else:
                r_speed = math.exp(-0.5 * (diff_kmh / 5.0) ** 2) + 0.01 * diff_kmh

            if abs(diff_kmh) < 2.0:
                exact_speed_bonus = 10 * math.exp(-0.5*(diff_kmh ** 2))
                r_speed = max(r_speed, exact_speed_bonus)

        ############### SAFE DISTANCE ###############
        ratio = 0.0
        if use_dist:
            safe_dist = max(state.safe_following_distance_m, 5.0)
            ratio = min_obstacle_dist / safe_dist

            if ratio > 5 and min_obstacle_dist > 10 and state.speed_ms < target_speed_ms:
                r_dist = (1 + ratio / 100) * r_speed
            else:
                r_dist = math.tanh(ratio - 1.0)

        ############### TRAFFIC LIGHT ###############
        if use_light:
            if state.light_color == LightColors.green:
                if state.speed_ms >= target_speed_ms * 0.7 and ratio > 1 :
                    r_light = 0.01

        r_speed = W_SPEED * r_speed
        r_dist = W_DIST * r_dist
        r_comfort = W_COMFORT * r_comfort
        r_light = W_LIGHT * r_light

        total_reward = r_speed + r_dist + r_comfort + r_light + r_crash

        total_reward = max(min(total_reward, 20.0), -30.0)

        components = {
            "Reward/Total": total_reward,
            "Reward/r_crash": r_crash,
            "Reward/r_comfort": r_comfort,
            "Reward/Speed": r_speed,
            "Reward/Distance": r_dist,
            "Reward/Lights": r_light,
            "State/distance/safety_margin": ratio,
            "State/safety/target_speed_kmh": target_speed_ms * 3.6,
            "State/safety/min_obstacle_dist": min_obstacle_dist,
            "State/safety/diff_kmh": diff_kmh,
            "State/safety/correct speed ratio": min(state.speed_ms/(target_speed_ms + 1e-3), 250),

            "State/VehicleState/speed_ms": state.speed_ms,
            "State/VehicleState/speed_limit_ms": state.speed_limit_ms,
            "State/VehicleState/lead_distance_m": state.lead_distance_m,
            "State/VehicleState/safe_following_distance_m": state.safe_following_distance_m,
            "State/VehicleState/crash_intensity": state.crash_intensity,
            "State/VehicleState/light_color": state.light_color.value,
            "State/VehicleState/light_dist_m": state.light_dist_m,
            "State/VehicleState/light_speed_ms": state.light_speed_ms,
            "State/VehicleState/g_force_ego": state.g_force_ego,
            "State/VehicleState/relative_speed_ms": state.relative_speed_ms
        }

        return total_reward, components

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, dict[ActionsEnum, float]]]:
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0

        try:

            if not self.engine.ego.is_alive():
                logging.debug("Ego pair incomplete. Attempting revival...")
                if self.engine.revive_ego_pair():
                    logging.debug("Ego successfully revived. Continuing step.")
                else:
                    raise RuntimeError("Ego vehicle disappeared and revival failed.")

            if self.engine.lead is not None:
                if not self.engine.lead.is_alive():
                    logging.debug("Lead pair incomplete. Attempting revival...")
                    if self.engine.revive_lead_pair(self.lead_speed_limit):
                        logging.debug("Lead successfully revived. Continuing step.")
                    else:
                        raise RuntimeError("Lead vehicle disappeared and revival failed.")



            # apply control
            tm_control = self.engine.ego.get_mirror_control()
            steering_dir = tm_control.steer
            action = array_to_action(action)

            #logging.info(f"throttle: {action[ActionsEnum.throttle]}, brake: {action[ActionsEnum.brake]}")

            final_control = carla.VehicleControl(
                throttle=action[ActionsEnum.throttle],
                brake=action[ActionsEnum.brake],
                steer=steering_dir,
                hand_brake=False,
                reverse=tm_control.reverse,
                manual_gear_shift=tm_control.manual_gear_shift,
                gear=tm_control.gear
            )

            self.engine.ego.apply_real_control(final_control)

            # apply goal
            if self.engine.lead is not None:
                self.engine.tm_mirror.set_path(self.engine.ego.mirror, [self.engine.lead.mirror.get_location()])

            # synchronization real npc with mirror npcs
            self.engine.synchronization_real_npc_with_mirror_npcs()

            # synchronization mirror ego with real ego
            self.engine.synchronization_mirror_ego_with_real_ego()

            # spectator
            self.engine.update_spectator()

            # lights
            self.engine.sync_traffic_lights()


            state = self.sensor_real.get_state()

            if state.crash_intensity > 0.0:
                logging.info("Car Crashed!")
                terminated = True
            state.steering_dir = steering_dir

            if state.light_color in [LightColors.red] and state.light_dist_m < 2.5 and state.speed_ms > 0.5:
                logging.info(f"Red Light Violation! Dist: {state.light_dist_m:.2f}")
                terminated = True

            reward, reward_components = self._reward(state)

            if self.engine.lead is not None:
                v = self.engine.lead.mirror.get_velocity() * 3.6
                lead_speed = v.length()
                reward_components["State/LeadSpeed"] = lead_speed
            else:
                reward_components["State/LeadSpeed"] = 0.0

            info.update(reward_components)


            obs = vehicle_state_to_array(state)

            self.engine.duo_world.tick()


            #this creates an infinite road to drive on
            if "CUSTOM_STRAIGHT" in self.engine.map_name:
                transform = self.engine.ego.real.get_transform()

                if self.engine.lead is not None and self.engine.lead.is_alive():
                    lead_transform = self.engine.lead.mirror.get_transform()
                    dist_to_lead = lead_transform.location.x - transform.location.x

                    if lead_transform.location.x > 1700:
                        new_location = carla.Location(x=1550, y=lead_transform.location.y, z=lead_transform.location.z)
                        self.engine.lead.real.set_location(new_location)
                        self.engine.lead.mirror.set_location(new_location)

                    if lead_transform.location.x > 500 and not self.light_check_done:
                        self.light_check_done = True
                        if random.random() < 0.25:
                            self.lead_speed_limit = 0.0
                            logging.info(f"Speed Limit Lead {self.lead_speed_limit}!")

                        self.engine.tm_mirror.set_desired_speed(self.engine.lead.mirror, self.lead_speed_limit)

                    if dist_to_lead > 300:
                        target_catchup_speed = max(0.0, state.speed_ms * 3.6 * random.uniform(0.5, 0.8))

                        if random.random() < 0.25:
                            target_catchup_speed = 0.0

                        # Only update if meaningful change to avoid spamming TM
                        if abs(self.lead_speed_limit - target_catchup_speed) > 1.0:
                            self.lead_speed_limit = target_catchup_speed
                            self.engine.tm_mirror.set_desired_speed(self.engine.lead.mirror, self.lead_speed_limit)
                            logging.info(f"Speed Limit Lead {self.lead_speed_limit}!")

                if transform.location.x > 1000:
                    new_location = carla.Location(x=10, y=transform.location.y, z=transform.location.z)
                    self.engine.ego.real.set_location(new_location)
                    self.engine.ego.mirror.set_location(new_location)
                    if self.engine.lead is not None:
                        self.light_check_done = False
                        lead_transform = self.engine.lead.real.get_transform()
                        distances_to_ego = lead_transform.location.x - transform.location.x
                        lead_location = carla.Location(x=distances_to_ego + new_location.x, y=lead_transform.location.y, z=lead_transform.location.z)
                        self.engine.lead.real.set_location(lead_location)
                        self.engine.lead.mirror.set_location(lead_location)

                        speed_limit_kh = int(state.speed_limit_ms * 3.6)

                        if random.random() < 0.75:
                            self.lead_speed_limit = random.randint(speed_limit_kh, speed_limit_kh + 30)
                            logging.info(f"Speed Limit Lead {self.lead_speed_limit}!")
                        else:
                            self.lead_speed_limit = random.randint(0, speed_limit_kh)
                            logging.info(f"Speed Limit Lead {self.lead_speed_limit}!")

                        if random.random() < 0.25:
                            logging.info(f"Speed Limit Lead Zero!")
                            self.lead_speed_limit = 0.0

                        if random.random() < 0.50:
                            logging.info(f"Lights Inverse!")
                            SingletonLightState().set_inverse_state(True)
                        else:
                            SingletonLightState().set_inverse_state(False)


                        self.engine.tm_mirror.set_desired_speed(self.engine.lead.mirror, self.lead_speed_limit)





        except Exception as e:
            print(f"\nAn critical error occurred during step in traing env: {e}")
            traceback.print_exc()
            terminated = True
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


class GymnasiumToGymWrapper:

    def __init__(self, env: CarlaEnv):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.step_count = 0

    @property
    def info(self):
        return self.env._mdp_info

    def reset(self, state=None):
        """Reset with extended retry logic and better error recovery."""
        max_retries = 10

        for attempt in range(max_retries):
            try:
                obs, _ = self.env.reset()

                # Verify ego is actually alive
                if self.env.engine.ego is not None and self.env.engine.ego.is_alive():
                    return obs

                logging.warning(f"Reset attempt {attempt + 1}: Ego is dead or None. Retrying...")

            except Exception as e:
                logging.warning(f"Reset attempt {attempt + 1} crashed: {e}. Retrying...")

                # Try to cleanup the broken state
                if hasattr(self.env, 'engine') and self.env.engine is not None:
                    try:
                        self.env.engine.cleanup()
                    except:
                        pass
                    self.env.engine = None

                # Increasing delay between retries
                time.sleep(2.0 + attempt)

        raise RuntimeError(f"Critical: Failed to reset environment after {max_retries} attempts.")


    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated

        if wandb.run is not None:
            log_dict = {k: v for k, v in info.items() if isinstance(v, (int, float))}
            log_dict['global_step'] = self.step_count
            wandb.log(log_dict)

        self.step_count += 1

        return obs, reward, terminated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


    def stop(self):
        return self.env.close()

