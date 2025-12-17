# https://mushroomrl.readthedocs.io/en/latest/?badge=latest
import logging

import carla

import wandb
#
from ACC.Training.Env import CarlaEnv, GymnasiumToGymWrapper, vehicle_state_to_array, array_to_action
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import TD3
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.dataset import compute_J
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict
from ACC.Engine.scenario import Scenario
import datetime
from mushroom_rl.policy import DeterministicPolicy
import os
import numpy as np
from mushroom_rl.core import Agent

from ACC.Utils.Logger import LossLogger
from ACC.Utils.abstractions import AbstractDecisionAgent, StateSensor
from ACC.Utils.abstractions import ActionsEnum
from ACC.Utils.abstractions import VehicleState

class TD3Config:
    # Training
    BATCH_SIZE = 256  #256
    LR_ACTOR = 3.5e-4 #1e-3
    LR_CRITIC = 6.0e-3 #3e-4
    TAU = 0.00126 #0.005
    POLICY_DELAY = 2 #2
    NOISE_STD = 0.21 #0.1
    NOISE_CLIP = 0.39 #0.2

    # Replay Buffer
    INITIAL_REPLAY_SIZE = 1000
    MAX_REPLAY_SIZE = 100_000

    # Timing / steps logic
    LOOPS_PER_SECOND = int(110) #something on my PC specifically

class TD3ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[0]
        n_output = output_shape[0]

        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_output),
            nn.Tanh()  # Forces output to [-1, 1]
        )

    def forward(self, state, **kwargs):
        return self.net(state)


class TD3CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[0]
        n_action = output_shape[0]

        self.net = nn.Sequential(
            nn.Linear(n_input + n_action, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action, **kwargs):
        # Concatenate state and action: [Speed, Dist, ... Throttle]
        x = torch.cat((state, action.float()), dim=1)
        return self.net(x).squeeze(1)


class ACC_TD3Agent():

    def __init__(self, args, load_model_name=None, scene=None):
        self.args = args
        self.dataset_callback = CollectDataset()
        self.env = None
        self.agent = None
        self.core = None

        # ---- Path setup ----
        self.project_root = self._get_project_root()
        self.models_dir = os.path.join(self.project_root, "ACC", "Agents", "models")
        os.makedirs(self.models_dir, exist_ok=True)
        if load_model_name is not None:
            load_model_path = os.path.join(self.models_dir, load_model_name)
        else:
            load_model_path = None

        if scene is None:
            self.scene = Scenario(
                'vehicle.tesla.model3',
                delta_seconds=self.args.delta_seconds,
                map_name=self.args.map,
                number_of_npc=0,
                lead_car_bp_name="vehicle.tesla.model3"
            )
        else:
            self.scene = scene

        # ---- WANDB ----
        if wandb.run is not None:
            wandb.finish()

        run_name = scene.rewards.get("name", "Unknown_Scenario") if scene else "ACC_Agent"
        if scene and hasattr(scene, 'name'):  # If scenario has a name attribute
            run_name = scene.name

        wandb.init(
            project="CARLA_ACC_Training",
            name=f"{run_name}_{datetime.datetime.now().strftime('%H%M')}",
            config={
                "delta_seconds": args.delta_seconds,
                "map": args.map,
                "lr_actor": TD3Config.LR_ACTOR,
                "lr_critic": TD3Config.LR_CRITIC,
                "batch_size": TD3Config.BATCH_SIZE,
                "scenario": str(scene.__dict__) if scene else "None"
            }
        )

        # ---- Initialize ----
        self._setup_env()
        if load_model_path is not None:
            self._setup_agent(load_model_path)
        else:
            self._setup_agent()
        self._setup_core()

    def _get_project_root(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(script_dir))

    def _setup_env(self):
        """Initialize the Carla Environment and Wrappers."""
        logging.info("Initializing Environment...")

        raw_env = CarlaEnv(self.args, self.scene)
        #raw_env.set_rewards(reward_geforce=False, reward_safe_distance=False)
        self.env = GymnasiumToGymWrapper(raw_env)

    def _setup_agent(self, load_model_path=None):
        logging.info("Initializing TD3 Agent...")

        actor_params = dict(
            network=TD3ActorNetwork,
            input_shape=self.env.observation_space.shape,
            output_shape=self.env.action_space.shape
        )

        critic_params = dict(
            network=TD3CriticNetwork,
            optimizer={'class': optim.AdamW, 'params': {'lr': TD3Config.LR_CRITIC}},
            loss=nn.MSELoss(),
            input_shape=self.env.observation_space.shape,
            output_shape=self.env.action_space.shape
        )

        self.agent = TD3(
            mdp_info=self.env.info,
            policy_class=DeterministicPolicy,
            policy_params={},
            actor_params=actor_params,
            actor_optimizer={'class': optim.AdamW, 'params': {'lr': TD3Config.LR_ACTOR}},
            critic_params=critic_params,
            batch_size=TD3Config.BATCH_SIZE,
            initial_replay_size=TD3Config.INITIAL_REPLAY_SIZE,
            max_replay_size=TD3Config.MAX_REPLAY_SIZE,
            tau=TD3Config.TAU,
            policy_delay=TD3Config.POLICY_DELAY,
            noise_std=TD3Config.NOISE_STD,
            noise_clip=TD3Config.NOISE_CLIP
        )

        if load_model_path is not None and os.path.exists(load_model_path):
            logging.info(f"Loading model from: {load_model_path}")
            self.agent = self.agent.load(load_model_path)
        elif load_model_path is not None:
            logging.warning(f"Model path {load_model_path} not found! Starting from scratch.")

        self.loss_logger = LossLogger(self.agent)


    def reset_buffer(self):
        if hasattr(self.agent, '_replay_memory'):
            logging.warning("Reward function changed: Wiping old Replay Memory to avoid contamination.")
            self.agent._replay_memory.reset()

    def _setup_core(self):
        self.core = Core(self.agent, self.env, callbacks_fit=[self.dataset_callback, self.loss_logger])

    def train(self, duration_seconds=None):
        if duration_seconds is None:
            duration_seconds = 12 * 60 * 60
        n_steps = int(TD3Config.LOOPS_PER_SECOND * duration_seconds)

        logging.info(f"Starting training for {n_steps} steps...")
        self.core.learn(n_steps=n_steps, n_steps_per_fit=1)

        logging.info("Training complete.")


        dataset = self.dataset_callback.get()
        J_metrics = compute_J(dataset, self.env.info.gamma)
        current_avg_reward = np.mean(J_metrics)

        wandb.log({"results/avg_reward" : current_avg_reward})

        wandb.finish()

    def evaluate(self, n_steps=80000):
        """Evaluate the current policy."""
        logging.info("Evaluating...")

        # Reset dataset to get clean stats
        self.dataset_callback.clean()

        self.core.evaluate(n_steps=n_steps, render=False)

        # Retrieve data
        dataset = self.dataset_callback.get()
        J = compute_J(dataset, self.env.info.gamma)
        logging.info(f"Average Reward (J): {np.mean(J)}")

        self.save_model()

    def save_model(self, suffix="Exp_Speed_Reward"):
        timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        filename = f'{timestamp}_TD3_{suffix}.msh'
        save_path = os.path.join(self.models_dir, filename)

        logging.info(f"Saving agent to {save_path}...")
        self.agent.save(save_path, full_save=True)

        return save_path

    def close(self):
        if self.env:
            self.env.close()


class RLDecisionAgent(AbstractDecisionAgent):
    """
    Wrapper that connects a CARLA StateSensor to a pre-trained MushroomRL Agent.
    """

    def __init__(self, sensor: StateSensor, model_name: str):
        self.sensor = sensor


        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", model_name)


        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        logging.info(f"Loading RL Agent from {model_path}...")

        self.agent = Agent.load(model_path)

        # Set to eval mode
        #self.agent.policy
        logging.info("RL Agent loaded and set to Eval mode.")

    def make_decision(self, tm_control: carla.VehicleControl) -> carla.VehicleControl:

        data = self.sensor.get_state()
        data.steering_dir = tm_control.steer

        observation = vehicle_state_to_array(data)

        raw_action = self.agent.policy.draw_action(observation)
        action = array_to_action(raw_action)

        final_control = carla.VehicleControl(
            throttle=action[ActionsEnum.throttle],
            brake=action[ActionsEnum.brake],
            steer=tm_control.steer,  # Keep TM steering
            hand_brake=tm_control.hand_brake,
            reverse=tm_control.reverse,
            manual_gear_shift=tm_control.manual_gear_shift,
            gear=tm_control.gear
        )

        return final_control