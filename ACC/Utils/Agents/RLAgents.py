# https://mushroomrl.readthedocs.io/en/latest/?badge=latest


from ACC.Training.Env import CarlaEnv, GymnasiumToGymWrapper
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO, TD3
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.callbacks import CollectDataset
import torch.nn as nn
import torch.optim as optim
import torch

import argparse
import logging
import traceback
import numpy

from ACC.Engine.scenario import Scenario
from ACC.Engine.start_words import CarlaServerManager
import datetime
from mushroom_rl.policy import DeterministicPolicy

import os
#class RLagent()
class BiasedActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[0]
        n_output = output_shape[0]

        self.network = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_output),
            nn.Tanh()
        )



    def forward(self, x, **kwargs):
        return self.network(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        n_input = input_shape[0]
        self._model = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)

        )

    def forward(self, x, **kwargs):
        return self._model(x)


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

        # Critic takes STATE + ACTION as input
        self.net = nn.Sequential(
            nn.Linear(n_input + n_action, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs Q-value
        )

    def forward(self, state, action, **kwargs):
        # Concatenate state and action: [Speed, Dist, ... Throttle]
        x = torch.cat((state, action.float()), dim=1)
        return self.net(x).squeeze(1)




def train_loop(args):
    scene = Scenario('vehicle.tesla.model3', delta_seconds=args.delta_seconds,
                     map_name=args.map, number_of_npc=0)
    env = CarlaEnv(args, scene)

    env.set_rewards(reward_geforce=False, reward_safe_distance=False, reward_crash=False)

    env = GymnasiumToGymWrapper(env)

    episode_over = False
    total_reward = 0

    #policy = GaussianTorchPolicy(
    #    network=BiasedActorNetwork,
    #    input_shape=env.observation_space.shape,
    #    output_shape=env.action_space.shape,
    #    use_cuda=True
    #)
#
    #agent = PPO(
    #    mdp_info=env.info,
    #    policy=policy,
    #    actor_optimizer={'class': optim.AdamW, 'params': {'lr': 3e-4}},
    #    critic_params={
    #        'network': CriticNetwork,
    #        'optimizer': {'class': optim.AdamW, 'params': {'lr': 1e-3}},
    #        'loss': nn.MSELoss(),
    #        'input_shape': env.observation_space.shape,
    #        'output_shape': (1,)
    #    },
    #    n_epochs_policy=10,
    #    batch_size=64,
    #    eps_ppo=0.2,
    #    lam=0.95,
    #    ent_coeff=0.1
    #)
    actor_params = dict(
        network=TD3ActorNetwork,
        input_shape=env.observation_space.shape,
        output_shape=env.action_space.shape
    )

    # TD3 uses TWO critics (Twin) to prevent over-optimism
    critic_params = dict(
        network=TD3CriticNetwork,
        optimizer={'class': optim.AdamW, 'params': {'lr': 1e-3}},
        loss=nn.MSELoss(),
        input_shape=env.observation_space.shape,
        output_shape=env.action_space.shape
    )


    agent = TD3(
        mdp_info=env.info,
        policy_class=DeterministicPolicy,
        policy_params={},

        actor_params=actor_params,
        actor_optimizer={'class': optim.AdamW, 'params': {'lr': 1e-3}},
        critic_params=critic_params,
        batch_size=100,  # TD3 likes larger batches
        initial_replay_size=1000,  # Drive random for 1000 steps to fill memory
        max_replay_size=100000,  # Remember last 100k steps
        tau=0.005,  # Soft update factor (Standard for TD3)
        policy_delay=2,  # Update actor half as often as critic
        noise_std=0.1,  # Exploration noise (10% jitter)
        noise_clip=0.5  # Don't jitter more than 50%
    )



    timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    logger = Logger(log_name=f'{timestamp}_carla_ppo', results_dir='./logs')
    agent.set_logger(logger)

    collect_dataset = CollectDataset()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    model_path = os.path.join(project_root, "ACC", "Utils", "models", "251122_015813.msh")

    logger.info(f"loading from: {model_path}")
    agent = agent.load(model_path)
    core = Core(agent, env, callbacks_fit=[collect_dataset])



    #core.learn(n_steps=100000, n_steps_per_fit=2048)

    core.learn(n_steps=3000000, n_steps_per_fit=1)

    # Evaluate trained agent
    print("Evaluating...")
    core.evaluate(n_steps=20000, render=False)

    dataset = collect_dataset.get()

    rewards = [item[2] for item in dataset]

    print(f"Average reward: {numpy.mean(rewards)}")

    env.close()

    agent.save(f'./models/{timestamp}.msh', full_save=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA ACC Dual Simulation (Mirror TM Only)')

    # CARLA
    parser.add_argument(
        '--carla-path',
        required=True,  # Make it mandatory unless you have a reliable default
        help='Path to the CARLA executable (CarlaUE4.sh or CarlaUE4.exe)'
    )

    # Server Ports
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')

    # real ports
    parser.add_argument('--real-port', default=2000, type=int,
                        help='TCP port for the REAL CARLA server (default: 2000)')
    parser.add_argument('--real-stream-port', default=2001, type=int,
                        help='Streaming port for the REAL CARLA server (default: 2001)')

    # mirror ports
    parser.add_argument('--mirror-port', default=4000, type=int,
                        help='TCP port for the MIRROR CARLA server (default: 4000)')
    parser.add_argument('--mirror-stream-port', default=4001, type=int,
                        help='Streaming port for the MIRROR CARLA server (default: 4001)')

    # Traffic Manager Ports
    parser.add_argument('--tm-mirror-port', default=9000, type=int,
                        help='Port for MIRROR Traffic Manager (default: 9000)')

    # Simulation Settings
    parser.add_argument('--map', default='random', help='Map to load (should match both servers) (default: Town04)')
    parser.add_argument('--delta-seconds', default=0.05, type=float,
                        help='Fixed delta seconds for simulation (default: 0.05)')
    parser.add_argument('--num-npcs', default=2, type=int, help='Number of NPC vehicles to spawn (default: 2)')

    # Camera
    parser.add_argument('--width', default=1280, type=int, help='Camera image width (default: 1280)')
    parser.add_argument('--height', default=720, type=int, help='Camera image height (default: 720)')

    # training
    parser.add_argument('--do_train', default=False, type=bool, help='Train an RL agent or just run the sim')

    args = parser.parse_args()

    real_server_process = None
    mirror_server_process = None
    server_manager = None

    try:
        server_manager = CarlaServerManager(args.carla_path, args.host)
        server_manager.launch_servers(
            args.real_port,
            args.real_stream_port,
            args.mirror_port,
            args.mirror_stream_port
        )

        logging.info("Servers launched successfully. Starting main simulation loop...")
        print("-" * 30)

        if not args.do_train:
            print("skipppppp")
            # main_loop(args)
        train_loop(args)


    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

    finally:
        # --- Terminate Servers ---
        if server_manager:
            server_manager.terminate_servers()
        else:
            print("Server manager was not initialized, skipping server termination.")

    print("Script finished.")
