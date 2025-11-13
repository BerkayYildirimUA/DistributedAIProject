import argparse
import logging
import traceback

#import carla
#from carla import BlueprintLibrary

from ACC.Engine.scenario import Scenario
from ACC.Engine.start_words import CarlaServerManager
from ACC.Utils.Sensors import CarlaLeadStateSensor, CarlaWorldStateSensor
from ACC.Utils.Agents.SimpleAgent import SimpleAccAgent
from ACC.Engine.engine import Engine

def main_loop(args):
    scene = Scenario('vehicle.tesla.model3', delta_seconds=args.delta_seconds,
                     map_name=args.map, number_of_npc=args.num_npcs)
    engine = Engine(args, scene)

    try:
        engine.connect_to_worlds()

        if not engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")

        # sensor and agent Setup (Real World)
        if engine.lead is not None:
            sensor_real = CarlaLeadStateSensor(engine.ego.real, engine.lead.real)
        else:
            sensor_real = CarlaWorldStateSensor(engine.ego.real, engine.duo_world.get_real_world())

        decisionAgent = SimpleAccAgent(engine.ego.real, sensor_real)



        while True:
            mirror_frame, _ = engine.duo_world.tick()

            # apply control
            tm_control = engine.ego.get_mirror_control()
            agent_control = decisionAgent.make_decision(tm_control)
            engine.ego.apply_real_control(agent_control)


            # apply goal
            if engine.lead is not None:
                engine.tm_mirror.set_path(engine.ego.mirror, [engine.lead.mirror.get_location()])

            # synchronization real npc with mirror npcs
            engine.synchronization_real_npc_with_mirror_npcs()

            # synchronization mirror ego with real ego
            engine.synchronization_mirror_ego_with_real_ego()

            # spectator
            engine.update_spectator()


    except KeyboardInterrupt:
        print("\nSimulation stopped by user (KeyboardInterrupt).")

    except Exception as e:
        print(f"\nAn critical error occurred during simulation loop: {e}")
        traceback.print_exc()
    finally:
        engine.cleanup()

from ACC.Training.Env import CarlaEnv, GymnasiumToGymWrapper
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.callbacks import CollectDataset
import torch.nn as nn
import torch.optim as optim
import torch

def train_loop(args):
    scene = Scenario('vehicle.tesla.model3', delta_seconds=args.delta_seconds,
                     map_name=args.map, number_of_npc=args.num_npcs)
    env = CarlaEnv(args, scene)

    env = GymnasiumToGymWrapper(env)

    episode_over = False
    total_reward = 0

    class BiasedActorNetwork(nn.Module):
        def __init__(self, input_shape, output_shape, **kwargs):
            super().__init__()
            n_input = input_shape[0]
            n_output = output_shape[0]

            self.network = nn.Sequential(
                nn.Linear(n_input, 32),  # Simpler: 32 instead of 64
                nn.ReLU(),
                nn.Linear(32, n_output),
                nn.Sigmoid()
            )

            # Bias towards throttle (index 0), away from brake (index 1)
            with torch.no_grad():
                self.network[-2].bias[0] = 100  # Start throttle around 0.5
                self.network[-2].bias[1] = -1  # Start brake near 0

        def forward(self, x, **kwargs):
            return self.network(x)

    def critic_network(input_shape, output_shape, **kwargs):
        n_input = input_shape[0]
        return nn.Sequential(
            nn.Linear(n_input, 32),  # Simpler
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    policy = GaussianTorchPolicy(
        network=BiasedActorNetwork,
        input_shape=env.observation_space.shape,
        output_shape=env.action_space.shape,
        use_cuda=True
    )

    agent = PPO(
        mdp_info=env.info,
        policy=policy,
        actor_optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
        critic_params={
            'network': critic_network,
            'optimizer': {'class': optim.Adam, 'params': {'lr': 1e-3}},
            'loss': nn.MSELoss(),
            'input_shape': env.observation_space.shape,
            'output_shape': (1,)
        },
        n_epochs_policy=10,
        batch_size=64,
        eps_ppo=0.2,
        lam=0.95,  # GAE lambda (Generalized Advantage Estimation)
        ent_coeff=0.1  # Entropy coefficient (optional)
    )

    logger = Logger(log_name='carla_ppo', results_dir='./logs')
    collect_dataset = CollectDataset()

    core = Core(agent, env, callbacks_fit=[collect_dataset])

    core.learn(n_steps=100000, n_steps_per_fit=2048)

    # Evaluate trained agent
    print("Evaluating...")
    core.evaluate(n_steps=5000, render=False)

    # Access collected data
    dataset = collect_dataset.get()
    print(f"Average reward: {dataset.mean()}")

    env.close()

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

    #real ports
    parser.add_argument('--real-port', default=2000, type=int,
                        help='TCP port for the REAL CARLA server (default: 2000)')
    parser.add_argument('--real-stream-port', default=2001, type=int,
                        help='Streaming port for the REAL CARLA server (default: 2001)')

    #mirror ports
    parser.add_argument('--mirror-port', default=4000, type=int,
                        help='TCP port for the MIRROR CARLA server (default: 4000)')
    parser.add_argument('--mirror-stream-port', default=4001, type=int,
                        help='Streaming port for the MIRROR CARLA server (default: 4001)')


    # Traffic Manager Ports
    parser.add_argument('--tm-mirror-port', default=9000, type=int,
                        help='Port for MIRROR Traffic Manager (default: 9000)')

    # Simulation Settings
    parser.add_argument('--map', default='Town03', help='Map to load (should match both servers) (default: Town04)')
    parser.add_argument('--delta-seconds', default=0.05, type=float,
                        help='Fixed delta seconds for simulation (default: 0.05)')
    parser.add_argument('--num-npcs', default=2, type=int, help='Number of NPC vehicles to spawn (default: 2)')

    # Camera
    parser.add_argument('--width', default=1280, type=int, help='Camera image width (default: 1280)')
    parser.add_argument('--height', default=720, type=int, help='Camera image height (default: 720)')

    #training
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
            #main_loop(args)
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