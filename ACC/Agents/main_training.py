# https://mushroomrl.readthedocs.io/en/latest/?badge=latest

import argparse
import logging
import traceback

import numpy as np
from mushroom_rl.utils.dataset import compute_J

from ACC.Engine.scenario import Scenario
from ACC.Engine.start_words import CarlaServerManager
from ACC.Agents.RLAgents import ACC_TD3Agent
import wandb
import time

from ACC.Utils.EarlyStopper import EarlyStopper


class loop_info:
    def __init__(self, duration_seconds, name):
        self.duration = duration_seconds
        self.name = name

def training_loop(args):

    scenarios_list = [
        #(
        #    loop_info(2 * 60 * 60, "speed"),
        #    Scenario(
        #        'vehicle.tesla.model3',
        #        delta_seconds=args.delta_seconds,
        #        map_name="CUSTOM_STRAIGHT",
        #        number_of_npc=0,
        #        reward_geforce=False,
        #        reward_safe_distance=False,
        #        reward_crash=True,
        #        reward_speed_limit=True
        #    )
        #),
      #  (
      #      loop_info(2 * 60 * 60, "speed_lead"),
      #      Scenario(
      #          'vehicle.tesla.model3',
      #          delta_seconds=args.delta_seconds,
      #          map_name="CUSTOM_STRAIGHT",
      #          number_of_npc=0,
      #          lead_car_bp_name='vehicle.tesla.model3',
      #          reward_geforce=True,
      #          reward_safe_distance=True,
      #          reward_crash=True,
      #          reward_speed_limit=True
      #      )
      #  ),
      #  (    #
         #   loop_info(2 * 60 * 60, "speed_lead_lights"),
         #   #}    Scenario(
        ##}        'vehicle.tesla.model3',
        ##}        delta_seconds=args.delta_seconds,
        ##}        map_name="CUSTOM_STRAIGHT_WITH_LIGHTS", #CUSTOM_STRAIGHT_WITH_LIGHTS
        ##}        number_of_npc=0,
        ##}        lead_car_bp_name='vehicle.tesla.model3',
        ##}        reward_geforce=True,
        ##}        reward_safe_distance=True,
        ##}        reward_crash=True,
        ##}        reward_speed_limit=True,
        ##}        reward_light=False
        #}    )
        #}),

            (
            loop_info(10 * 60, f"speed_lead_lights_r_{args.model_nr}"),
            Scenario(
                'vehicle.tesla.model3',
                delta_seconds=args.delta_seconds,
                map_name="CUSTOM_STRAIGHT_WITH_LIGHTS",  # CUSTOM_STRAIGHT_WITH_LIGHTS
                number_of_npc=0,
                lead_car_bp_name='vehicle.tesla.model3',
                reward_geforce=True,
                reward_safe_distance=True,
                reward_crash=True,
                reward_speed_limit=True,
                reward_light=True
            )
        )
    ]

    current_model_name= args.load_model

    chunk_duration_seconds = 1 * 60 * 60


    for info, scene in scenarios_list:
        stopper = EarlyStopper(patience=3, min_delta=1)
        scene.name = info.name
        total_duration = info.duration
        elapsed_duration = 0

        logging.info(f"========================================")
        logging.info(f"STARTING SCENARIO: {info.name}")
        logging.info(f"Loading from: {current_model_name if current_model_name else 'Scratch'}")
        logging.info(f"========================================")

        while elapsed_duration < total_duration:
            remaining_time = total_duration - elapsed_duration
            current_chunk_duration = min(chunk_duration_seconds, remaining_time)

            logging.info(f"Starting training chunk: {current_chunk_duration}s. Elapsed: {elapsed_duration}s / {total_duration}s")
            logging.info(f"Loading from: {current_model_name if current_model_name else 'Scratch'}")

            server_manager = None
            agent = None

            try:
                server_manager = CarlaServerManager(args.carla_path, args.host)
                server_manager.launch_servers(
                    args.real_port,
                    args.real_stream_port,
                    args.mirror_port,
                    args.mirror_stream_port,
                    no_render=args.no_display
                )

                if current_model_name != "":
                    agent = ACC_TD3Agent(args, load_model_name=current_model_name, scene=scene)
                else:
                    agent = ACC_TD3Agent(args, scene=scene)

                if elapsed_duration == -1: #TODO: I MADE IT -1 SO IT NEVER TRIGGERS
                    agent.reset_buffer()

                agent.train(duration_seconds=current_chunk_duration)

                dataset = agent.dataset_callback.get()
                J_metrics = compute_J(dataset, agent.env.info.gamma)
                current_avg_reward = np.mean(J_metrics)

                logging.info(f"Chunk Reward Stats: Mean J={current_avg_reward:.2f} | Max J={np.max(J_metrics):.2f}")

                if stopper(current_avg_reward):
                    checkpoint_suffix = f"{info.name}_CONVERGED"
                    current_model_name = agent.save_model(suffix=checkpoint_suffix)
                    agent.close()
                    server_manager.terminate_servers()
                    break


                checkpoint_suffix = f"{info.name}_chunk_{int(elapsed_duration + current_chunk_duration)}"
                current_model_name = agent.save_model(suffix=checkpoint_suffix)
                logging.info(f"Chunk finished. Model saved to {current_model_name}")

                elapsed_duration += current_chunk_duration
                agent.dataset_callback.clean()

            except Exception as e:
                logging.error(f"Crash detected during training chunk: {e}")
                traceback.print_exc()
                logging.info("Attempting to restart servers and resume from last checkpoint...")
                time.sleep(5)

            finally:
                if agent:
                    try:
                        agent.close()
                    except:
                        pass

                del agent
                agent = None

                if server_manager:
                    try:
                        server_manager.terminate_servers()
                    except:
                        pass

                    del server_manager
                    server_manager = None

                import gc
                gc.collect()
                time.sleep(20)
        logging.info(f"Scenario {info.name} COMPLETED.")







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
    parser.add_argument('--map', default='random', help='Map to load (default: random)')
    parser.add_argument('--spawn_point', default='random', help='Spawn point of the ego (default: random)')


    parser.add_argument('--delta-seconds', default=0.01, type=float,
                        help='Fixed delta seconds for simulation (default: 0.05)')
    parser.add_argument('--num-npcs', default=2, type=int, help='Number of NPC vehicles to spawn (default: 2)')

    # Camera
    parser.add_argument('--width', default=1280, type=int, help='Camera image width (default: 1280)')
    parser.add_argument('--height', default=720, type=int, help='Camera image height (default: 720)')

    # training
    parser.add_argument('--do_train', action='store_true', help='Train an RL agent or just run the sim')
    parser.add_argument('--horizon', default=20000, help='max sim length before resetting')

    parser.add_argument('--no_display', action='store_true', help='Disable rendering for SSH')
    parser.add_argument('--random_speed_limit', action='store_true', help='to train better at this')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    parser.add_argument('--load_model', type=str, default="", help='Path to .msh file to load')
    parser.add_argument('--model_nr', type=int, default=600, help='How long to train before restarting')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    real_server_process = None
    mirror_server_process = None
    server_manager = None

    try:

        if not args.do_train:
            server_manager = CarlaServerManager(args.carla_path, args.host)
            server_manager.launch_servers(
                args.real_port,
                args.real_stream_port,
                args.mirror_port,
                args.mirror_stream_port,
                no_render=args.no_display
            )

            logging.info("Servers launched successfully. Starting main simulation loop...")
            print("-" * 30)
            manager = ACC_TD3Agent(args, load_model_name="251204_122515_TD3_Speed_limit_and_safe_dist_chunk_28800.msh")
            manager.evaluate()
            if server_manager: server_manager.terminate_servers()
        else:
            training_loop(args)


    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

    print("Script finished.")
