# https://mushroomrl.readthedocs.io/en/latest/?badge=latest

import argparse
import logging
import traceback

from ACC.Engine.scenario import Scenario
from ACC.Engine.start_words import CarlaServerManager
from ACC.Agents.RLAgents import ACC_TD3Agent


class loop_info:
    def __init__(self, duration_seconds, name):
        self.duration = duration_seconds
        self.name = name

def training_loop(args):

    scenarios_list = [
        (
            loop_info(2 * 60 * 60, "keep_speed"),
            Scenario(
                'vehicle.tesla.model3',
                delta_seconds=args.delta_seconds,
                map_name=args.map,
                number_of_npc=0
            )
        ),
        (
            loop_info(2 * 60 * 60, "Speed_limit_and_safe_dist"),
            Scenario(
                'vehicle.tesla.model3',
                delta_seconds=args.delta_seconds,
                map_name=args.map,
                number_of_npc=0,
                lead_car_bp_name='vehicle.tesla.model3'
            )
        )
    ]

    current_model_path = None

    for info, scene in scenarios_list:
        logging.info(f"========================================")
        logging.info(f"STARTING SCENARIO: {info.name}")
        logging.info(f"Loading from: {current_model_path if current_model_path else 'Scratch'}")
        logging.info(f"========================================")

        try:
            manager = ACC_TD3Agent(args, load_model_path=current_model_path, scene=scene)

            manager.train(duration_seconds=info.duration)

            current_model_path = manager.save_model(suffix=info.name)

            logging.info(f"Scenario {info.name} finished. Model saved to {current_model_path}")

        except Exception as e:
            logging.error(f"Error during scenario {info.name}: {e}")
            traceback.print_exc()
            break

        finally:
            if 'manager' in locals():
                manager.close()







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


    parser.add_argument('--delta-seconds', default=0.05, type=float,
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

        if not args.do_train:
            manager = ACC_TD3Agent(args, load_model_name="")
            manager.evaluate()
        else:
            training_loop(args)


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
