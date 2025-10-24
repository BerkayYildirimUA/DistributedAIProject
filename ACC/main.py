import argparse
import logging
import traceback

from ACC.Engine.start_words import CarlaServerManager
from ACC.Utils.implementations import CarlaStateSensor, SimpleAccAgent
from ACC.Engine.engine import Engine

def main_loop(args):
    engine = Engine(args)

    try:
        engine.connect_to_worlds()
        if not engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")

        # sensor and agent Setup (Real World)
        sensor_real = CarlaStateSensor(engine.ego.real, engine.lead.real)
        decisionAgent = SimpleAccAgent(engine.ego.real, sensor_real)



        while True:
            mirror_frame, _ = engine.duo_world.tick()

            # apply control
            tm_control = engine.ego.get_mirror_control()
            agent_control = decisionAgent.make_decision(tm_control)
            engine.ego.apply_real_control(agent_control)


            # apply goal
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

        main_loop(args)

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