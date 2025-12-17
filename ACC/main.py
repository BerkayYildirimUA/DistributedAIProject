import argparse
import logging
import traceback
import numpy as np
import subprocess

import pygame

from ACC.Engine.scenario import Scenario
from ACC.Engine.start_words import CarlaServerManager
from ACC.Utils.Sensors import CarlaVBWorldStateSensor, CarlaWorldStateSensor
from ACC.Engine.engine import Engine
from ACC.hud_display import HUD
from app.data_processors.objects_in_front_calculator import ObjectsInFrontCalculator
from app.memory.shared_memory import VehicleStateMemory, FrameIdMemory
from app.data_processors.metrics_logger import MetricsLogger
from app.memory.shared_memory import VehicleStateMemory
from ACC.Agents.RLAgents import RLDecisionAgent
import app.constants  as constants
"""
RL FEEDBACK 

no need to gragh rewawrd for the PP. 
Don't use TM data either.

Just focus on stuff the rewards are based on:
    - geforces over time, maybe as histogram
    - speed
    - speed limit
    - etc etc
    
train in steps, don't overcomplicate at first. Turn on rewards as we go

maybe change action space center if need be, like 0 =! do nothing, perhabs. 


--no_display
"""


def main_loop(args):
    scene = Scenario('vehicle.tesla.model3', delta_seconds=args.delta_seconds,
                     map_name=args.map, number_of_npc=0, lead_car_bp_name='vehicle.tesla.model3')
    engine = Engine(args, scene)

    # Logging
    GT_lead_distance_logger = MetricsLogger(constants.GT_LEAD_DISTANCE_FILE, compress=True)
    lead_distance_logger = MetricsLogger(constants.LEAD_DISTANCE_FILE, compress=True)
    GT_speed_limit_logger = MetricsLogger(constants.GT_SPEED_LIMIT_FILE, compress=True)
    speed_logger = MetricsLogger(constants.SPEED_FILE, compress=True)
    GT_g_force_logger = MetricsLogger(constants.GT_G_FORCE_FILE, compress=True)
    g_force_logger = MetricsLogger(constants.G_FORCE_FILE, compress=True)
    GT_safe_following_distance_logger = MetricsLogger(constants.GT_SAFE_FOLLOWING_DISTANCE_FILE, compress=True)
    GT_object_count_metrics_logger = MetricsLogger(constants.GT_OBJECTS_IN_FRONT_COUNT_FILE, compress=True)
    GT_traffic_sign_count_metrics_logger = MetricsLogger(constants.GT_TRAFFIC_SIGN_COUNT_FILE, compress=True)
    GT_traffic_light_count_metrics_logger = MetricsLogger(constants.GT_TRAFFIC_LIGHT_COUNT_FILE, compress=True)
    GT_pedestrian_count_metrics_logger = MetricsLogger(constants.GT_PEDESTRIAN_COUNT_FILE, compress=True)
    GT_vehicle_count_metrics_logger = MetricsLogger(constants.GT_VEHICLE_COUNT_FILE, compress=True)

    frame_id_memory = FrameIdMemory().get_write_access()
    client_clock = pygame.time.Clock()
    try:
        engine.connect_to_worlds()

        if not engine.setup():
            raise RuntimeError("Engine setup failed. Exiting.")

        # sensor and agent Setup (Real World)
        # Two sensors are used:
        # - one to collect the ground truth info from the carla simulator itself
        # - one to pass the current state to the RL agent
        sensor_ground_truth =  CarlaWorldStateSensor(engine.ego.real, engine.duo_world.get_real_world())
        sensor_real = CarlaVBWorldStateSensor(
            engine.ego.real,
            engine.duo_world.get_real_world(),
            use_traffic_lights=False,
            use_traffic_signs=True
        )

        decisionAgent = RLDecisionAgent(sensor_real, "251210_001928_TD3_Aldebaran_chunk_7200.msh")
        #Initialize Pygame HUD Display
        pygame.init() #initialize pygame modules

        hud_width = 220
        hud_height = 400

        display = pygame.display.set_mode((hud_width, hud_height))
        #hud = HUD(args.width, args.height) #HUD initialization
        hud = HUD(hud_width, hud_height)
        engine.duo_world.real_world.on_tick(hud.on_world_tick)


        crash_detected = False
        frames_after_crash = 0
        MAX_FRAMES_AFTER_CRASH = 60  # then reset or exit
        # Create needed memory access to sync carla data from sensors to newer python env
        vehicle_state_memory = VehicleStateMemory().get_write_access()
        # Used for logging
        objects_in_front_calculator = ObjectsInFrontCalculator(engine.duo_world.get_real_world(), engine.ego.real,
                                                               max_distance=20.0)

        while True:
            # Tick the simulation
            try:
                mirror_frame, frame_id  = engine.duo_world.tick()
                frame_id_memory.write(frame_id)
            except Exception as e:
                logging.error(f"Failed to tick world: {e}")
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            client_clock.tick(60)
            hud.tick(engine.duo_world.real_world, engine.ego.real, client_clock)

            # SAFETY CHECK
            if not engine.ego or not engine.ego.is_alive():
                logging.error("Ego vehicle is dead or missing. Exiting loop.")
                break

            try:
                state = sensor_real.get_state()
            except Exception as e:
                logging.error(f"Failed to get sensor state: {e}")
                break

            # ---- CRASH HANDLING ----
            if state.crash_intensity > 0.0:
                if not crash_detected:
                    logging.warning("Crash detected! Stopping vehicle...")
                    crash_detected = True

                frames_after_crash += 1

                # Apply brakes and stop after crash
                try:
                    import carla
                    stop_control = carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0,
                        steer=0.0,
                        hand_brake=True
                    )
                    engine.ego.apply_real_control(stop_control)
                except Exception as e:
                    logging.warning(f"Failed to apply brake after crash: {e}")

                # Exit after some frames post-crash
                if frames_after_crash > MAX_FRAMES_AFTER_CRASH:
                    logging.info("Exiting simulation after crash cooldown.")
                    break

                # Skip normal control logic after crash
                continue

            # ---- NORMAL OPERATION (no crash) ----
            try:
                # Get TM control for steering
                tm_control = engine.ego.get_mirror_control()
                agent_control = decisionAgent.make_decision(tm_control)
                engine.ego.apply_real_control(agent_control)
            except Exception as e:
                logging.error(f"Error in control loop: {e}")
                break

            # ---- LEAD VEHICLE HANDLING (with safety checks) ----
            if engine.lead is not None:
                try:
                    if engine.lead.is_alive() and engine.lead.mirror and engine.lead.mirror.is_alive:
                        lead_location = engine.lead.mirror.get_location()
                        engine.tm_mirror.set_path(engine.ego.mirror, [lead_location])
                    else:
                        logging.warning("Lead vehicle is no longer alive, skipping path update")
                except RuntimeError as e:
                    logging.warning(f"Failed to update lead path (actor may be dead): {e}")
                except Exception as e:
                    logging.warning(f"Unexpected error updating lead path: {e}")

            # ---- SYNCHRONIZATION ----
            try:
                engine.synchronization_real_npc_with_mirror_npcs()
            except Exception as e:
                logging.warning(f"NPC sync failed: {e}")

            # Get state and write to shared memory
            real_ego_state = sensor_real.get_state()
            vehicle_state_memory.write(np.array([real_ego_state.speed_ms*3.6, real_ego_state.steer_rad], dtype=np.float32))
            try:
                engine.synchronization_mirror_ego_with_real_ego()
            except Exception as e:
                logging.warning(f"Ego sync failed: {e}")

            try:
                engine.sync_traffic_lights()
            except Exception as e:
                logging.warning(f"traffic lights sync failed: {e}")

            try:
                engine.update_spectator()
            except Exception as e:
                logging.debug(f"Spectator update failed: {e}")
                
            # Metrics
            ground_truth_state = sensor_ground_truth.get_state()
            GT_lead_distance = ground_truth_state.lead_distance_m
            lead_distance = real_ego_state.lead_distance_m
            experienced_g_force = real_ego_state.g_force_ego
            driving_speed = real_ego_state.speed_ms
            GT_speed_limit = real_ego_state.speed_limit_ms
            GT_safe_following_distance=ground_truth_state.safe_following_distance_m
            object_count = objects_in_front_calculator.count_objects_in_front()
            GT_object_count = object_count["total"]
            GT_traffic_sign_count = object_count["traffic_signs"]
            GT_traffic_light_count = object_count["traffic_lights"]
            GT_vehicle_count = object_count["vehicles"]
            GT_pedestrian_count = object_count["pedestrians"]

            GT_lead_distance_logger.log(
                frame_id=frame_id,
                lead_distance=float(GT_lead_distance),
            )
            lead_distance_logger.log(
                frame_id=frame_id,
                lead_distance=float(lead_distance),
            )
            GT_safe_following_distance_logger.log(
                frame_id=frame_id,
                safe_following_distance=float(GT_safe_following_distance),
            )
            g_force_logger.log(
                frame_id=frame_id,
                force=float(experienced_g_force)
            )
            GT_g_force_logger.log(
                frame_id=frame_id,
                force=float(0.09)
            )
            speed_logger.log(
                frame_id=frame_id,
                speed=float(driving_speed*3.6)
            )
            GT_speed_limit_logger.log(
                frame_id=frame_id,
                speed_limit=float(GT_speed_limit*3.6)
            )
            GT_object_count_metrics_logger.log(
                frame_id=frame_id,
                ground_truth_objects=int(GT_object_count),
            )
            GT_traffic_sign_count_metrics_logger.log(
                frame_id=frame_id,
                ground_truth_traffic_signs=int(GT_traffic_sign_count),
            )
            GT_traffic_light_count_metrics_logger.log(
                frame_id=frame_id,
                ground_truth_traffic_lights=int(GT_traffic_light_count),
            )
            GT_vehicle_count_metrics_logger.log(
                frame_id=frame_id,
                ground_truth_vehicles_front_count=int(GT_vehicle_count),
            )
            GT_pedestrian_count_metrics_logger.log(
                frame_id=frame_id,
                ground_truth_pedestrians=int(GT_pedestrian_count),
            )
            
             # Render all components
            display.fill((0, 0, 0))  # Clear the screen (assuming black background)

            hud.render(display)  # Render the HUD on the Pygame surface
            pygame.display.flip()  # Update the screen

    except KeyboardInterrupt:
        print("\nSimulation stopped by user (KeyboardInterrupt).")

    except Exception as e:
        print(f"\nA critical error occurred during simulation loop: {e}")
        traceback.print_exc()
    finally:
        # Close loggers
        GT_lead_distance_logger.close()
        lead_distance_logger.close()
        GT_safe_following_distance_logger.close()
        g_force_logger.close()
        speed_logger.close()
        GT_speed_limit_logger.close()
        GT_object_count_metrics_logger.close()
        GT_traffic_sign_count_metrics_logger.close()
        GT_traffic_light_count_metrics_logger.close()
        GT_vehicle_count_metrics_logger.close()
        GT_pedestrian_count_metrics_logger.close()

        engine.cleanup()
        # Clean up sensor first
        if 'sensor_real' in locals() and sensor_real is not None:
            try:
                sensor_real.cleanup()
            except Exception as e:
                logging.warning(f"Error cleaning up sensor: {e}")

        engine.cleanup(True)



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

    parser.add_argument('--venv', default="../venv_python310/bin/python3.12", type=str, help='Path to the venv containing python >3.12')

    # Camera
    parser.add_argument('--width', default=1280, type=int, help='Camera image width (default: 1280)')
    parser.add_argument('--height', default=720, type=int, help='Camera image height (default: 720)')

    # training
    parser.add_argument('--do_train', default=False, type=bool, help='Train an RL agent or just run the sim')
    parser.add_argument('--horizon', default=20000, help='max sim length before resetting')

    parser.add_argument('--no_display', action='store_true', help='Disable rendering for SSH')
    parser.add_argument('--random_speed_limit', action='store_true', help='to train better at this')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')


    args = parser.parse_args()


    real_server_process = None
    mirror_server_process = None
    server_manager = None

    try:
        # Start the script to run the vehicle pov in  the modern python env and run it in background
        subprocess.Popen([f"{args.venv}","-m", "app.run_vehicle_pov.py"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True)

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
