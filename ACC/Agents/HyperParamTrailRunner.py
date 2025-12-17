import argparse
import json
import logging
import sys

import numpy as np
from mushroom_rl.utils.dataset import compute_J

from ACC.Engine.scenario import Scenario
from ACC.Engine.start_words import CarlaServerManager
from ACC.Agents.RLAgents import ACC_TD3Agent, TD3Config


def run_trial(args, hyperparams: dict, output_file: str):

    TD3Config.LR_ACTOR = hyperparams["lr_actor"]
    TD3Config.LR_CRITIC = hyperparams["lr_critic"]
    TD3Config.BATCH_SIZE = hyperparams["batch_size"]
    TD3Config.TAU = hyperparams["tau"]
    TD3Config.POLICY_DELAY = hyperparams["policy_delay"]
    TD3Config.NOISE_STD = hyperparams["noise_std"]
    TD3Config.NOISE_CLIP = hyperparams["noise_clip"]

    scene = Scenario(
        'vehicle.tesla.model3',
        delta_seconds=args.delta_seconds,
        map_name="CUSTOM_STRAIGHT_WITH_LIGHTS",
        number_of_npc=0,
        lead_car_bp_name='vehicle.tesla.model3',
        reward_geforce=True,
        reward_safe_distance=True,
        reward_crash=True,
        reward_speed_limit=True,
        reward_light=True
    )
    scene.name = f"HPO_Trial_{hyperparams.get('trial_number', 0)}"

    server_manager = None
    agent = None
    result = {"success": False, "avg_reward": float('-inf'), "error": None}

    try:
        server_manager = CarlaServerManager(args.carla_path, args.host)
        server_manager.launch_servers(
            args.real_port,
            args.real_stream_port,
            args.mirror_port,
            args.mirror_stream_port,
            no_render=args.no_display
        )

        agent = ACC_TD3Agent(args, scene=scene)
        agent.train(duration_seconds=args.trial_duration)

        dataset = agent.dataset_callback.get()
        J_metrics = compute_J(dataset, agent.env.info.gamma)
        avg_reward = float(np.mean(J_metrics))

        result["success"] = True
        result["avg_reward"] = avg_reward
        logging.info(f"Trial completed. Avg Reward: {avg_reward:.4f}")

    except Exception as e:
        logging.error(f"Trial failed: {e}")
        result["error"] = str(e)

    finally:
        if agent:
            try:
                agent.close()
            except:
                pass

        if server_manager:
            try:
                server_manager.terminate_servers()
            except:
                pass

    with open(output_file, 'w') as f:
        json.dump(result, f)

    return result["success"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run single HPO trial')

    # CARLA settings
    parser.add_argument('--carla-path', default=r"D:\UA\Master\Semester1\AI\Project\Carla\CarlaUE4.exe")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--real-port', default=2000, type=int)
    parser.add_argument('--real-stream-port', default=2001, type=int)
    parser.add_argument('--mirror-port', default=4000, type=int)
    parser.add_argument('--mirror-stream-port', default=4001, type=int)
    parser.add_argument('--tm-mirror-port', default=9000, type=int)
    parser.add_argument('--delta-seconds', default=0.05, type=float)
    parser.add_argument('--horizon', default=12000, type=int)
    parser.add_argument('--map', default='CUSTOM_STRAIGHT_WITH_LIGHTS')
    parser.add_argument('--spawn_point', default='random')
    parser.add_argument('--no_display', action='store_true', default=True)
    parser.add_argument('--random_speed_limit', action='store_true', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)

    # Trial settings
    parser.add_argument('--trial-duration', default=1800, type=int)
    parser.add_argument('--hyperparams-file', required=True, help='JSON file with hyperparameters')
    parser.add_argument('--output-file', required=True, help='JSON file to write results')

    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    with open(args.hyperparams_file, 'r') as f:
        hyperparams = json.load(f)

    success = run_trial(args, hyperparams, args.output_file)

    sys.exit(0 if success else 1)