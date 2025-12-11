import subprocess
import os
import time
import glob
import sys
import re
import threading
import queue

from ACC.Agents.main_training import force_cleanup

PYTHON_EXE = sys.executable
TRAIN_MODULE = "ACC.Agents.main_training"
CARLA_PATH = r"D:\UA\Master\Semester1\AI\Project\Carla\CarlaUE4.exe"
MAP_NAME = "CUSTOM_STRAIGHT_WITH_LIGHTS"

current_script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(current_script_dir, "models")
PROJECT_ROOT = os.path.abspath(os.path.join(current_script_dir, "..", ".."))

# Error detection settings
ERROR_PATTERN = "streaming client: connection failed"
MAX_ERRORS = 10

# Progress Bar Settings
PROGRESS_UPDATE_INTERVAL = 1

# --- TIMEOUT SETTINGS ---
MAX_RUNTIME_SECONDS = 30 * 60
MAX_SILENCE_SECONDS = 3 * 60





def get_latest_model_info():
    """Finds the latest checkpoint and calculates the next index."""
    if not os.path.exists(MODELS_DIR):
        return "", 0

    search_pattern = os.path.join(MODELS_DIR, "*_TD3_speed_lead_lights_r_*.msh")
    files = glob.glob(search_pattern)

    if not files:
        print("Launcher: No existing models found. Starting from 0.")
        return "", 0

    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]

    match = re.search(r"_r_(\d+)(?:_chunk|_CONVERGED)", latest_file)

    current_nr = 0
    if match:
        current_nr = int(match.group(1))
    else:
        print(f"Launcher Warning: Could not extract number from {os.path.basename(latest_file)}")

    return latest_file, current_nr + 1


def enqueue_output(out, q):
    """Reads lines from the process output and puts them into a queue."""
    for line in iter(out.readline, ''):
        q.put(line)
    out.close()


def run_one_round():
    latest_model_path, next_nr = get_latest_model_info()

    cmd = [
        PYTHON_EXE, "-m", TRAIN_MODULE,
        "--carla-path", CARLA_PATH,
        "--horizon", "12000",
        "--map", "CUSTOM_STRAIGHT_WITH_LIGHTS",
        "--random_speed_limit",
        "--no_display",
        "--do_train",
        "--model_nr", str(next_nr)
    ]

    if latest_model_path:
        print(f"Launcher: Resuming from {os.path.basename(latest_model_path)}")
        cmd.extend(["--load_model", latest_model_path])
    else:
        print("Launcher: Starting FRESH training run.")

    print(f"Launcher: Starting Chunk #{next_nr} as MODULE...")

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )

    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(process.stdout, q))
    t.daemon = True  # Thread dies if main script dies
    t.start()

    error_counter = 0
    last_progress_time = 0.0

    start_time = time.time()
    last_activity_time = time.time()

    try:
        while True:
            # Check if process is still running AND queue is empty
            if process.poll() is not None and q.empty():
                break

            # TIMEOUT CHECKS
            now = time.time()

            # Global Timeout
            if now - start_time > MAX_RUNTIME_SECONDS:
                print(f"\nLauncher: TIMEOUT REACHED ({MAX_RUNTIME_SECONDS / 60} mins). RESTARTING...")
                process.kill()
                break

            # Silence Timeout
            if now - last_activity_time > MAX_SILENCE_SECONDS:
                print(f"\nLauncher: SILENCE DETECTED (No output for {MAX_SILENCE_SECONDS / 60} mins). RESTARTING...")
                process.kill()
                break

            # READ OUTPUT
            try:
                line = q.get_nowait()  # Get line without waiting
                last_activity_time = now  # Update silence timer
            except queue.Empty:
                time.sleep(0.1)  # Wait a bit to not peg CPU
                continue

            # Anti-Spam Logic
            is_progress = ("%|" in line) and ("it/s" in line or "s/it" in line)

            if is_progress:
                if time.time() - last_progress_time > PROGRESS_UPDATE_INTERVAL:
                    print(f"\r{line.strip().ljust(120)}", end='', flush=True)
                    last_progress_time = time.time()
            else:
                print(f"\r{line}", end='')

            # Check for zombie client errors
            if ERROR_PATTERN in line:
                error_counter += 1
                if error_counter >= MAX_ERRORS:
                    print(f"\nLauncher: DETECTED {MAX_ERRORS} STREAMING ERRORS. KILLING PROCESS...")
                    process.kill()
                    break

        # Cleanup after loop
        if process.poll() is None:  # If we broke loop but process is alive
            process.kill()
            process.wait()

        if process.returncode == 0:
            print(f"\nLauncher: Chunk #{next_nr} completed successfully.")
        else:
            print(f"\nLauncher: Chunk #{next_nr} ENDED (Code {process.returncode}).")

    except KeyboardInterrupt:
        print("\nLauncher: Stopping loop.")
        process.kill()
        force_cleanup()
        sys.exit(0)


if __name__ == "__main__":
    print("------------------------------------------------")
    print("   INFINITE TRAINING LOOP - PROCESS ISOLATION   ")
    print(f"   Max Runtime: {MAX_RUNTIME_SECONDS / 60} mins")
    print(f"   Max Silence: {MAX_SILENCE_SECONDS / 60} mins")
    print(f"   Root: {PROJECT_ROOT}")
    print("------------------------------------------------")

    force_cleanup()

    while True:
        run_one_round()

        force_cleanup()

        print("Launcher: Cooling down for 10 seconds...")
        time.sleep(10)