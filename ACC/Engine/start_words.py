import os
import subprocess
import time
import logging
from typing import Optional, List, Dict, Tuple
import carla


class CarlaServerManager:

    def __init__(self, carla_path: str, host: str = '127.0.0.1'):

        if not os.path.exists(carla_path):
            raise FileNotFoundError(f"CARLA executable not found at '{carla_path}'")
        self.carla_path = carla_path
        self.host = host
        self.real_server_process: Optional[subprocess.Popen] = None
        self.mirror_server_process: Optional[subprocess.Popen] = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def launch_servers(self, real_port: int, real_stream_port: int, mirror_port: int, mirror_stream_port: int, connect_timeout: int = 60):

        if self.real_server_process or self.mirror_server_process:
            logging.warning("Servers already seem to be launched. Skipping launch.")
            return

        try:
            # --- Launch Real Server (Rendered) ---
            real_cmd = [
                self.carla_path,
                f"-carla-rpc-port={real_port}",
                f"-carla-streaming-port={real_stream_port}",
                "-nosound",
            ]
            logging.info(f"Starting REAL server: {' '.join(real_cmd)}")
            self.real_server_process = subprocess.Popen(real_cmd)
            logging.info(f"REAL server process started with PID: {self.real_server_process.pid}")

            # --- Launch Mirror Server (No Rendering) ---
            mirror_cmd = [
                self.carla_path,
                "-nullrhi",
                f"-carla-rpc-port={mirror_port}",
                f"-carla-streaming-port={mirror_stream_port}",
                "-nosound"
            ]
            logging.info(f"Starting MIRROR server: {' '.join(mirror_cmd)}")
            self.mirror_server_process = subprocess.Popen(mirror_cmd)
            logging.info(f"MIRROR server process started with PID: {self.mirror_server_process.pid}")

            # --- Wait for Servers to Become Connectable ---
            logging.info(f"Waiting up to {connect_timeout} seconds for servers to become connectable...")
            start_time = time.time()
            real_ready = False
            mirror_ready = False

            while time.time() - start_time < connect_timeout:
                if not real_ready:
                    real_ready = self._check_server_connection(real_port)
                if not mirror_ready:
                    mirror_ready = self._check_server_connection(mirror_port)

                if real_ready and mirror_ready:
                    logging.info("Both servers are connectable.")
                    return  # Success

                # Check if any process terminated unexpectedly
                if self.real_server_process and self.real_server_process.poll() is not None:
                    raise RuntimeError(
                        f"Real server process (PID: {self.real_server_process.pid}) terminated unexpectedly.")
                if self.mirror_server_process and self.mirror_server_process.poll() is not None:
                    raise RuntimeError(
                        f"Mirror server process (PID: {self.mirror_server_process.pid}) terminated unexpectedly.")

                time.sleep(2)

            raise TimeoutError(f"Servers did not become connectable within {connect_timeout} seconds.")

        except Exception as e:
            logging.error(f"Failed during server launch or connection check: {e}")
            self.terminate_servers()
            raise

    def _check_server_connection(self, port: int, timeout: float = 2.0) -> bool:
        """Attempts to connect to a CARLA server at the given port."""
        try:
            client = carla.Client(self.host, port)
            client.set_timeout(timeout)
            client.get_server_version()  # Simple command to check connectivity
            logging.debug(f"Successfully connected to server on port {port}.")
            return True
        except RuntimeError as e:
            logging.debug(f"Connection attempt to port {port} failed: {e}")
            return False
        except Exception as e:
            logging.warning(f"Unexpected error checking connection to port {port}: {e}")
            return False

    def terminate_servers(self):
        """Terminates any running CARLA server processes managed by this instance."""
        logging.info("Attempting to terminate CARLA server processes...")
        processes_to_terminate = [
            ("MIRROR", self.mirror_server_process),
            ("REAL", self.real_server_process)
        ]

        for name, process in processes_to_terminate:
            if process and process.poll() is None: # Check if process exists and is running
                logging.info(f"Terminating {name} server (PID: {process.pid})...")
                process.kill()
                try:
                    process.kill()
                except Exception as e:
                     logging.error(f"Error during {name} server termination (PID: {process.pid}): {e}")

        # Reset process variables
        self.real_server_process = None
        self.mirror_server_process = None
        logging.info("Server termination attempts complete.")


