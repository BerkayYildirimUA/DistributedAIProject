# How to Run the CARLA Simulation

This setup uses a primary script (`run_world.py`) to manage the CARLA world and sensor data (writing to shared memory). It also automatically launches a secondary script (`run_vehicle_pov.py`) to process and visualize this data (reading from shared memory).

---

## Running Locally

The setup may require two different Python versions depending on your CARLA installation:

* **`run_vehicle_pov.py`:** Generally requires Python 3.12 or newer for its dependencies.
* **`run_world.py`:** Requires a Python version compatible with your CARLA Python API installation (often Python 3.7 or 3.8 for older CARLA versions).

You only need to execute the main script (`run_world.py`), which will then launch the visualization script.

1.  **Start the CARLA Server:**
    Ensure your CARLA server (e.g., `./CarlaUE4.sh` or `CarlaUE4.exe`) is running.

2.  **Run the Simulation Bridge Script:**
    Open a terminal and navigate to the script's directory.

    * **Option A: If your CARLA installation is compatible with Python 3.12+:**
        Run the main script directly. It will use its own interpreter to launch `run_vehicle_pov.py`.
        ```bash
        python run_world.py
        ```

    * **Option B: If your CARLA installation requires an older Python (e.g., 3.7/3.8):**
        Run the main script using your CARLA-compatible Python interpreter and use the `--interpreter` argument to specify the path to the Python 3.12+ executable needed for `run_vehicle_pov.py`.
        ```bash
        # Example using Python 3.8 for the main script and Python 3.12 for the second
        # (Replace paths as necessary)

        # Windows
        python run_world.py --interpreter "C:\path\to\your\python312\python.exe"

        # Linux
        python run_world.py --interpreter /usr/bin/python3.12
        ```
        *Tip for PyCharm:* You can add the `--interpreter` argument in your run configuration (Edit Configurations... -> `run_world.py` -> Parameters field).

---

## Running on the VNC

1.  **Start CARLA on the VNC:**
    ```bash
    cd ~/shared/carla
    ./CarlaUE4.sh
    ```

2.  **Activate CARLA-Compatible Virtual Environment:**
    Activate the Python environment where `run_world.py` and the CARLA Python API are installed (e.g., Python 3.7/3.8).
    ```bash
    # Example using a virtual environment named 'py_carla'
    source ~/shared/carla/PythonAPI/examples/py_carla/bin/activate
    ```

3.  **Run the Simulation Bridge Script:**
    Navigate to the project directory and run the main script. Use the `--interpreter` argument to specify the path to the Python 3.12 executable located within its own virtual environment (which is called `venv_python310` in our case for some reason).
    ```bash
    cd ~/shared/gitrepo/DistributedAIProject/app # Navigate to the script's location

    # Specify the Python 3.12 interpreter from its venv for the visualization script
    python run_world.py --interpreter ~/shared/gitrepo/venv_python310/bin/python3.12
    ```

    The `run_world.py` script will now:
    * Load the world and vehicles in CARLA using the active (e.g., Python 3.8) environment.
    * Start sending sensor data to shared memory.
    * Automatically launch `run_vehicle_pov.py` using the specified Python 3.12 interpreter. This secondary script will read from shared memory and display the vehicle's camera view with object detection overlays.

---