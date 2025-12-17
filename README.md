# Distributed AI Project  
**Adaptive Cruise Control**

## Overview
This project consists of two main components: **Computer Vision (CV)** and **Reinforcement Learning (RL)**, which are also reflected in the directory structure.

- **ACC**: Contains all code to train, evaluate, and run the RL agents in the CARLA simulator.
- **app**: Contains all code required to implement and visualize the CV modules.
- **training**: Contains code used to train the CV models.

## Environments
To run the project, two Python virtual environments must be set up:

- **Python 3.8**: Used to run the CARLA simulator and RL agents. Dependencies should be installed using `requirements/requirements_carla.txt`.
- **Python 3.11 or higher**: Used for the CV modules to take advantage of modern libraries.  Dependencies should be installed using `requirements/requirements_modern.txt`.

Data is exchanged between the environments using shared memory. Specifically, `.dat` files are created for reading and writing data.  
This data management is abstracted through dedicated Python classes.

## Running
The project provides two ways to run the implemented modules in CARLA.

### Using CARLA Autopilot
In this mode, the vehicle is controlled by CARLA’s built-in autopilot.  
The scripts spawn additional actors, enabling a realistic simulation to evaluate the CV modules.

Two commands need to be executed:

**Start the world** (run with Python 3.8 from the root directory):
```bash
python app.run_world.py
````
**Start the CV modules** (run with python 3.11 or higher from the root directory):

````
python app.run_vehicle_pov.py
````

### With RL agents
Run the following command with Python 3.8 from the root directory:
````
python -m ACC.main --carla-path ../../carla/CarlaUE4.sh --map Town05
````
To specify a different Python virtual environment for the CV modules:

````
python -m ACC.main --carla-path {carla path} --map Town05 --venv {your path}
````

### Visualizations
When running with RL agents, data is collected automatically.
This data can be visualized to evaluate agent performance using the following command:
````
python -m ACC.export_metrics
````
This will generate images containing plots of the collected metrics, which will be saved in the root directory.
## Documentation
Additional documentation can be found in directory ``documentation``.
## Demo video
A demonstation video of the project can be found here:  [demo video](https://youtu.be/aEYZUimxtJ4).