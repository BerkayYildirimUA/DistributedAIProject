# How to run locally?
You need to run make use of two different python versions.

## 1. Carla
First start the carla script to generate the car data in python 
3.8/3.7.
``
python call_inference.py
``
## 2. Object detection
Run ``
python run_inference.py
``
in newer python version: 3.12/3.13
to visualize the car camera in real-time with boxes overlay.

# How to run on vnc?

1. Start carla on vnc

``cd ~/shared/carla``

``./CarlaUE4.sh``
2. Enter Carla venv

``cd ~/shared/carla/PythonAPI/examples``

``source py_carla/bin/activate``
3. Run Carla client script:
This script will load the world, vehicles in the carla simulator and adds spectator view
to track our car. It sends every frame to a shared memory that will be processed by another script
see step 5.

``cd ~/shared/gitrepo/DistributedAIProject/app``

``python run_world.py``

4. Open new terminal and enter venv for processing the data. We use for this venv a new python verion: 3.12.

``cd ~/shared/gitrepo``

``source venv_python310/bin/activate``
5. Run the data processing script: it will read the shared memory and render the camera view of the car in 
a separate window and overlay the boxes (object detection). 

``cd ~/shared/gitrepo/DistributedAIProject/app``

``python3.12 run_vehicle_pov.py``