# Running
## VNC
If on the vnc, do it in folder ``/DistributedAIProject`` and make sure you activated the venv of carla.

``
python -m ACC.main --carla-path ../../carla/CarlaUE4.sh --map Town02
``
## Own pc
You need to run it in the root folder (``/DistributedAIProject``) as well. You can execute it in your python 3.8 or 3.7 and 
pass the path to the modern venv  >= python 3.12. Just pass the path to the root folder, ex. on the vnc it would be ../venv_python310.

``
python -m ACC.main --carla-path ../../carla/CarlaUE4.sh --map Town02 --venv {your path}
``