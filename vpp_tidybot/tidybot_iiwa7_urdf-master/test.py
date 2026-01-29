import pybullet as p
import time
import os
import pathlib
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

script_dir = pathlib.Path(__file__).resolve().parent
urdf_path = str(script_dir / "tidybot_iiwa7.urdf")


robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=False,
    flags=p.URDF_USE_SELF_COLLISION
)

while True:
    p.stepSimulation()
    time.sleep(0.01)
