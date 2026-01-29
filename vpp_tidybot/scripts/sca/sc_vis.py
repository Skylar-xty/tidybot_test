import pybullet as p
import pybullet_data
import time
import os
import sys

# Import Panda class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Panda import Panda

# 1. Initialize Panda robot (handles connection, URDF loading, etc.)
urdf_path = "G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/base_fix.urdf"
panda_robot = Panda(stepsize=1e-3, realtime=0, connection_mode=p.GUI, urdf_path=urdf_path)
franka_id = panda_robot.robot

# 2. Optional: Add gravity (Panda class does this, but ensuring it here doesn't hurt)
p.setGravity(0, 0, -9.81)

# 3. Set additional search path (Panda class handles mesh path, this is for other assets if needed)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 4. Collision filter (already handled in Panda class if flags are set, but explicit filter for specific pairs)
p.setCollisionFilterPair(franka_id, franka_id, 4, 6, enableCollision=0)

# 5. Set the 7 arm joints to the target configuration
# Panda class identifies arm joints as indices 2..8
joints = panda_robot.joints # This should be [2, 3, 4, 5, 6, 7, 8]
target_positions = [-2.232897369, 2.04634149, -2.899228957, -0.313787628, 2.130369678, -0.406421803, -3.013027418]

if len(joints) != len(target_positions):
    print(f"Warning: Number of joints ({len(joints)}) does not match target positions ({len(target_positions)})")

for i, q in zip(joints, target_positions):
    p.resetJointState(franka_id, i, q)

# 关键：设置位置控制，让电机出力保持住这个姿态，抵抗重力
p.setJointMotorControlArray(
    bodyUniqueId=franka_id,
    jointIndices=joints,
    controlMode=p.POSITION_CONTROL,
    targetPositions=target_positions,
    forces=[1000.0] * len(joints), # 给予足够大的最大力矩
    positionGains=[0.1] * len(joints),
    velocityGains=[1.0] * len(joints)
)

# 6. Run simulation steps to detect collisions
for _ in range(10):
    p.stepSimulation()
    time.sleep(1/240)

# 7. Check for self-collisions
contacts = p.getContactPoints(bodyA=franka_id, bodyB=franka_id)
if len(contacts) == 0:
    print("✅ No self‐collision detected.")
    print("Press Ctrl+C or close the window to exit.")
    while True:
        p.stepSimulation()
        time.sleep(1/240)
else:
    print(f"❌ {len(contacts)} contact(s) found. Drawing debug markers at each contact point...")
    for c in contacts:
        # c[5] is positionOnA (vec3)
        contact_xyz = c[5]

        # Draw debug lines (RGB cross)
        p.addUserDebugLine(
            contact_xyz,
            [contact_xyz[0] + 0.05, contact_xyz[1], contact_xyz[2]],
            [1, 0, 0],    # Red
            lineWidth=3,
            lifeTime=0
        )
        p.addUserDebugLine(
            contact_xyz,
            [contact_xyz[0], contact_xyz[1] + 0.05, contact_xyz[2]],
            [0, 1, 0],    # Green
            lineWidth=3,
            lifeTime=0
        )
        p.addUserDebugLine(
            contact_xyz,
            [contact_xyz[0], contact_xyz[1], contact_xyz[2] + 0.05],
            [0, 0, 1],    # Blue
            lineWidth=3,
            lifeTime=0
        )
        
        # Highlight colliding links
        linkA = c[3]
        linkB = c[4]
        if linkA >= -1: p.changeVisualShape(franka_id, linkA, rgbaColor=[1, 0, 0, 1])
        if linkB >= -1: p.changeVisualShape(franka_id, linkB, rgbaColor=[1, 0, 0, 1])

    print("Collision markers drawn. Colliding links highlighted in red.")
    print("Use mouse to rotate/zoom.")
    while True:
        p.stepSimulation()
        time.sleep(1/240)

p.disconnect()
