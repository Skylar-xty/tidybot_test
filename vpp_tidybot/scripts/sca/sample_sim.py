#!/usr/bin/env python3
import os
import time
import numpy as np
import pybullet as p
import pybullet_data


def main():
    # ---- import Panda wrapper ----
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(this_dir))  # adjust if your scripts folder depth differs
    from Panda_toppling import Panda

    # ---- settings ----
    urdf_path = "G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7.urdf"
    stepsize = 2e-3

    # "垂下"姿态：这是一个示例。你可以替换成你认为更像垂下的关节角。
    # 关节顺序是 Panda.py 里 self.joints 对应 iiwa_joint_1..7（索引 2..8）
    # q_down = [1.5,-1.2, -1.1, 1.8, 1, 1.2, 0.0]

    q_down = [0,-2.4, -1.1, 0, 1, 1.2, 0.0]
    # q_down = [0.0,1.0,1.0,0.0,0.0,1.0,0.0]
    # 挥动参数（“大力挥动”）
    swing_joints = []       # 挥动第0、1关节（更容易带动整体）
    A = 5.0                     # 振幅（rad），可试 0.6~1.2
    f = 2.0                     # 频率（Hz），可试 1.0~3.0
    # max_force = 1400.0          # 关键：力要大（按需要调 500~4000）
    max_force = 0.0
    run_seconds = 30.0

    # ---- start sim ----
    panda = Panda(
        stepsize=stepsize,
        realtime=0,
        connection_mode=p.GUI,
        urdf_path=urdf_path,
        use_fixed_base=False,
    )
    robot = panda.robot
    joint_indices = list(panda.joints)
    print("Joint indices:", joint_indices)
    # global sim params
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # make contacts stable (simple)
    if hasattr(panda, "plane"):
        p.changeDynamics(panda.plane, -1, lateralFriction=1.0, restitution=0.0)
    p.changeDynamics(robot, -1, lateralFriction=1.0, restitution=0.0)

    # ---- reset base and arm pose ----
    p.resetBasePositionAndOrientation(robot, [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])

    for jid, q in zip(joint_indices, q_down):
        p.resetJointState(robot, jid, targetValue=float(q), targetVelocity=0.0)

    # settle a bit
    for _ in range(200):
        p.stepSimulation()
        time.sleep(stepsize)

    # ---- swing loop ----
    t0 = time.time()
    k = 0
    while True:
        t = time.time() - t0
        if t > run_seconds:
            break

        # desired positions
        q_cmd = list(q_down)
        for j in swing_joints:
            q_cmd[j] = q_down[j] + A * np.sin(2.0 * np.pi * f * t)

        # strong position control
        p.setJointMotorControlArray(
            bodyUniqueId=robot,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_cmd,
            # positionGains=[0.3] * len(joint_indices), velocityGains=[1.0] * len(joint_indices),
            forces=[max_force] * len(joint_indices),
        )

        p.stepSimulation()

        # optional: print base roll/pitch occasionally
        if (k % 200) == 0:
            pos, orn = p.getBasePositionAndOrientation(robot)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            print(f"t={t:5.2f}  base_xy=({pos[0]:+.3f},{pos[1]:+.3f})  z={pos[2]:.3f}  roll={roll:+.2f}  pitch={pitch:+.2f}")

        time.sleep(stepsize)
        k += 1

    print("Done.")
    p.disconnect()


if __name__ == "__main__":
    main()
