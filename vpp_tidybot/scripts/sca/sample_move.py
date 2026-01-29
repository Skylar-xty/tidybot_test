# # sample_move.py
# import time
# import numpy as np
# import pybullet as p
# import os
# import sys

# # Ensure we can import Panda_toppling from current directory
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# try:
#     from Panda_toppling import Panda
# except ImportError:
#     # Fallback if in a subdirectory
#     sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#     from Panda_toppling import Panda


# def main():
#     # -----------------------------
#     # Parameters
#     # -----------------------------
#     stepsize = 2e-3
#     duration = 5.0

#     FORCE_MAG = 500.0
#     APPLY_TIME = 2.0
#     base_force = np.array([FORCE_MAG, 0.0, 0.0], dtype=np.float32)

#     # Arm torque safety limits
#     TAU_MAX = 200.0

#     # Base PD hold gains (for prismatic joints 0/1)
#     # 这些需要调：Kp_b 越大越“硬”，越小越“软”(更容易被推走)
#     Kp_b = 1500.0
#     Kd_b = 150.0
#     FB_MAX = 2000.0  # base force clamp (N). 适当夹住，避免数值爆炸

#     # -----------------------------
#     # Init robot
#     # -----------------------------
#     robot = Panda(stepsize=stepsize, use_fixed_base=False)
#     robot.setControlMode("torque")

#     # IMPORTANT: override Panda_toppling's zero-gravity
#     p.setGravity(0, 0, -9.81)

#     # Show base joints
#     for j in range(p.getNumJoints(robot.robot)):
#         name = p.getJointInfo(robot.robot, j)[1].decode()
#         jtype = p.getJointInfo(robot.robot, j)[2]
#         if name in ["world_to_base_x", "base_x_to_base_y"]:
#             print(j, name, jtype)

#     # Disable default motors on base joints once (we will command TORQUE_CONTROL ourselves)
#     for jid in [0, 1]:
#         p.setJointMotorControl2(robot.robot, jid, controlMode=p.VELOCITY_CONTROL, force=0)

#     # Optional: filter self-collisions
#     p.setCollisionFilterPair(robot.robot, robot.robot, 8, 6, enableCollision=0)

#     # -----------------------------
#     # Arm state (NOTE: wrapper returns ONLY 7 arm joints)
#     # -----------------------------
#     q_init_arm, qd_init_arm = robot.getJointStates()
#     dof_arm = len(q_init_arm)  # should be 7 for this wrapper
#     print(f"[INFO] Arm DOF reported by wrapper: {dof_arm}")

#     q_des_arm = np.array(q_init_arm, dtype=np.float32)

#     # Arm PD Gains
#     KP_arm = np.array([600, 600, 500, 500, 300, 200, 200], dtype=np.float32)
#     KD_arm = np.array([20, 20, 20, 20, 15, 10, 10], dtype=np.float32)

#     # -----------------------------
#     # Base desired position for PD hold
#     # -----------------------------
#     bx0 = float(p.getJointState(robot.robot, 0)[0])
#     by0 = float(p.getJointState(robot.robot, 1)[0])

#     print("[INFO] Holding arm q_des with ID + PD torque.")
#     print("[INFO] Holding base (x,y) with PD (soft) while applying external force.")
#     print(f"[INFO] Applying planar base force for first {APPLY_TIME:.2f} seconds.")

#     # -----------------------------
#     # Main loop
#     # -----------------------------
#     n_steps = int(duration / stepsize)
#     apply_steps = int(APPLY_TIME / stepsize)

#     zero_acc_arm = [0.0] * dof_arm

#     for i in range(n_steps):
#         # ---- (A) Base PD hold (joint 0/1 are prismatic) ----
#         bx, bxd = p.getJointState(robot.robot, 0)[0:2]
#         by, byd = p.getJointState(robot.robot, 1)[0:2]

#         fb_x = Kp_b * (bx0 - bx) - Kd_b * bxd
#         fb_y = Kp_b * (by0 - by) - Kd_b * byd

#         fb_x = float(np.clip(fb_x, -FB_MAX, FB_MAX))
#         fb_y = float(np.clip(fb_y, -FB_MAX, FB_MAX))

#         # Apply base forces via TORQUE_CONTROL (for prismatic this is linear force)
#         p.setJointMotorControl2(robot.robot, 0, controlMode=p.TORQUE_CONTROL, force=fb_x)
#         p.setJointMotorControl2(robot.robot, 1, controlMode=p.TORQUE_CONTROL, force=fb_y)

#         # ---- (B) Arm control (wrapper returns arm only) ----
#         q_raw, qd_raw = robot.getJointStates()
#         q_arm = np.array(q_raw, dtype=np.float32)
#         qd_arm = np.array(qd_raw, dtype=np.float32)

#         # Inverse Dynamics (may return zeros if it fails inside wrapper)
#         tau_id_list = robot.solveInverseDynamics(q_raw, qd_raw, zero_acc_arm)
#         tau_id_arm = np.array(tau_id_list, dtype=np.float32)

#         # PD
#         e_arm = q_des_arm - q_arm
#         tau_pd_arm = KP_arm * e_arm - KD_arm * qd_arm

#         # Sum and Clip
#         tau_cmd_arm = tau_id_arm + tau_pd_arm
#         tau_cmd_arm = np.clip(tau_cmd_arm, -TAU_MAX, TAU_MAX)

#         # Send to arm joints only
#         robot.setTargetTorques(tau_cmd_arm.tolist())

#         # ---- (C) Apply External Disturbance to base_link (linkIndex=1) ----
#         if i < apply_steps:
#             p.applyExternalForce(
#                 objectUniqueId=robot.robot,
#                 linkIndex=1,  # base_link
#                 forceObj=base_force.tolist(),
#                 posObj=[0, 0, 0],
#                 flags=p.WORLD_FRAME,
#             )

#         # Step sim
#         robot.step()

#         # Debug print
#         if i % 100 == 0:
#             # base positions are directly bx,by
#             print(
#                 f"t={robot.t:.3f} | base(q)=[{bx:.3f},{by:.3f}] "
#                 f"| base(qd)=[{bxd:.3f},{byd:.3f}] "
#                 f"| arm_err_norm={np.linalg.norm(e_arm):.4f} "
#                 f"| base_force_cmd=[{fb_x:.1f},{fb_y:.1f}]"
#             )

#         # Optional realtime sync
#         time.sleep(stepsize)

#     print("[INFO] Done.")


# if __name__ == "__main__":
#     main()

# sample_move.py (for unified 9-DoF Panda_toppling)
import time
import numpy as np
import pybullet as p
import os
import sys

# Ensure we can import Panda_toppling from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Panda_toppling import Panda
except ImportError:
    # Fallback if in a subdirectory
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from Panda_toppling import Panda


def main():
    # -----------------------------
    # Parameters
    # -----------------------------
    stepsize = 2e-3
    duration = 5.0

    FORCE_MAG = 500.0
    APPLY_TIME = 2.0
    base_force = np.array([FORCE_MAG, 0.0, 0.0], dtype=np.float32)

    # Clamp (base: force N, arm: torque Nm)
    TAU_MAX_ARM  = 200.0
    TAU_MAX_BASE = 2000.0

    # Base behavior
    # - "soft_hold": base is softly held near initial (x,y) with PD (can still be pushed)
    # - "free": base joints are unactuated (0 force), easiest to push
    BASE_MODE = "soft_hold"

    # Base PD hold gains (only used if BASE_MODE == "soft_hold")
    Kp_b = 1500.0
    Kd_b = 150.0
    FB_MAX = 2000.0  # base force clamp

    # Arm PD gains (7-dim for the arm only)
    KP_arm = np.array([600, 600, 500, 500, 300, 200, 200], dtype=np.float32)
    KD_arm = np.array([20, 20, 20, 20, 15, 10, 10], dtype=np.float32)

    # -----------------------------
    # Init robot (unified interface)
    # -----------------------------
    robot = Panda(stepsize=stepsize, use_fixed_base=False)
    robot.setControlMode("torque")

    # Override gravity (toppling requires gravity)
    p.setGravity(0, 0, -9.81)

    # Optional: filter self-collisions (adjust if needed)
    try:
        p.setCollisionFilterPair(robot.robot, robot.robot, 8, 6, enableCollision=0)
    except Exception:
        pass

    # -----------------------------
    # Joint partition sanity
    # -----------------------------
    if not hasattr(robot, "arm_pos_idx") or not hasattr(robot, "base_pos_idx"):
        raise RuntimeError(
            "Your Panda_toppling is not the unified version (missing arm_pos_idx/base_pos_idx). "
            "Please replace Panda_toppling.py with the unified interface version."
        )

    arm_idx = np.array(robot.arm_pos_idx, dtype=np.int64)
    base_idx = np.array(robot.base_pos_idx, dtype=np.int64)

    print("[INFO] Unified DOF =", robot.dof)
    print("[INFO] base_pos_idx =", base_idx, "len =", len(base_idx))
    print("[INFO] arm_pos_idx  =", arm_idx,  "len =", len(arm_idx))

    assert robot.dof == len(robot.joints), "robot.dof mismatch; check Panda_toppling unified code."
    assert len(base_idx) == 2, f"Expected 2 base joints, got {len(base_idx)}"
    assert len(arm_idx) == 7, f"Expected 7 arm joints, got {len(arm_idx)}"
    assert KP_arm.shape == (7,) and KD_arm.shape == (7,), "Arm gains must be length 7."

    # -----------------------------
    # Desired configuration
    # -----------------------------
    q_init, qd_init = robot.getJointStates()  # 9-dim
    q_init = np.array(q_init, dtype=np.float32)
    qd_init = np.array(qd_init, dtype=np.float32)

    # Hold arm joint angles at initial
    q_des = q_init.copy()  # 9-dim desired (we'll mainly use arm part)
    q_des_arm = q_init[arm_idx].copy()

    # Base desired position (only for soft_hold)
    bx0 = float(q_init[base_idx[0]])
    by0 = float(q_init[base_idx[1]])

    print("[INFO] Holding arm joint angles with ID + PD.")
    print(f"[INFO] Base mode = {BASE_MODE}")
    print(f"[INFO] Applying planar base force for first {APPLY_TIME:.2f} seconds.")

    # -----------------------------
    # Prebuild 9-dim gain vectors
    # -----------------------------
    KP = np.zeros(robot.dof, dtype=np.float32)
    KD = np.zeros(robot.dof, dtype=np.float32)
    KP[arm_idx] = KP_arm
    KD[arm_idx] = KD_arm

    # -----------------------------
    # Main loop
    # -----------------------------
    n_steps = int(duration / stepsize)
    apply_steps = int(APPLY_TIME / stepsize)

    zero_acc = [0.0] * robot.dof

    for i in range(n_steps):
        # 1) Read unified state
        q, qd = robot.getJointStates()
        q = np.array(q, dtype=np.float32)
        qd = np.array(qd, dtype=np.float32)

        # 2) Inverse dynamics (9-dim)
        try:
            tau_id = np.array(
                robot.solveInverseDynamics(q.tolist(), qd.tolist(), zero_acc),
                dtype=np.float32
            )
        except Exception:
            # If Bullet fails (rare), fall back to zeros
            tau_id = np.zeros(robot.dof, dtype=np.float32)

        # 3) PD term in unified space
        #    - arm: standard PD around q_des_arm
        #    - base: either soft hold PD or free (0)
        e = q_des - q
        tau_pd = KP * e - KD * qd  # only arm dims contribute (base KP/KD are 0)

        # base override
        if BASE_MODE == "soft_hold":
            bx, bxd = q[base_idx[0]], qd[base_idx[0]]
            by, byd = q[base_idx[1]], qd[base_idx[1]]

            fb_x = Kp_b * (bx0 - bx) - Kd_b * bxd
            fb_y = Kp_b * (by0 - by) - Kd_b * byd

            fb_x = float(np.clip(fb_x, -FB_MAX, FB_MAX))
            fb_y = float(np.clip(fb_y, -FB_MAX, FB_MAX))

            tau_pd[base_idx[0]] = fb_x
            tau_pd[base_idx[1]] = fb_y

        elif BASE_MODE == "free":
            tau_pd[base_idx] = 0.0
        else:
            raise ValueError('BASE_MODE must be "soft_hold" or "free"')

        # 4) Compose command + clamp
        tau_cmd = tau_id + tau_pd
        tau_cmd[base_idx] = np.clip(tau_cmd[base_idx], -TAU_MAX_BASE, TAU_MAX_BASE)
        tau_cmd[arm_idx]  = np.clip(tau_cmd[arm_idx],  -TAU_MAX_ARM,  TAU_MAX_ARM)

        # 5) Send unified torques
        robot.setTargetTorques(tau_cmd.tolist())

        # 6) Apply external disturbance to base_link (your URDF: linkIndex=1 is typically base_link)
        if i < apply_steps:
            p.applyExternalForce(
                objectUniqueId=robot.robot,
                linkIndex=1,  # base_link
                forceObj=base_force.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
            )

        # 7) Step
        robot.step()

        # Debug print
        if i % 100 == 0:
            bx, by = q[base_idx[0]], q[base_idx[1]]
            bxd, byd = qd[base_idx[0]], qd[base_idx[1]]
            arm_err = q_des_arm - q[arm_idx]
            base_cmd = tau_cmd[base_idx]
            print(
                f"t={robot.t:.3f} | base(q)=[{bx:.3f},{by:.3f}] "
                f"| base(qd)=[{bxd:.3f},{byd:.3f}] "
                f"| arm_err_norm={np.linalg.norm(arm_err):.4f} "
                f"| base_cmd=[{base_cmd[0]:.1f},{base_cmd[1]:.1f}]"
            )

        time.sleep(stepsize)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
