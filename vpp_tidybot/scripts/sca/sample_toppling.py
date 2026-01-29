# #!/usr/bin/env python3
# import pybullet as p
# import pybullet_data
# import numpy as np
# import csv
# import argparse
# import os
# import time


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Sample joint configurations; detect self-collision; apply a joint torque and observe physical response/toppling."
#     )

#     # sampling
#     parser.add_argument("--limit_sampling", action="store_true",
#                         help="Enable near-limit sampling for a subset of joints")
#     parser.add_argument("--limit_fraction", type=float, default=0.05,
#                         help="Fraction of joint range to sample near limits")
#     parser.add_argument("--limit_joints", type=int, default=6,
#                         help="Number of joints to sample near their limits")
#     parser.add_argument("--n_samples", type=int, default=800000,
#                         help="Total number of samples to generate")

#     # sim / urdf
#     parser.add_argument("--urdf_path", type=str,
#                         default="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7.urdf",
#                         help="Path to the URDF file")
#     parser.add_argument("--gui", action="store_true",
#                         help="Enable GUI visualization")

#     # biasing for collisions
#     parser.add_argument("--collision_bias", action="store_true",
#                         help="Bias sampling towards self-collisions by forcing more joints to limits")

#     # topple detection
#     parser.add_argument("--topple_thresh", type=float, default=0.8,
#                         help="Topple threshold in radians for |roll| or |pitch|. Default 0.8 (~45 deg).")
#     parser.add_argument("--base_z", type=float, default=0.00,
#                         help="Initial base height (z) used when resetting the floating base.")
#     parser.add_argument("--settle_steps", type=int, default=120,
#                         help="Steps to settle after reset before applying torque.")
#     parser.add_argument("--total_steps", type=int, default=40000,
#                         help="Total simulation steps for the torque rollout.")
#     parser.add_argument("--torque_steps", type=int, default=100,
#                         help="How many initial steps to apply torque (<= total_steps).")

#     # torque injection
#     parser.add_argument("--torque_joint", type=int, default=6,
#                         help="Which arm joint (0..N-1 in joint_indices order) to apply torque.")
#     parser.add_argument("--torque_value", type=float, default=30.0,
#                         help="Torque magnitude in N*m (for revolute joints).")
#     parser.add_argument("--torque_profile", type=str, default="constant",
#                         choices=["constant", "half_sine"],
#                         help="Torque profile during torque_steps: constant or half_sine.")
#     parser.add_argument("--apply_to_qe", action="store_true",
#                         help="If set, also run torque rollout at qe and OR the topple labels (more expensive).")

#     # output
#     parser.add_argument("--out_file", type=str, default="collision_torque_topple_results.csv",
#                         help="Output csv file name.")

#     return parser.parse_args()


# def sample_configuration(joint_position_limits, joint_velocity_limits, args):
#     # Base random sampling for all joints
#     q = [np.random.uniform(low, high) for (low, high) in joint_position_limits]
#     qd = [np.random.uniform(low, high) for (low, high) in joint_velocity_limits]

#     use_limit = args.limit_sampling
#     num_limits = args.limit_joints

#     if args.collision_bias:
#         # In collision bias mode, force more joints to limits to increase chance of self-collision.
#         if np.random.rand() < 0.5:
#             use_limit = True
#             num_limits = np.random.randint(4, len(joint_position_limits) + 1)
#         else:
#             use_limit = False

#     if not use_limit:
#         return q, qd

#     # Resample near limits for a subset of joints
#     num = min(num_limits, len(joint_position_limits))
#     idxs = np.random.choice(len(joint_position_limits), num, replace=False)
#     for j in idxs:
#         low, high = joint_position_limits[j]
#         span = high - low
#         frac = args.limit_fraction
#         if np.random.rand() < 0.4:
#             q[j] = np.random.uniform(low, low + frac * span)
#         else:
#             q[j] = np.random.uniform(high - frac * span, high)

#     return q, qd


# def clip_to_limits(q, joint_position_limits):
#     q2 = list(q)
#     for i, (low, high) in enumerate(joint_position_limits):
#         if np.isfinite(low) and np.isfinite(high):
#             q2[i] = float(np.clip(q2[i], low, high))
#         else:
#             q2[i] = float(q2[i])
#     return q2


# def reset_floating_base(robot, base_z):
#     # Reset base pose and clear velocities to avoid cross-sample contamination
#     p.resetBasePositionAndOrientation(
#         robot,
#         [0.0, 0.0, float(base_z)],
#         p.getQuaternionFromEuler([0.0, 0.0, 0.0])
#     )
#     p.resetBaseVelocity(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


# def set_joints_state(robot, joint_indices, q, qd=None):
#     if qd is None:
#         for jid, angle in zip(joint_indices, q):
#             p.resetJointState(robot, jid, targetValue=float(angle), targetVelocity=0.0)
#     else:
#         for jid, angle, vel in zip(joint_indices, q, qd):
#             p.resetJointState(robot, jid, targetValue=float(angle), targetVelocity=float(vel))


# def settle(robot, steps, gui_sleep_s=0.0):
#     for _ in range(int(steps)):
#         p.stepSimulation()
#         if gui_sleep_s > 0.0:
#             time.sleep(gui_sleep_s)


# def disable_default_motors(robot, joint_indices):
#     """
#     Critical: By default, PyBullet may have motors enabled (velocity control) for joints.
#     To apply pure torque and observe physical response, disable them.
#     """
#     for jid in joint_indices:
#         p.setJointMotorControl2(robot, jid, controlMode=p.VELOCITY_CONTROL, force=0)


# def torque_at_step(profile, t, torque_steps, torque_value):
#     if t >= torque_steps:
#         return 0.0
#     if profile == "constant":
#         return float(torque_value)
#     # half_sine: 0 -> peak -> 0 over [0, torque_steps)
#     phase = (t + 1) / float(torque_steps)  # (0,1]
#     return float(torque_value) * float(np.sin(np.pi * phase))


# def rollout_with_joint_torque(
#     robot,
#     joint_indices,
#     torque_joint_local_idx,
#     torque_value,
#     torque_steps,
#     total_steps,
#     profile,
#     topple_thresh,
#     gui_sleep_s=0.0,
# ):
#     """
#     Pure physical rollout:
#       - disable default motors
#       - apply torque on one joint for torque_steps
#       - let system evolve for total_steps
#       - detect topple by base roll/pitch threshold
#     Returns:
#       toppled (bool), topple_step (int or -1),
#       max_tilt (float), min_base_z (float), final_rpy (tuple)
#     """
#     torque_steps = int(min(torque_steps, total_steps))
#     total_steps = int(total_steps)

#     if torque_joint_local_idx < 0 or torque_joint_local_idx >= len(joint_indices):
#         raise ValueError(f"--torque_joint must be in [0, {len(joint_indices)-1}], got {torque_joint_local_idx}")

#     disable_default_motors(robot, joint_indices)

#     jid = joint_indices[int(torque_joint_local_idx)]

#     toppled = False
#     topple_step = -1
#     max_tilt = 0.0
#     min_base_z = 1e9
#     final_rpy = (0.0, 0.0, 0.0)

#     for t in range(total_steps):
#         tau = torque_at_step(profile, t, torque_steps, torque_value)

#         # Apply torque to selected joint; do NOT command other joints (they remain passive).
#         p.setJointMotorControl2(robot, jid, controlMode=p.TORQUE_CONTROL, force=float(tau))

#         p.stepSimulation()

#         pos, orn = p.getBasePositionAndOrientation(robot)
#         roll, pitch, yaw = p.getEulerFromQuaternion(orn)
#         final_rpy = (roll, pitch, yaw)

#         tilt = max(abs(roll), abs(pitch))
#         max_tilt = max(max_tilt, tilt)
#         min_base_z = min(min_base_z, float(pos[2]))

#         if tilt > topple_thresh:
#             toppled = True
#             topple_step = t
#             break

#         if gui_sleep_s > 0.0:
#             time.sleep(gui_sleep_s)

#     return toppled, topple_step, max_tilt, min_base_z, final_rpy


# def main():
#     args = parse_args()

#     # Import Panda class wrapper
#     import sys
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from Panda_toppling import Panda

#     connection_mode = p.GUI if args.gui else p.DIRECT
#     panda_robot = Panda(
#         stepsize=2e-3,
#         realtime=0,
#         connection_mode=connection_mode,
#         urdf_path=args.urdf_path,
#         use_fixed_base=False
#     )
#     robot = panda_robot.robot

#     if args.gui:
#         p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
#         p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
#         p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -9.81)

#     # Optional: ignore a specific self-collision pair (your original line)
#     p.setCollisionFilterPair(robot, robot, 8, 6, enableCollision=0)

#     # Sample only the arm joints defined by your wrapper
#     joint_indices = list(panda_robot.joints)
#     if not joint_indices:
#         raise RuntimeError(f"No movable joints found in URDF at {args.urdf_path}")

#     # Build joint limits from PyBullet
#     joint_position_limits = []
#     joint_velocity_limits = []
#     joint_torque_limits = []

#     # (min,max) acceleration limits for qe computation (your original)
#     joint_acceleration_limits = [(-15, 15), (-7.5, 7.5), (-10, 10), (-12.5, 12.5),
#                                  (-15, 15), (-20, 20), (-20, 20)]
#     if len(joint_acceleration_limits) != len(joint_indices):
#         raise RuntimeError(
#             f"joint_acceleration_limits has {len(joint_acceleration_limits)} entries, "
#             f"but joint_indices has {len(joint_indices)} joints."
#         )

#     for jid in joint_indices:
#         info = p.getJointInfo(robot, jid)
#         joint_position_limits.append((info[8], info[9]))
#         joint_velocity_limits.append((-info[11], info[11]))
#         joint_torque_limits.append((-info[10], info[10]))

#     # Output buffers
#     n_samples = int(args.n_samples)
#     pos_samples = []
#     vel_samples = []
#     end_pos = []

#     collision_flags = []
#     torque_topple_flags = []
#     torque_topple_step = []
#     torque_max_tilt = []
#     torque_min_base_z = []

#     N_collision = 0
#     N_torque_topple = 0

#     progress_every = max(1, n_samples // 10)

#     gui_sleep = 0.0
#     if args.gui:
#         gui_sleep = 0.0  # set to 0.001 if you want slow-mo

#     for idx in range(n_samples):
#         q, qd = sample_configuration(joint_position_limits, joint_velocity_limits, args)

#         # Compute qe ("stopped" pose under bounded deceleration)
#         qe = []
#         for j, vel in enumerate(qd):
#             a_max = joint_acceleration_limits[j][1]
#             if vel == 0:
#                 qe_j = q[j]
#             else:
#                 t_stop = abs(vel) / a_max
#                 delta = 0.5 * vel * t_stop
#                 qe_j = q[j] + delta
#             qe.append(qe_j)

#         q_clipped = clip_to_limits(q, joint_position_limits)
#         qe_clipped = clip_to_limits(qe, joint_position_limits)

#         pos_samples.append(q_clipped)
#         vel_samples.append(qd)
#         end_pos.append(qe_clipped)

#         # ---------------------------
#         # 1) Collision label at q
#         # ---------------------------
#         reset_floating_base(robot, args.base_z)
#         set_joints_state(robot, joint_indices, q_clipped)
#         settle(robot, args.settle_steps, gui_sleep_s=gui_sleep)
#         contacts_q = p.getContactPoints(bodyA=robot, bodyB=robot)
#         collision_q = len(contacts_q) > 0

#         # ---------------------------
#         # 2) Collision label at qe
#         # ---------------------------
#         reset_floating_base(robot, args.base_z)
#         set_joints_state(robot, joint_indices, qe_clipped)
#         settle(robot, args.settle_steps, gui_sleep_s=gui_sleep)
#         contacts_qe = p.getContactPoints(bodyA=robot, bodyB=robot)
#         collision_qe = len(contacts_qe) > 0

#         collision = bool(collision_q or collision_qe)
#         collision_flags.append(collision)
#         if collision:
#             N_collision += 1

#         # ---------------------------
#         # 3) Pure-physics torque rollout at q (and optionally qe)
#         # ---------------------------
#         def run_torque_topple_at_pose(q_pose):
#             reset_floating_base(robot, args.base_z)
#             set_joints_state(robot, joint_indices, q_pose)
#             settle(robot, args.settle_steps, gui_sleep_s=gui_sleep)

#             return rollout_with_joint_torque(
#                 robot=robot,
#                 joint_indices=joint_indices,
#                 torque_joint_local_idx=args.torque_joint,
#                 torque_value=args.torque_value,
#                 torque_steps=args.torque_steps,
#                 total_steps=args.total_steps,
#                 profile=args.torque_profile,
#                 topple_thresh=args.topple_thresh,
#                 gui_sleep_s=gui_sleep,
#             )

#         toppled, tstep, max_tilt, min_z, _ = run_torque_topple_at_pose(q_clipped)

#         if args.apply_to_qe:
#             toppled2, tstep2, max_tilt2, min_z2, _ = run_torque_topple_at_pose(qe_clipped)
#             # OR labels; keep earliest topple step if any
#             if toppled2 and (not toppled or tstep2 < tstep):
#                 toppled, tstep, max_tilt, min_z = toppled2, tstep2, max_tilt2, min_z2
#             else:
#                 max_tilt = max(max_tilt, max_tilt2)
#                 min_z = min(min_z, min_z2)

#         torque_topple_flags.append(bool(toppled))
#         torque_topple_step.append(int(tstep))
#         torque_max_tilt.append(float(max_tilt))
#         torque_min_base_z.append(float(min_z))

#         if toppled:
#             N_torque_topple += 1
#             if args.gui:
#                 print(f"[{idx}] Torque-topple detected at step {tstep}, max_tilt={max_tilt:.3f}, min_z={min_z:.4f}")
#                 time.sleep(0.02)

#         # Progress
#         if (idx + 1) % progress_every == 0:
#             print(f"Sample {idx+1}/{n_samples}: collisions={N_collision}, torque_topple={N_torque_topple}")

#     # ---------------------------
#     # Write results to CSV
#     # ---------------------------
#     out_file = args.out_file
#     with open(out_file, "w", newline="") as f:
#         writer = csv.writer(f)

#         header = (
#             [f"joint_{i}_pos" for i in range(len(joint_indices))] +
#             [f"joint_{i}_vel" for i in range(len(joint_indices))] +
#             [f"joint_{i}_final_pos" for i in range(len(joint_indices))] +
#             ["collision"] +
#             ["torque_toppled", "torque_topple_step", "torque_max_tilt", "torque_min_base_z"] +
#             ["torque_joint", "torque_value", "torque_profile", "torque_steps", "total_steps", "settle_steps", "topple_thresh"] +
#             ["apply_to_qe"]
#         )
#         writer.writerow(header)

#         for q_row, qd_row, qe_row, col_flag, top_flag, tstep, mtilt, mz in zip(
#             pos_samples, vel_samples, end_pos,
#             collision_flags, torque_topple_flags, torque_topple_step, torque_max_tilt, torque_min_base_z
#         ):
#             writer.writerow(
#                 list(q_row) + list(qd_row) + list(qe_row)
#                 + [int(col_flag)]
#                 + [int(top_flag), int(tstep), float(mtilt), float(mz)]
#                 + [int(args.torque_joint), float(args.torque_value), str(args.torque_profile),
#                    int(args.torque_steps), int(args.total_steps), int(args.settle_steps), float(args.topple_thresh)]
#                 + [int(bool(args.apply_to_qe))]
#             )

#     print(f"Done. Results saved to {out_file}")
#     p.disconnect()


# if __name__ == "__main__":
#     main()


# # sample_toppling_stable.py
# import argparse
# import time
# import numpy as np
# import pybullet as p
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Panda_toppling_2 import Panda


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--gui", action="store_true")
#     ap.add_argument("--stepsize", type=float, default=2e-3)
#     ap.add_argument("--duration", type=float, default=6.0)
#     ap.add_argument("--force", type=float, default=50.0)
#     ap.add_argument("--push_time", type=float, default=1.0)
#     ap.add_argument("--settle", type=float, default=0.3)
#     ap.add_argument("--height", type=float, default=0.6)
#     args = ap.parse_args()
#     URDF_PATH = "G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7toppling.urdf"
#     robot = Panda(
#         stepsize=args.stepsize,
#         connection_mode=(p.GUI if args.gui else p.DIRECT),
#         urdf_path=URDF_PATH,
#         use_fixed_base=False,
#         gravity=-9.81,
#     )
#     robot.setControlMode("torque")

#     # ---- Stabilize physics ----
#     p.setPhysicsEngineParameter(numSolverIterations=200, fixedTimeStep=args.stepsize, enableFileCaching=0)
#     p.changeDynamics(robot.plane, -1, lateralFriction=1.0)
#     p.changeDynamics(robot.robot, -1, lateralFriction=1.0, rollingFriction=0.001, spinningFriction=0.001)

#     # ---- Debug base dynamics ----
#     mass, _, inertia_diag, *_ = p.getDynamicsInfo(robot.robot, -1)
#     pos, quat = p.getBasePositionAndOrientation(robot.robot)
#     print("[DEBUG] base mass =", mass)
#     print("[DEBUG] base inertia diag =", inertia_diag)
#     print("[DEBUG] base quat =", quat, " | norm =", np.linalg.norm(quat))

#     if robot.base_mode != "free_base":
#         print("[WARN] Not free_base; this URDF cannot topple via roll/pitch.")

#     n_steps = int(args.duration / args.stepsize)
#     push_steps = int(args.push_time / args.stepsize)
#     settle_steps = int(args.settle / args.stepsize)
#     push_start = settle_steps
#     push_end = push_start + push_steps

#     F_world = np.array([args.force, 0.0, 0.0], dtype=np.float32)

#     for i in range(n_steps):
#         # 你自己的 arm hold（ID+PD）放这里：robot.setTargetTorques(...)
#         # 先给个最小占位：不施加任何关节力矩也能测试 base 是否会 nan
#         robot.setTargetTorques([0.0] * robot.dof_actuated)

#         if push_start <= i < push_end:
#             base_pos, _ = p.getBasePositionAndOrientation(robot.robot)
#             base_pos = np.array(base_pos, dtype=np.float32)
#             pos_world = (base_pos + np.array([0.0, 0.0, args.height], dtype=np.float32)).tolist()

#             p.applyExternalForce(
#                 robot.robot, -1,
#                 forceObj=F_world.tolist(),
#                 posObj=pos_world,
#                 flags=p.WORLD_FRAME,
#             )

#         robot.step()

#         if i % int(0.1 / args.stepsize) == 0:
#             rpy = robot.getBaseRPY()
#             print(f"t={robot.t:6.3f} | roll={np.rad2deg(rpy[0]):7.2f} | pitch={np.rad2deg(rpy[1]):7.2f}")

#         if args.gui:
#             time.sleep(args.stepsize)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
sample_toppling_fixed.py

Robust toppling test for a free-base (floating-base) robot in PyBullet.

What this script does
- Load a URDF with useFixedBase=False (i.e., floating base => roll/pitch can change).
- Put the arm into a desired joint configuration (teleport + settle by default).
- Hold the arm pose with gravity-compensated joint-space PD (arm joints only).
- Apply an external push to the BASE at a specified height to induce a tipping moment.
- Detect toppling using roll/pitch threshold (and optional base height threshold).
- Optional: run multiple random samples and write a CSV.

Why this is more stable than many quick scripts
- Disables default Bullet joint motors to avoid hidden controllers fighting yours.
- Uses ramped push force (avoids impulsive spikes that cause solver blow-ups / NaNs).
- Sets solver iterations + contact/friction parameters explicitly.
- Uses safe quaternion->RPY conversion and NaN guarding.

Notes
- In PyBullet, `calculateInverseDynamics` expects joint DOFs only (not the 6-DOF floating base state).
- If your URDF has extra virtual prismatic joints, those will appear as actuated joints; this script treats
  *all non-fixed joints* as actuated by default. Use --arm_name_prefix to restrict to arm joints.
"""

import argparse
import csv
import math
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pybullet as p
import pybullet_data

URDF_PATH="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7toppling.urdf"
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gui", action="store_true", help="Enable PyBullet GUI.")
    ap.add_argument("--stepsize", type=float, default=0.002, help="Simulation timestep (s).")
    ap.add_argument("--duration", type=float, default=5.0, help="Rollout duration (s).")

    # Base initialization
    ap.add_argument("--base_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="Initial base position (world). If your mesh intersects ground, raise z a bit.")
    ap.add_argument("--base_rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="Initial base orientation RPY (rad).")

    # Arm selection
    ap.add_argument("--arm_name_prefix", type=str, default="iiwa_joint",
                    help="Only joints whose names start with this prefix are controlled as arm joints. "
                         "Set to '' to control all non-fixed joints.")
    ap.add_argument("--q", type=str, default="",
                    help="Comma-separated desired arm joint positions (rad). If empty: use current or random.")
    ap.add_argument("--random_q", action="store_true",
                    help="If set, ignore --q and sample a random arm configuration within joint limits.")
    ap.add_argument("--teleport_to_q", action="store_true",
                    help="Teleport joints to q before rollout (recommended to avoid 'topple while moving').")

    # Arm hold (joint-space PD + optional gravity compensation)
    ap.add_argument("--hold_arm", action="store_true", help="Hold arm pose with PD torque control.")
    ap.add_argument("--kp", type=float, default=50.0, help="Arm joint-space PD Kp.")
    ap.add_argument("--kd", type=float, default=20.0, help="Arm joint-space PD Kd.")
    ap.add_argument("--tau_max", type=float, default=100.0, help="Arm torque clamp (Nm).")
    ap.add_argument("--gravity_comp", action="store_true", help="Add inverse dynamics gravity compensation.")

    # Push parameters (base)
    ap.add_argument("--push_mag", type=float, default=15.0, help="Push force magnitude (N).")
    ap.add_argument("--push_time", type=float, default=1.0, help="Push duration (s).")
    ap.add_argument("--push_dir", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Push direction (world).")
    ap.add_argument("--push_height", type=float, default=0.4,
                    help="Push application point height ABOVE base position (m). Creates tipping moment.")
    ap.add_argument("--push_ramp_time", type=float, default=0.25,
                    help="Ramp time (s) to smoothly increase push from 0 to full.")

    # Topple detection
    ap.add_argument("--topple_deg", type=float, default=35.0, help="Topple threshold on |roll| or |pitch| (deg).")
    ap.add_argument("--min_base_z", type=float, default=-1e9,
                    help="Optional: declare topple if base z drops below this value. Default disabled.")

    # Solver / contact parameters
    ap.add_argument("--gravity", type=float, default=-9.81, help="World gravity z (m/s^2).")
    ap.add_argument("--solver_iters", type=int, default=200, help="Bullet solver iterations.")
    ap.add_argument("--lateral_friction", type=float, default=8.9, help="Lateral friction for base + plane.")
    ap.add_argument("--spinning_friction", type=float, default=0.01, help="Spinning friction for base + plane.")
    ap.add_argument("--rolling_friction", type=float, default=0.01, help="Rolling friction for base + plane.")
    ap.add_argument("--lin_damping", type=float, default=0.04, help="Base linear damping.")
    ap.add_argument("--ang_damping", type=float, default=0.04, help="Base angular damping.")
    ap.add_argument("--settle_steps", type=int, default=240, help="Number of settle steps after reset/teleport.")
    ap.add_argument("--realtime", action="store_true", help="Sleep to approximate realtime.")

    # Sampling
    ap.add_argument("--n_samples", type=int, default=500, help="Run multiple samples and log CSV.")
    ap.add_argument("--csv_out", type=str, default="", help="Output CSV path (optional).")

    return ap.parse_args()


def _safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _quat_to_rpy_safe(quat) -> Tuple[float, float, float]:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        return (math.nan, math.nan, math.nan)
    # normalize to reduce drift
    n = np.linalg.norm(q)
    if n < 1e-12:
        return (math.nan, math.nan, math.nan)
    q = (q / n).tolist()
    return p.getEulerFromQuaternion(q)


def _smooth_ramp(t: float, ramp: float) -> float:
    """0->1 smoothstep-ish ramp over [0,ramp]."""
    if ramp <= 1e-9:
        return 1.0
    x = max(0.0, min(1.0, t / ramp))
    return x * x * (3.0 - 2.0 * x)


def connect(gui: bool):
    if gui:
        p.connect(p.GUI, options="--width=2048 --height=1536")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.6, cameraYaw=35, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.6]
        )
    else:
        p.connect(p.DIRECT)


def setup_world(stepsize: float, gravity_z: float, solver_iters: int):
    p.resetSimulation()
    p.setTimeStep(stepsize)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, gravity_z)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf", useFixedBase=True)

    p.setPhysicsEngineParameter(
        fixedTimeStep=stepsize,
        numSolverIterations=solver_iters,
        numSubSteps=0,
        enableConeFriction=1,
        deterministicOverlappingPairs=1,
    )
    return plane


def load_robot(urdf_path: str, base_xyz, base_rpy):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    base_quat = p.getQuaternionFromEuler(base_rpy)
    robot = p.loadURDF(
        urdf_path,
        basePosition=list(base_xyz),
        baseOrientation=base_quat,
        useFixedBase=False,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
    )
    return robot


def collect_joints(robot: int):
    n = p.getNumJoints(robot)
    joint_infos = []
    for j in range(n):
        info = p.getJointInfo(robot, j)
        joint_infos.append(info)
    return joint_infos


def pick_arm_joint_indices(joint_infos, name_prefix: str) -> List[int]:
    arm = []
    for j, info in enumerate(joint_infos):
        jtype = info[2]
        name = info[1].decode("utf-8")
        if jtype == p.JOINT_FIXED:
            continue
        if name_prefix == "" or name.startswith(name_prefix):
            arm.append(j)
    return arm


def disable_all_motors(robot: int, joint_indices: List[int]):
    # Disable Bullet's default motor on those joints (otherwise hidden velocity control fights torque control)
    for jid in joint_indices:
    # for jid in range(2):
        p.setJointMotorControl2(robot, jid, controlMode=p.VELOCITY_CONTROL, force=0.0)


def set_contact_and_damping(robot: int, plane: int, lateral: float, spin: float, roll: float, lin_damp: float, ang_damp: float):
    # plane dynamics
    p.changeDynamics(plane, -1, lateralFriction=lateral, spinningFriction=spin, rollingFriction=roll)

    # base link dynamics (linkIndex=-1)
    p.changeDynamics(robot, -1, lateralFriction=lateral, spinningFriction=spin, rollingFriction=roll,
                     linearDamping=lin_damp, angularDamping=ang_damp)

    # also set friction on all links to reduce "ice skating"
    for j in range(p.getNumJoints(robot)):
        p.changeDynamics(robot, j, lateralFriction=lateral, spinningFriction=spin, rollingFriction=roll)


def get_joint_state(robot: int, joint_indices: List[int]):
    js = p.getJointStates(robot, joint_indices)
    q = [s[0] for s in js]
    qd = [s[1] for s in js]
    return np.asarray(q, dtype=np.float64), np.asarray(qd, dtype=np.float64)


def teleport_joints(robot: int, joint_indices: List[int], q: np.ndarray, qd: Optional[np.ndarray] = None):
    if qd is None:
        qd = np.zeros_like(q)
    for jid, qi, qdi in zip(joint_indices, q.tolist(), qd.tolist()):
        p.resetJointState(robot, jid, targetValue=float(qi), targetVelocity=float(qdi))


def settle(steps: int, realtime: bool, stepsize: float):
    for _ in range(int(steps)):
        p.stepSimulation()
        if realtime:
            time.sleep(stepsize)


def calc_inv_dyn_safe(robot: int, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    try:
        tau = p.calculateInverseDynamics(robot, q.tolist(), qd.tolist(), qdd.tolist(),flags=1)
        return np.asarray(tau, dtype=np.float64)
    except Exception:
        # Not all Bullet builds support inverse dynamics for every model; PD-only will still work.
        return np.zeros_like(q, dtype=np.float64)


def rollout_one(
    robot: int,
    arm_jids: List[int],
    q_des: np.ndarray,
    stepsize: float,
    total_steps: int,
    hold_arm: bool,
    kp: float,
    kd: float,
    tau_max: float,
    gravity_comp: bool,
    push_steps: int,
    push_force_world: np.ndarray,
    push_height: float,
    push_ramp_time: float,
    topple_rad: float,
    min_base_z: float,
    realtime: bool,
):
    # state trackers
    max_tilt = 0.0
    toppled = False
    sim_failed = False
    topple_step = -1
    # 在 rollout_one 函数开头（for 之前），加一些参数
    swing_joint_local = 0          # 在 arm_jids 里的索引，0 就是第一个关节
    swing_duration = 1.0           # 挥动持续时间 (s)
    swing_amp = 0.6                # 挥动幅度 (rad)，自己根据情况调
    swing_freq = 1.5               # 频率 (Hz)
    swing_steps = int(swing_duration / stepsize)

    qdd0 = np.zeros(len(arm_jids), dtype=np.float64)

    for t in range(total_steps):
        # --- hold arm pose (torque control) ---
        if hold_arm and len(arm_jids) > 0:
            q, qd = get_joint_state(robot, arm_jids)

            # tau_gc = np.zeros_like(q)
            # if gravity_comp:
            #     tau_gc = calc_inv_dyn_safe(robot, q, qd, qdd0)

            # e = q_des - q
            # tau_pd = kp * e - kd * qd
            tau_gc = np.zeros_like(q)
            if gravity_comp:
                tau_gc = calc_inv_dyn_safe(robot, q, qd, qdd0)

            # 目标轨迹：基于 q_des，加一个时间相关的摆动
            q_target = q_des.copy()
            if t < swing_steps and len(arm_jids) > swing_joint_local:
                t_sec = t * stepsize
                q_target[swing_joint_local] = (
                    q_des[swing_joint_local]
                    + swing_amp * math.sin(2.0 * math.pi * swing_freq * t_sec)
                )

            e = q_target - q
            tau_pd = kp * e - kd * qd
            tau_cmd = tau_gc + tau_pd
            tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)

            p.setJointMotorControlArray(robot, arm_jids, controlMode=p.TORQUE_CONTROL, forces=tau_cmd.tolist())

        # --- apply push to base ---
        # if t < push_steps:
        #     # ramped push to avoid impulse instability
        #     t_sec = t * stepsize
        #     ramp = _smooth_ramp(t_sec, push_ramp_time)
        #     f = (ramp * push_force_world).tolist()

        #     base_pos, _ = p.getBasePositionAndOrientation(robot)
        #     push_pos = [base_pos[0], base_pos[1], base_pos[2] + push_height]

        #     p.applyExternalForce(
        #         objectUniqueId=robot,
        #         linkIndex=-1,  # base/root
        #         forceObj=f,
        #         posObj=push_pos,
        #         flags=p.WORLD_FRAME,
        #     )

        p.stepSimulation()

        # --- topple check ---
        pos, orn = p.getBasePositionAndOrientation(robot)
        roll, pitch, _ = _quat_to_rpy_safe(orn)
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            sim_failed = True
            toppled = False
            topple_step = t
            break

        tilt = max(abs(roll), abs(pitch))
        max_tilt = max(max_tilt, tilt)

        if (tilt > topple_rad) or (pos[2] < min_base_z):
            toppled = True
            topple_step = t
            break

        if realtime:
            time.sleep(stepsize)

    # final orientation
    _, orn = p.getBasePositionAndOrientation(robot)
    final_rpy = _quat_to_rpy_safe(orn)
    return {
        "toppled": toppled,
        "sim_failed": sim_failed,
        "topple_step": int(topple_step),
        "max_tilt_rad": float(max_tilt),
        "final_rpy": tuple(float(x) for x in final_rpy),
    }


def main():
    args = parse_args()

    connect(args.gui)
    plane = setup_world(args.stepsize, args.gravity, args.solver_iters)

    robot = load_robot(URDF_PATH, args.base_xyz, args.base_rpy)

    joint_infos = collect_joints(robot)

    print("---------------- Joint Info ----------------")
    for j, info in enumerate(joint_infos):
        print(f"ID: {j}, Name: {info[1].decode('utf-8')}, Type: {info[2]}")

    arm_jids = pick_arm_joint_indices(joint_infos, args.arm_name_prefix)
    print(f"[INFO] arm_name_prefix='{args.arm_name_prefix}', arm_jids={arm_jids}, dof_actuated={len(arm_jids)}")

    # disable default motors on arm joints (we will use torque control)
    disable_all_motors(robot, arm_jids)

    # stable contact + damping
    set_contact_and_damping(
        robot, plane,
        lateral=args.lateral_friction,
        spin=args.spinning_friction,
        roll=args.rolling_friction,
        lin_damp=args.lin_damping,
        ang_damp=args.ang_damping,
    )

    # desired arm pose
    q_now, qd_now = get_joint_state(robot, arm_jids)

    if args.random_q and len(arm_jids) > 0:
        lows, highs = [], []
        for jid in arm_jids:
            info = joint_infos[jid]
            lo, hi = float(info[8]), float(info[9])
            # some URDFs use 0,0 for continuous; guard
            if lo >= hi:
                lo, hi = -math.pi, math.pi
            lows.append(lo); highs.append(hi)
        lows = np.asarray(lows, dtype=np.float64)
        highs = np.asarray(highs, dtype=np.float64)
        q_des = lows + (highs - lows) * np.random.rand(len(arm_jids))
    elif args.q.strip() != "" and len(arm_jids) > 0:
        q_list = [float(x) for x in args.q.split(",")]
        if len(q_list) != len(arm_jids):
            raise ValueError(f"--q has {len(q_list)} values but dof_actuated={len(arm_jids)}")
        q_des = np.asarray(q_list, dtype=np.float64)
    else:
        # default: hold current joint configuration
        q_des = q_now.copy()

    if args.teleport_to_q and len(arm_jids) > 0:
        teleport_joints(robot, arm_jids, q_des, np.zeros_like(q_des))
        settle(args.settle_steps, args.realtime, args.stepsize)

    # push force
    push_dir = _safe_unit(np.asarray(args.push_dir, dtype=np.float64))
    push_force = float(args.push_mag) * push_dir

    total_steps = int(args.duration / args.stepsize)
    push_steps = int(args.push_time / args.stepsize)
    topple_rad = math.radians(float(args.topple_deg))

    # Optional CSV
    csv_writer = None
    csv_file = None
    if args.csv_out.strip() != "":
        csv_file = open(args.csv_out, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = [f"q{i}" for i in range(len(arm_jids))] + ["toppled", "sim_failed", "topple_step", "max_tilt_deg", "final_roll_deg", "final_pitch_deg"]
        csv_writer.writerow(header)

    print("[INFO] Running...")
    for k in range(int(args.n_samples)):
        # reset base pose/vel each sample
        base_quat = p.getQuaternionFromEuler(args.base_rpy)
        p.resetBasePositionAndOrientation(robot, args.base_xyz, base_quat)
        p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])

        # choose q_des per sample if random
        if args.random_q and len(arm_jids) > 0:
            lows, highs = [], []
            for jid in arm_jids:
                info = joint_infos[jid]
                lo, hi = float(info[8]), float(info[9])
                if lo >= hi:
                    lo, hi = -math.pi, math.pi
                lows.append(lo); highs.append(hi)
            lows = np.asarray(lows, dtype=np.float64)
            highs = np.asarray(highs, dtype=np.float64)
            q_des = lows + (highs - lows) * np.random.rand(len(arm_jids))

        if args.teleport_to_q and len(arm_jids) > 0:
            teleport_joints(robot, arm_jids, q_des, np.zeros_like(q_des))

        settle(args.settle_steps, args.realtime, args.stepsize)

        res = rollout_one(
            robot=robot,
            arm_jids=arm_jids,
            q_des=q_des,
            stepsize=args.stepsize,
            total_steps=total_steps,
            hold_arm=args.hold_arm,
            kp=args.kp,
            kd=args.kd,
            tau_max=args.tau_max,
            gravity_comp=args.gravity_comp,
            push_steps=push_steps,
            push_force_world=push_force,
            push_height=args.push_height,
            push_ramp_time=args.push_ramp_time,
            topple_rad=topple_rad,
            min_base_z=args.min_base_z,
            realtime=args.realtime,
        )

        max_tilt_deg = math.degrees(res["max_tilt_rad"])
        fr, fp, _ = res["final_rpy"]
        fr_deg = math.degrees(fr) if math.isfinite(fr) else math.nan
        fp_deg = math.degrees(fp) if math.isfinite(fp) else math.nan

        print(
            f"sample={k:04d} | toppled={res['toppled']} | sim_failed={res['sim_failed']} "
            f"| topple_step={res['topple_step']} | max_tilt={max_tilt_deg:6.2f} deg "
            f"| final_roll={fr_deg:7.2f} deg | final_pitch={fp_deg:7.2f} deg"
        )

        if csv_writer is not None:
            row = q_des.tolist() + [int(res["toppled"]), int(res["sim_failed"]), res["topple_step"], max_tilt_deg, fr_deg, fp_deg]
            csv_writer.writerow(row)

    if csv_file is not None:
        csv_file.close()
        print(f"[INFO] Wrote CSV to: {args.csv_out}")

    p.disconnect()


if __name__ == "__main__":
    main()
