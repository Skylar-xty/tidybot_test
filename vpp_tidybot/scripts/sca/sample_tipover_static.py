# # #!/usr/bin/env python3
# # import os
# # import time
# # import csv
# # import argparse
# # import numpy as np
# # import pybullet as p
# # import pybullet_data


# # def parse_args():
# #     parser = argparse.ArgumentParser(
# #         description="Sample joint configurations and collect TIP-OVER (toppling) labels by swinging the arm (no external force)."
# #     )

# #     # sampling
# #     parser.add_argument("--n_samples", type=int, default=2000,
# #                         help="Number of samples (each runs a rollout).")
# #     parser.add_argument("--urdf_path", type=str,
# #                         default="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7.urdf",
# #                         help="Path to URDF (must be floating-base version to allow roll/pitch).")
# #     parser.add_argument("--gui", action="store_true", help="Enable GUI.")
# #     parser.add_argument("--limit_sampling", action="store_true",
# #                         help="Sample some joints near limits.")
# #     parser.add_argument("--limit_fraction", type=float, default=0.05,
# #                         help="Fraction of joint range to sample near limits.")
# #     parser.add_argument("--limit_joints", type=int, default=3,
# #                         help="How many joints to bias near limits.")
# #     parser.add_argument("--collision_bias", action="store_true",
# #                         help="More aggressive near-limit sampling (may increase instability).")
# #     parser.add_argument("--hold_force", type=float, default=20000.0,
# #                         help="Motor force used to HOLD q_hold during settle/observe.")

# #     # swing rollout
# #     parser.add_argument("--t_swing", type=float, default=3.0,
# #                         help="Seconds of sinusoidal swing.")
# #     parser.add_argument("--t_observe", type=float, default=10.5,
# #                         help="Seconds to observe after swing (hold pose).")
# #     parser.add_argument("--A", type=float, default=1.0,
# #                         help="Swing amplitude (rad). Start with 0.6~1.2; too large can violate joint limits.")
# #     parser.add_argument("--f", type=float, default=2.0,
# #                         help="Swing frequency (Hz). Try 1~3.")
# #     parser.add_argument("--swing_joints", type=str, default="1",
# #                         help="Comma-separated indices in [0..6] to swing, e.g. '0,1' or '1'.")
# #     parser.add_argument("--max_force", type=float, default=0.0,
# #                         help="Max force per joint for POSITION_CONTROL. Must be >0 to move.")
# #     parser.add_argument("--settle_steps", type=int, default=500,
# #                         help="Steps to settle after reset, to stabilize contacts.")
# #     parser.add_argument("--t_settle", type=float, default=10.0,
# #                         help="Seconds to settle after reset before swing.")
# #     parser.add_argument("--t_rest", type=float, default=20.0,
# #                         help="Extra seconds to keep sim running after observe (can catch delayed topples).")

# #     # toppling criteria / reset
# #     parser.add_argument("--roll_pitch_thresh_deg", type=float, default=45.0,
# #                         help="Topple if |roll| or |pitch| exceeds this threshold (deg).")
# #     parser.add_argument("--z_min", type=float, default=0.0,
# #                         help="Topple if base z drops below this threshold.")
# #     parser.add_argument("--base_z0", type=float, default=0.0,
# #                         help="Reset base z (start height). Increase if penetration happens.")
# #     parser.add_argument("--lateral_friction", type=float, default=1.0,
# #                         help="Friction for plane and base.")
# #     return parser.parse_args()


# # def parse_swing_joints(s: str):
# #     parts = [x.strip() for x in s.split(",") if x.strip() != ""]
# #     js = [int(x) for x in parts]
# #     # validate within 0..6
# #     for j in js:
# #         if j < 0 or j > 6:
# #             raise ValueError(f"swing_joints contains {j}, but must be in [0..6].")
# #     return tuple(js)


# # def sample_configuration(joint_position_limits, joint_velocity_limits, args):
# #     """
# #     Sample q (positions) and qd (velocities). For this tip-over script, we only use q as q_hold.
# #     """
# #     q = [np.random.uniform(low, high) for (low, high) in joint_position_limits]
# #     qd = [np.random.uniform(low, high) for (low, high) in joint_velocity_limits]

# #     use_limit = args.limit_sampling
# #     num_limits = args.limit_joints

# #     if args.collision_bias:
# #         if np.random.rand() < 0.5:
# #             use_limit = True
# #             num_limits = np.random.randint(max(1, args.limit_joints), len(joint_position_limits) + 1)
# #         else:
# #             use_limit = False

# #     if not use_limit:
# #         return q, qd

# #     num = min(num_limits, len(joint_position_limits))
# #     idxs = np.random.choice(len(joint_position_limits), num, replace=False)
# #     for j in idxs:
# #         low, high = joint_position_limits[j]
# #         span = high - low
# #         frac = args.limit_fraction
# #         if np.random.rand() < 0.5:
# #             q[j] = np.random.uniform(low, low + frac * span)
# #         else:
# #             q[j] = np.random.uniform(high - frac * span, high)

# #     return q, qd


# # # def reset_and_settle(robot_id, joint_indices, q_hold, base_z0, settle_steps):
# # def reset_and_settle(robot_id, joint_indices, q_hold, base_z0, t_settle,step_size, hold_force):
# #     """
# #     Reset base pose/vel and joints to q_hold, disable motors, then settle.
# #     This improves repeatability across samples.
# #     """
# #     # disable motors to avoid leftover commands pushing during settle
# #     # p.setJointMotorControlArray(
# #     #     bodyUniqueId=robot_id,
# #     #     jointIndices=joint_indices,
# #     #     controlMode=p.VELOCITY_CONTROL,
# #     #     targetVelocities=[0.0] * len(joint_indices),
# #     #     forces=[0.0] * len(joint_indices),
# #     # )

# #     p.resetBasePositionAndOrientation(
# #         robot_id,
# #         [0.0, 0.0, float(base_z0)],
# #         p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
# #     )
# #     p.resetBaseVelocity(robot_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

# #     for jid, q in zip(joint_indices, q_hold):
# #         p.resetJointState(robot_id, jid, targetValue=float(q), targetVelocity=0.0)

# #     # for _ in range(int(settle_steps)):
# #     # for _ in range(int(float(t_settle) / step_size)):
# #     n = max(1, int(float(t_settle) / float(step_size)))
# #     q_hold_f = list(map(float, q_hold))
# #     for _ in range(n):
# #         p.setJointMotorControlArray(
# #             bodyUniqueId=robot_id,
# #             jointIndices=joint_indices,
# #             controlMode=p.POSITION_CONTROL,
# #             targetPositions=q_hold_f,
# #             forces=[float(hold_force)] * len(joint_indices),
# #         )
# #         p.stepSimulation()


# # def did_topple(robot_id, roll_pitch_thresh_rad, z_min):
# #     pos, orn = p.getBasePositionAndOrientation(robot_id)
# #     roll, pitch, yaw = p.getEulerFromQuaternion(orn)
# #     if abs(roll) > roll_pitch_thresh_rad or abs(pitch) > roll_pitch_thresh_rad:
# #         return True, (pos, (roll, pitch, yaw))
# #     if pos[2] < z_min:
# #         return True, (pos, (roll, pitch, yaw))
# #     return False, (pos, (roll, pitch, yaw))

# # def rollout_swing_topple(robot_id, joint_indices, q_hold, swing_joints, A, f, t_swing, t_observe,
# #                          max_force, stepsize, roll_pitch_thresh_deg, z_min, base_z0, t_settle, t_rest,hold_force):
# # # def rollout_swing_topple(robot_id, joint_indices, q_hold, swing_joints, A, f, t_swing, t_observe,
# # #                          max_force, stepsize, roll_pitch_thresh_deg, z_min, base_z0, settle_steps):
# #     """
# #     Swing selected joints sinusoidally around q_hold using POSITION_CONTROL.
# #     Returns: topple(bool), t_topple(float or -1), end_pos(list3), end_rpy(tuple3)
# #     """
# #     reset_and_settle(robot_id, joint_indices, q_hold, base_z0, t_settle,stepsize, hold_force)

# #     roll_pitch_thresh = np.deg2rad(roll_pitch_thresh_deg)

# #     # swing
# #     n_swing = max(1, int(t_swing / stepsize))
# #     for k in range(n_swing):
# #         t = (k + 1) * stepsize
# #         q_cmd = list(map(float, q_hold))
# #         for j in swing_joints:
# #             q_cmd[j] = float(q_hold[j]) + float(A) * np.sin(2.0 * np.pi * float(f) * t)

# #         p.setJointMotorControlArray(
# #             bodyUniqueId=robot_id,
# #             jointIndices=joint_indices,
# #             controlMode=p.POSITION_CONTROL,
# #             targetPositions=q_cmd,
# #             forces=[float(max_force)] * len(joint_indices),
# #         )
# #         p.stepSimulation()

# #         topple, (pos, rpy) = did_topple(robot_id, roll_pitch_thresh, z_min)
# #         if topple:
# #             return True, float(t), list(pos), tuple(rpy)

# #     # observe (hold)
# #     n_obs = max(1, int(t_observe / stepsize))
# #     p.setJointMotorControlArray(
# #         bodyUniqueId=robot_id,
# #         jointIndices=joint_indices,
# #         controlMode=p.POSITION_CONTROL,
# #         targetPositions=list(map(float, q_hold)),
# #         forces=[float(max_force)] * len(joint_indices),
# #     )
# #     for k in range(n_obs):
# #         t = float(t_swing) + (k + 1) * stepsize
# #         p.stepSimulation()

# #         topple, (pos, rpy) = did_topple(robot_id, roll_pitch_thresh, z_min)
# #         if topple:
# #             return True, float(t), list(pos), tuple(rpy)
# #     # rest phase (keep sim running, hold last command) to catch delayed topples
# #     n_rest = max(1, int(float(t_rest) / float(stepsize)))
# #     for k in range(n_rest):
# #         t = float(t_swing) + float(t_observe) + (k + 1) * stepsize
# #         p.stepSimulation()
# #         topple, (pos, rpy) = did_topple(robot_id, roll_pitch_thresh, z_min)
# #         if topple:
# #             return True, float(t), list(pos), tuple(rpy)
# #     # not toppled
# #     pos, orn = p.getBasePositionAndOrientation(robot_id)
# #     roll, pitch, yaw = p.getEulerFromQuaternion(orn)
# #     return False, -1.0, list(pos), (roll, pitch, yaw)

# # # def did_topple_static(robot_id, roll_pitch_thresh_deg=45.0, z_min=0.0):
# # #     """Return (topple: bool, pos: tuple3, rpy: tuple3)."""
# # #     pos, orn = p.getBasePositionAndOrientation(robot_id)
# # #     roll, pitch, yaw = p.getEulerFromQuaternion(orn)
# # #     thr = np.deg2rad(float(roll_pitch_thresh_deg))
# # #     topple = (abs(roll) > thr) or (abs(pitch) > thr) or (pos[2] < float(z_min))
# # #     return topple, pos, (roll, pitch, yaw)
# # def did_topple_rp_only(robot_id, roll_pitch_thresh_deg=45.0):
# #     pos, orn = p.getBasePositionAndOrientation(robot_id)
# #     roll, pitch, yaw = p.getEulerFromQuaternion(orn)
# #     thr = np.deg2rad(float(roll_pitch_thresh_deg))
# #     topple = (abs(roll) > thr) or (abs(pitch) > thr)
# #     return topple, pos, (roll, pitch, yaw)


# # def rollout_static_topple(
# #     robot_id,
# #     joint_indices,
# #     q_hold,
# #     stepsize,
# #     t_settle=2.0,
# #     t_observe=10.0,
# #     t_rest=0.0,
# #     hold_force=20000.0,
# #     base_pos=(0.0, 0.0, 0.0),
# #     base_rpy=(0.0, 0.0, 0.0),
# #     roll_pitch_thresh_deg=45.0,
# #     z_min=0.0,
# #     gui=False,
# #     gui_sleep=0.0,
# # ):
# #     """
# #     Static rollout (no motion): reset to q_hold, hold it, and observe toppling.
# #     Returns:
# #         topple(bool), t_topple(float or -1), end_xyz(list3), end_rpy(tuple3)
# #     """

# #     # 1) Reset base pose/vel (optional but recommended for repeatability)
# #     p.resetBasePositionAndOrientation(
# #         robot_id,
# #         list(map(float, base_pos)),
# #         p.getQuaternionFromEuler(list(map(float, base_rpy))),
# #     )
# #     p.resetBaseVelocity(robot_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

# #     # 2) Reset joints to q_hold (static pose)
# #     for jid, angle in zip(joint_indices, q_hold):
# #         p.resetJointState(robot_id, jid, float(angle), targetVelocity=0.0)

# #     # Immediate check right after reset (optional but useful)
# #     p.stepSimulation()
# #     if gui and gui_sleep > 0:
# #         time.sleep(gui_sleep)
# #     topple, pos, rpy = did_topple_rp_only(robot_id, roll_pitch_thresh_deg)

# #     # topple, pos, rpy = did_topple_static(robot_id, roll_pitch_thresh_deg, z_min)
# #     if topple:
# #         return True, 0.0, list(pos), tuple(rpy)

# #     q_hold_f = list(map(float, q_hold))
# #     forces = [float(hold_force)] * len(joint_indices)

# #     # 3) Settle phase: hold pose for a short while to let contacts stabilize
# #     n_settle = max(1, int(float(t_settle) / float(stepsize)))
# #     for _ in range(n_settle):
# #         p.setJointMotorControlArray(
# #             bodyUniqueId=robot_id,
# #             jointIndices=joint_indices,
# #             controlMode=p.POSITION_CONTROL,
# #             targetPositions=q_hold_f,
# #             forces=forces,
# #         )
# #         p.stepSimulation()
# #         if gui and gui_sleep > 0:
# #             time.sleep(gui_sleep)
# #         topple, pos, rpy = did_topple_rp_only(robot_id, roll_pitch_thresh_deg)
# #         # topple, pos, rpy = did_topple_static(robot_id, roll_pitch_thresh_deg, z_min)
# #         if topple:
# #             # topple time counted from start of settle as well;你也可以改成只从 observe 计
# #             t = float((_ + 1) * stepsize)
# #             return True, t, list(pos), tuple(rpy)

# #     # 4) Observe phase: keep holding q_hold and check toppling
# #     n_obs = max(1, int(float(t_observe) / float(stepsize)))
# #     for k in range(n_obs):
# #         p.setJointMotorControlArray(
# #             bodyUniqueId=robot_id,
# #             jointIndices=joint_indices,
# #             controlMode=p.POSITION_CONTROL,
# #             targetPositions=q_hold_f,
# #             forces=forces,
# #         )
# #         p.stepSimulation()
# #         if gui and gui_sleep > 0:
# #             time.sleep(gui_sleep)
# #         topple, pos, rpy = did_topple_rp_only(robot_id, roll_pitch_thresh_deg)

# #         # topple, pos, rpy = did_topple_static(robot_id, roll_pitch_thresh_deg, z_min)
# #         if topple:
# #             t = float(t_settle) + float((k + 1) * stepsize)
# #             return True, t, list(pos), tuple(rpy)

# #     # 5) Rest phase: optional extra time to catch delayed topple
# #     n_rest = max(0, int(float(t_rest) / float(stepsize)))
# #     for k in range(n_rest):
# #         p.setJointMotorControlArray(
# #             bodyUniqueId=robot_id,
# #             jointIndices=joint_indices,
# #             controlMode=p.POSITION_CONTROL,
# #             targetPositions=q_hold_f,
# #             forces=forces,
# #         )
# #         p.stepSimulation()
# #         if gui and gui_sleep > 0:
# #             time.sleep(gui_sleep)
# #         topple, pos, rpy = did_topple_rp_only(robot_id, roll_pitch_thresh_deg)

# #         # topple, pos, rpy = did_topple_static(robot_id, roll_pitch_thresh_deg, z_min)
# #         if topple:
# #             t = float(t_settle) + float(t_observe) + float((k + 1) * stepsize)
# #             return True, t, list(pos), tuple(rpy)

# #     # Not toppled: return final base state
# #     topple, pos, rpy = did_topple_rp_only(robot_id, roll_pitch_thresh_deg)

# #     # topple, pos, rpy = did_topple_static(robot_id, roll_pitch_thresh_deg, z_min)
# #     return False, -1.0, list(pos), tuple(rpy)


# # def main():
# #     args = parse_args()
# #     swing_joints = parse_swing_joints(args.swing_joints)

# #     # Import Panda wrapper
# #     import sys
# #     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #     from Panda_toppling import Panda

# #     connection_mode = p.GUI if args.gui else p.DIRECT
# #     panda_robot = Panda(
# #         stepsize=2e-3,
# #         realtime=0,
# #         connection_mode=connection_mode,
# #         urdf_path=args.urdf_path,
# #         use_fixed_base=False,
# #     )
# #     robot = panda_robot.robot
# #     stepsize = panda_robot.stepsize

# #     if args.gui:
# #         p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
# #         p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
# #         p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

# #     p.setAdditionalSearchPath(pybullet_data.getDataPath())
# #     p.setGravity(0, 0, -9.81)

# #     # friction
# #     if hasattr(panda_robot, "plane"):
# #         p.changeDynamics(panda_robot.plane, -1,
# #                          lateralFriction=float(args.lateral_friction),
# #                          restitution=0.0)
# #     p.changeDynamics(robot, -1,
# #                      lateralFriction=float(args.lateral_friction),
# #                      restitution=0.0,
# #                      linearDamping=0.0,
# #                      angularDamping=0.0)

# #     # arm joints as in Panda_toppling
# #     joint_indices = list(panda_robot.joints)
# #     print("Sampling joint indices:", joint_indices)
# #     if not joint_indices or len(joint_indices) != 7:
# #         raise RuntimeError(f"Expected 7 arm joints, got {len(joint_indices)}: {joint_indices}")

# #     # joint limits for sampling
# #     joint_position_limits = []
# #     joint_velocity_limits = []
# #     for jid in joint_indices:
# #         info = p.getJointInfo(robot, jid)
# #         joint_position_limits.append((info[8], info[9]))
# #         joint_velocity_limits.append((-info[11], info[11]))

# #     # storage
# #     n_samples = args.n_samples
# #     q_list = []
# #     topple_list = []
# #     t_topple_list = []
# #     base_end_xyz = []
# #     base_end_rpy = []

# #     prog_every = max(1, n_samples // 10)
# #     n_topple = 0

# #     for idx in range(n_samples):
# #         q_hold, _ = sample_configuration(joint_position_limits, joint_velocity_limits, args)
# #         topple, t_topple, end_xyz, end_rpy = rollout_static_topple(
# #             robot_id=robot,
# #             joint_indices=joint_indices,
# #             q_hold=q_hold,
# #             stepsize=panda_robot.stepsize,
# #             t_settle=2.0,
# #             t_observe=10.0,
# #             t_rest=5.0,
# #             # hold_force=2000.0,
# #             hold_force=2000.0,
# #             base_pos=(0.0, 0.0, 0.0),   # 需要的话可改 base_z0
# #             base_rpy=(0.0, 0.0, 0.0),
# #             roll_pitch_thresh_deg=45.0,
# #             z_min=0.0,
# #             gui=args.gui,
# #             gui_sleep=0.0,             # GUI 想慢点可以设 0.01
# #         )

# #         # topple, t_topple, end_xyz, end_rpy = rollout_swing_topple(
# #         #     robot_id=robot,
# #         #     joint_indices=joint_indices,
# #         #     q_hold=q_hold,
# #         #     swing_joints=swing_joints,
# #         #     A=args.A,
# #         #     f=args.f,
# #         #     t_swing=args.t_swing,
# #         #     t_observe=args.t_observe,
# #         #     max_force=args.max_force,
# #         #     stepsize=stepsize,
# #         #     roll_pitch_thresh_deg=args.roll_pitch_thresh_deg,
# #         #     z_min=args.z_min,
# #         #     base_z0=args.base_z0,
# #         #     t_settle=args.t_settle,
# #         #     t_rest=args.t_rest,
# #         #     hold_force=args.hold_force,
# #         # )

# #         q_list.append(q_hold)
# #         topple_list.append(int(topple))
# #         t_topple_list.append(float(t_topple))
# #         base_end_xyz.append(end_xyz)
# #         base_end_rpy.append(end_rpy)

# #         if topple:
# #             n_topple += 1
# #             if args.gui:
# #                 print(f"[TOPPLE] sample {idx} at t={t_topple:.3f}s  end_rpy={end_rpy}")

# #         if (idx + 1) % prog_every == 0:
# #             print(f"Sample {idx+1}/{n_samples}: topples so far = {n_topple}")

# #     out_file = f"tipover_swing_results_N{n_samples}_A{args.A}_f{args.f}_F{args.max_force}.csv"
# #     with open(out_file, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         header = (
# #             [f"joint_{i}_pos" for i in range(7)] +
# #             ["topple", "t_topple"] +
# #             ["base_end_x", "base_end_y", "base_end_z"] +
# #             ["base_end_roll", "base_end_pitch", "base_end_yaw"]
# #         )
# #         writer.writerow(header)
# #         for q, flag, tt, xyz, rpy in zip(q_list, topple_list, t_topple_list, base_end_xyz, base_end_rpy):
# #             writer.writerow(list(q) + [flag, tt] + list(xyz) + [rpy[0], rpy[1], rpy[2]])

# #     print(f"Done. Results saved to {out_file}")
# #     p.disconnect()


# # if __name__ == "__main__":
# #     main()


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
#         description="Sample joint configurations and detect TOPPLING during max-deceleration-to-stop rollout."
#     )

#     # sampling
#     parser.add_argument("--limit_sampling", action="store_true",
#                         help="Enable near-limit sampling for a subset of joints")
#     parser.add_argument("--limit_fraction", type=float, default=0.05,
#                         help="Fraction of joint range to sample near limits")
#     parser.add_argument("--limit_joints", type=int, default=6,
#                         help="Number of joints to sample near their limits")
#     parser.add_argument("--collision_bias", action="store_true",
#                         help="Bias sampling more aggressively to limits (same idea as your script)")
#     parser.add_argument("--n_samples", type=int, default=200000,
#                         help="Total number of samples to generate")

#     # robot / sim
#     parser.add_argument("--urdf_path", type=str,
#                         default="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7.urdf",
#                         help="Path to URDF")
#     parser.add_argument("--gui", action="store_true",
#                         help="Enable GUI visualization")
#     parser.add_argument("--base_z0", type=float, default=0.05,
#                         help="Reset base initial height. Use >0 to avoid initial penetration impulses.")
#     parser.add_argument("--lateral_friction", type=float, default=1.0,
#                         help="Friction for plane and base")
#     parser.add_argument("--settle_time", type=float, default=1.0,
#                         help="Seconds to hold q (zero vel) before starting decel rollout")

#     # dynamics / control
#     parser.add_argument("--move_force", type=float, default=3000.0,
#                         help="Motor force used during deceleration tracking")
#     parser.add_argument("--hold_force", type=float, default=3000.0,
#                         help="Motor force used when holding poses (settle/observe/rest)")
#     parser.add_argument("--observe_time", type=float, default=2.0,
#                         help="Seconds to observe after stopping at qe")
#     parser.add_argument("--rest_time", type=float, default=0.0,
#                         help="Extra seconds after observe to catch delayed topples")

#     # toppling criteria
#     parser.add_argument("--roll_pitch_thresh_deg", type=float, default=45.0,
#                         help="Topple if |roll| or |pitch| exceeds this threshold (deg)")
#     parser.add_argument("--debounce_steps", type=int, default=5,
#                         help="Require threshold exceeded for this many consecutive steps to count as topple")

#     # output
#     parser.add_argument("--out_csv", type=str, default="topple_stop_results.csv",
#                         help="Output CSV filename")

#     return parser.parse_args()


# def sample_configuration(joint_position_limits, joint_velocity_limits, args):
#     """
#     Same sampling pattern as your collision script:
#     - Uniform sampling by default
#     - If limit_sampling: randomly pick some joints and sample near limits
#     - If collision_bias: more aggressive limit sampling stochastically
#     """
#     q = [np.random.uniform(low, high) for (low, high) in joint_position_limits]
#     qd = [np.random.uniform(low, high) for (low, high) in joint_velocity_limits]

#     use_limit = args.limit_sampling
#     num_limits = args.limit_joints

#     if args.collision_bias:
#         if np.random.rand() < 0.5:
#             use_limit = True
#             num_limits = np.random.randint(4, len(joint_position_limits) + 1)
#         else:
#             use_limit = False

#     if not use_limit:
#         return q, qd

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


# def did_topple_rp(robot_id, roll_pitch_thresh_deg):
#     """Topple check based ONLY on roll/pitch (robust; avoids z misclassification)."""
#     pos, orn = p.getBasePositionAndOrientation(robot_id)
#     roll, pitch, yaw = p.getEulerFromQuaternion(orn)
#     thr = np.deg2rad(float(roll_pitch_thresh_deg))
#     topple = (abs(roll) > thr) or (abs(pitch) > thr)
#     return topple, pos, (roll, pitch, yaw)


# def compute_stop_trajectory(q0, qd0, acc_limits):
#     """
#     For each joint:
#       a = -sign(qd0)*a_max
#       stop time t_stop = |qd0|/a_max
#       qe = q0 + 0.5*qd0*t_stop  (same as your collision script's delta = 0.5*v*t_stop)
#     Returns:
#       qe (list), t_stop_each (list), T_stop (float)
#     """
#     qe = []
#     t_stop_each = []
#     for j, v0 in enumerate(qd0):
#         a_max = float(acc_limits[j][1])
#         v0 = float(v0)
#         if abs(v0) < 1e-12:
#             t_stop = 0.0
#             qe_j = float(q0[j])
#         else:
#             t_stop = abs(v0) / a_max
#             qe_j = float(q0[j]) + 0.5 * v0 * t_stop
#         qe.append(qe_j)
#         t_stop_each.append(t_stop)
#     T_stop = float(max(t_stop_each)) if len(t_stop_each) > 0 else 0.0
#     return qe, t_stop_each, T_stop


# def desired_qv_at_time(q0, qd0, acc_limits, t):
#     """
#     Piecewise per-joint:
#       if t <= t_stop:
#          q(t) = q0 + v0*t + 0.5*a*t^2,  v(t) = v0 + a*t
#       else:
#          q(t) = qe, v(t)=0
#     where a = -sign(v0)*a_max
#     """
#     q_des = []
#     v_des = []
#     for j, v0 in enumerate(qd0):
#         a_max = float(acc_limits[j][1])
#         v0 = float(v0)
#         q0j = float(q0[j])
#         if abs(v0) < 1e-12:
#             q_des.append(q0j)
#             v_des.append(0.0)
#             continue

#         t_stop = abs(v0) / a_max
#         sgn = 1.0 if v0 > 0 else -1.0
#         a = -sgn * a_max  # opposite to velocity

#         if t <= t_stop:
#             q_t = q0j + v0 * t + 0.5 * a * t * t
#             v_t = v0 + a * t
#         else:
#             q_t = q0j + 0.5 * v0 * t_stop
#             v_t = 0.0

#         q_des.append(float(q_t))
#         v_des.append(float(v_t))

#     return q_des, v_des


# def reset_base_and_joints(robot_id, joint_indices, q, base_z0):
#     # base
#     p.resetBasePositionAndOrientation(
#         robot_id,
#         [0.0, 0.0, float(base_z0)],
#         p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
#     )
#     p.resetBaseVelocity(robot_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

#     # joints
#     for jid, angle in zip(joint_indices, q):
#         p.resetJointState(robot_id, jid, targetValue=float(angle), targetVelocity=0.0)

#     p.stepSimulation()


# def hold_pose(robot_id, joint_indices, q_hold, stepsize, seconds, hold_force,
#               roll_pitch_thresh_deg, debounce_steps, gui=False):
#     """
#     Hold q_hold for some duration, checking toppling with debounce.
#     Returns topple flag and time.
#     """
#     n = max(1, int(float(seconds) / float(stepsize)))
#     q_hold_f = list(map(float, q_hold))
#     forces = [float(hold_force)] * len(joint_indices)

#     over_cnt = 0
#     for k in range(n):
#         p.setJointMotorControlArray(
#             bodyUniqueId=robot_id,
#             jointIndices=joint_indices,
#             controlMode=p.POSITION_CONTROL,
#             targetPositions=q_hold_f,
#             forces=forces,
#         )
#         p.stepSimulation()
#         if gui:
#             time.sleep(0.0)

#         topple, pos, rpy = did_topple_rp(robot_id, roll_pitch_thresh_deg)
#         if topple:
#             over_cnt += 1
#         else:
#             over_cnt = 0

#         if over_cnt >= int(debounce_steps):
#             t = float((k + 1) * stepsize)
#             return True, t, list(pos), tuple(rpy)

#     topple, pos, rpy = did_topple_rp(robot_id, roll_pitch_thresh_deg)
#     return False, -1.0, list(pos), tuple(rpy)

# def passive_warmup_and_check_topple(
#     robot_id,
#     stepsize,
#     warm_steps=20,
#     check_steps=500,
#     roll_pitch_thresh_deg=60.0,
#     debounce_steps=10,
#     gui=False,
# ):
#     """
#     No control / passive sim stepping.
#     - warm_steps: steps to skip judgement (let contacts settle)
#     - check_steps: steps to actually judge toppling
#     """
#     thr = np.deg2rad(float(roll_pitch_thresh_deg))
#     over = 0

#     # warm-up: no judgement
#     for _ in range(int(warm_steps)):
#         p.stepSimulation()
#         if gui:
#             time.sleep(0.0)

#     # check window
#     for k in range(int(check_steps)):
#         p.stepSimulation()
#         if gui:
#             time.sleep(0.0)

#         pos, orn = p.getBasePositionAndOrientation(robot_id)
#         roll, pitch, yaw = p.getEulerFromQuaternion(orn)

#         topple_now = (abs(roll) > thr) or (abs(pitch) > thr)
#         over = over + 1 if topple_now else 0

#         if over >= int(debounce_steps):
#             t = float((k + 1) * stepsize)
#             return True, t, list(pos), (roll, pitch, yaw)

#     pos, orn = p.getBasePositionAndOrientation(robot_id)
#     roll, pitch, yaw = p.getEulerFromQuaternion(orn)
#     return False, -1.0, list(pos), (roll, pitch, yaw)

# def wait_for_user(gui=False):
#     n_tail = 10000
#     for _ in range(n_tail):
#         p.stepSimulation()
#         if gui:
#             time.sleep(0.0)
# def rollout_stop_topple(robot_id, joint_indices, q0, qd0,
#                        acc_limits, stepsize,
#                        base_z0, settle_time,
#                        move_force, hold_force,
#                        observe_time, rest_time,
#                        roll_pitch_thresh_deg, debounce_steps,
#                        gui=False):
#     """
#     Rollout:
#       1) Reset to q0, base_z0, settle (hold q0 with 0 vel)
#       2) Set initial joint velocities to qd0
#       3) Track the max-decel-to-stop trajectory q(t) until all joints stopped
#          (per-joint a = -sign(v0)*a_max)
#          During this phase, check toppling
#       4) Hold qe for observe_time + rest_time, check toppling
#     Returns:
#       topple(bool), t_topple(float or -1), phase(str), qe(list),
#       end_xyz(list3), end_rpy(tuple3)
#     """
#     # reset
#     reset_base_and_joints(robot_id, joint_indices, q0, base_z0)

#     # # settle at q0
#     # topple, t_topple, end_xyz, end_rpy = hold_pose(
#     #     robot_id, joint_indices, q0, stepsize, settle_time, hold_force,
#     #     roll_pitch_thresh_deg, debounce_steps, gui=gui
#     # )

#     # PASSIVE settle (self-collision style): warm-up only, no motor hold
#     topple, t_topple, end_xyz, end_rpy = passive_warmup_and_check_topple(
#         robot_id=robot_id,
#         stepsize=stepsize,
#         warm_steps=30,          # 你可以 20~100 试
#         check_steps=0,          # settle 阶段只 warm-up，不做 topple 判断
#         roll_pitch_thresh_deg=roll_pitch_thresh_deg,
#         debounce_steps=debounce_steps,
#         gui=gui,
#     )
#     if topple:
#         return True, t_topple, "settle", None, end_xyz, end_rpy

#     # set initial joint velocities to qd0 at the *current* q0
#     for jid, angle, vel in zip(joint_indices, q0, qd0):
#         p.resetJointState(robot_id, jid, targetValue=float(angle), targetVelocity=float(vel))
#     p.stepSimulation()

#     # compute qe and stop horizon
#     qe, t_stop_each, T_stop = compute_stop_trajectory(q0, qd0, acc_limits)

#     # deceleration phase (track q(t))
#     n_stop = max(1, int(float(T_stop) / float(stepsize))) if T_stop > 0 else 1
#     forces = [float(move_force)] * len(joint_indices)

#     over_cnt = 0
#     for k in range(n_stop):
#         t = float((k + 1) * stepsize)
#         q_des, v_des = desired_qv_at_time(q0, qd0, acc_limits, t)

#         p.setJointMotorControlArray(
#             bodyUniqueId=robot_id,
#             jointIndices=joint_indices,
#             controlMode=p.POSITION_CONTROL,
#             targetPositions=q_des,
#             targetVelocities=v_des,
#             forces=forces,
#         )
#         p.stepSimulation()
#         if gui:
#             time.sleep(0.0)

#         topple_now, pos, rpy = did_topple_rp(robot_id, roll_pitch_thresh_deg)
#         if topple_now:
#             over_cnt += 1
#         else:
#             over_cnt = 0

#         if over_cnt >= int(debounce_steps):
#             wait_for_user(gui)
#             return True, float(t), "decel_to_stop", qe, list(pos), tuple(rpy)

#     # # hold at qe and observe/rest
#     # total_hold = float(observe_time) + float(rest_time)
#     # topple, t_hold, end_xyz, end_rpy = hold_pose(
#     #     robot_id, joint_indices, qe, stepsize, total_hold, hold_force,
#     #     roll_pitch_thresh_deg, debounce_steps, gui=gui
#     # )
#     # if topple:
#     #     # time measured relative to hold phase start; convert to global approx
#     #     t_topple = float(T_stop) + float(t_hold)
#     #     return True, t_topple, "post_stop_hold", qe, end_xyz, end_rpy

#     # passive observe at qe (no motor hold)
#     # 先把关节 reset 到 qe，速度置 0（类似你自碰撞脚本）
#     # for jid, angle in zip(joint_indices, qe):
#     #     p.resetJointState(robot_id, jid, float(angle), targetVelocity=0.0)
#     # p.stepSimulation()

#     # topple, t_hold, end_xyz, end_rpy = passive_warmup_and_check_topple(
#     #     robot_id=robot_id,
#     #     stepsize=stepsize,
#     #     warm_steps=20,
#     #     check_steps=int((observe_time + rest_time) / stepsize),
#     #     roll_pitch_thresh_deg=roll_pitch_thresh_deg,
#     #     debounce_steps=debounce_steps,
#     #     gui=gui,
#     # )
#     # if topple:
#     #     t_topple = float(T_stop) + float(t_hold)
#     #     wait_for_user(gui)
#     #     return True, t_topple, "post_stop_passive", qe, end_xyz, end_rpy


#     return False, -1.0, "none", qe, end_xyz, end_rpy


# def main():
#     args = parse_args()

#     # Import Panda class
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
#     stepsize = panda_robot.stepsize

#     if args.gui:
#         p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
#         p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
#         p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.setGravity(0, 0, -9.81)

#     # friction
#     if hasattr(panda_robot, "plane"):
#         p.changeDynamics(panda_robot.plane, -1,
#                          lateralFriction=float(args.lateral_friction),
#                          restitution=0.0)
#     p.changeDynamics(robot, -1,
#                      lateralFriction=float(args.lateral_friction),
#                      restitution=0.0,
#                      linearDamping=0.0,
#                      angularDamping=0.0)

#     # If you had to disable a specific self-collision pair, keep it (optional)
#     # NOTE: link indices here depend on your URDF; keep what you used before if needed.
#     p.setCollisionFilterPair(robot, robot, 8, 6, enableCollision=0)

#     # joints (7 arm joints as defined by Panda_toppling.Panda)
#     joint_indices = list(panda_robot.joints)
#     if not joint_indices or len(joint_indices) != 7:
#         raise RuntimeError(f"Expected 7 arm joints, got {len(joint_indices)}: {joint_indices}")

#     joint_position_limits = []
#     joint_velocity_limits = []
#     for jid in joint_indices:
#         info = p.getJointInfo(robot, jid)
#         joint_position_limits.append((info[8], info[9]))
#         joint_velocity_limits.append((-info[11], info[11]))

#     # acceleration limits (same as your previous code)
#     joint_acceleration_limits = [(-15, 15), (-7.5, 7.5), (-10, 10),
#                                  (-12.5, 12.5), (-15, 15), (-20, 20), (-20, 20)]
#     if len(joint_acceleration_limits) != len(joint_indices):
#         raise RuntimeError("joint_acceleration_limits length must match joint count")

#     n_samples = int(args.n_samples)
#     prog_every = max(1, n_samples // 10)
#     N_topple = 0

#     # Stream-write CSV (no huge RAM usage)
#     out_file = args.out_csv
#     with open(out_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         header = (
#             [f"joint_{i}_pos" for i in range(len(joint_indices))] +
#             [f"joint_{i}_vel" for i in range(len(joint_indices))] +
#             [f"joint_{i}_stop_pos" for i in range(len(joint_indices))] +
#             ["topple", "t_topple", "phase"] +
#             ["base_end_x", "base_end_y", "base_end_z"] +
#             ["base_end_roll", "base_end_pitch", "base_end_yaw"]
#         )
#         writer.writerow(header)

#         for idx in range(n_samples):
#             q0, qd0 = sample_configuration(joint_position_limits, joint_velocity_limits, args)

#             topple, t_topple, phase, qe, end_xyz, end_rpy = rollout_stop_topple(
#                 robot_id=robot,
#                 joint_indices=joint_indices,
#                 q0=q0,
#                 qd0=qd0,
#                 acc_limits=joint_acceleration_limits,
#                 stepsize=stepsize,
#                 base_z0=args.base_z0,
#                 settle_time=args.settle_time,
#                 move_force=args.move_force,
#                 hold_force=args.hold_force,
#                 observe_time=args.observe_time,
#                 rest_time=args.rest_time,
#                 roll_pitch_thresh_deg=args.roll_pitch_thresh_deg,
#                 debounce_steps=args.debounce_steps,
#                 gui=args.gui
#             )

#             if qe is None:
#                 # If toppled in settle phase, qe not meaningful; still output something consistent
#                 qe = [np.nan] * len(joint_indices)

#             if topple:
#                 N_topple += 1
#                 if args.gui:
#                     print(f"[TOPPLE] sample {idx} t={t_topple:.4f}s phase={phase} end_rpy={end_rpy}")

#             writer.writerow(
#                 list(map(float, q0)) +
#                 list(map(float, qd0)) +
#                 list(map(float, qe)) +
#                 [int(topple), float(t_topple), str(phase)] +
#                 list(map(float, end_xyz)) +
#                 [float(end_rpy[0]), float(end_rpy[1]), float(end_rpy[2])]
#             )

#             if (idx + 1) % prog_every == 0:
#                 print(f"Sample {idx+1}/{n_samples}: topples so far = {N_topple}")

#     print(f"Done. Results saved to {out_file}")
#     p.disconnect()


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import numpy as np
import csv
import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample joint configurations and detect self-collisions (optionally also detect toppling) on a floating-base robot"
    )

    # sampling
    parser.add_argument("--limit_sampling", action="store_true",
                        help="Enable near-limit sampling for a subset of joints")
    parser.add_argument("--limit_fraction", type=float, default=0.05,
                        help="Fraction of joint range to sample near limits")
    parser.add_argument("--limit_joints", type=int, default=6,
                        help="Number of joints to sample near their limits")
    parser.add_argument("--n_samples", type=int, default=800000,
                        help="Total number of samples to generate")

    # sim / urdf
    parser.add_argument("--urdf_path", type=str,
                        default="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7.urdf",
                        help="Path to the URDF file")
    parser.add_argument("--gui", action="store_true",
                        help="Enable GUI visualization")

    # biasing for collisions
    parser.add_argument("--collision_bias", action="store_true",
                        help="Bias sampling towards self-collisions by forcing more joints to limits")

    # --- toppling rollout options ---
    # NOTE: if rollout_steps > 0, each sample will simulate for rollout_steps steps (very expensive for large n_samples).
    parser.add_argument("--rollout_steps", type=int, default=5000,
                        help="If >0, run a short rollout after resetting each pose and detect toppling. "
                             "Example: 250~1000. Set 0 to disable (default).")
    parser.add_argument("--topple_thresh", type=float, default=0.8,
                        help="Topple threshold in radians for |roll| or |pitch|. Default 0.8 (~45 deg).")
    parser.add_argument("--base_z", type=float, default=0.08,
                        help="Initial base height (z) used when resetting the floating base.")
    parser.add_argument("--hold_pose", action="store_true",
                        help="If set, use POSITION_CONTROL during rollout to hold the joint pose (recommended for meaningful COM/topple checks).")
    parser.add_argument("--hold_force", type=float, default=200.0,
                        help="Max force/torque per joint used for holding pose during rollout (only if --hold_pose).")
    parser.add_argument("--pos_gain", type=float, default=0.3,
                        help="Position gain for holding pose (only if --hold_pose).")
    parser.add_argument("--vel_gain", type=float, default=1.0,
                        help="Velocity gain for holding pose (only if --hold_pose).")

    return parser.parse_args()


def sample_configuration(joint_position_limits, joint_velocity_limits, args):
    """
    If limit_sampling is not enabled, perform uniform random sampling across all joints;
    otherwise, perform a global random sample, then randomly select args.limit_joints joints
    and sample near their lower or upper limits by args.limit_fraction.
    """
    # Base random sampling for all joints
    q = [np.random.uniform(low, high) for (low, high) in joint_position_limits]
    qd = [np.random.uniform(low, high) for (low, high) in joint_velocity_limits]

    use_limit = args.limit_sampling
    num_limits = args.limit_joints

    if args.collision_bias:
        # In collision bias mode, we want to force more joints to limits more often
        # to increase chance of self-collision.
        if np.random.rand() < 0.5:
            use_limit = True
            num_limits = np.random.randint(4, len(joint_position_limits) + 1)
        else:
            use_limit = False

    if not use_limit:
        return q, qd

    # Resample near limits for a subset of joints
    num = min(num_limits, len(joint_position_limits))
    idxs = np.random.choice(len(joint_position_limits), num, replace=False)
    for j in idxs:
        low, high = joint_position_limits[j]
        span = high - low
        frac = args.limit_fraction
        if np.random.rand() < 0.4:
            q[j] = np.random.uniform(low, low + frac * span)
        else:
            q[j] = np.random.uniform(high - frac * span, high)

    return q, qd


def clip_to_limits(q, joint_position_limits):
    q2 = list(q)
    for i, (low, high) in enumerate(joint_position_limits):
        if np.isfinite(low) and np.isfinite(high):
            q2[i] = float(np.clip(q2[i], low, high))
        else:
            q2[i] = float(q2[i])
    return q2


def reset_floating_base(robot, base_z):
    # Reset base pose and clear velocities to avoid cross-sample contamination
    p.resetBasePositionAndOrientation(robot, [0.0, 0.0, float(base_z)], p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
    p.resetBaseVelocity(robot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def set_joints_state(robot, joint_indices, q, qd=None):
    if qd is None:
        for jid, angle in zip(joint_indices, q):
            p.resetJointState(robot, jid, targetValue=float(angle), targetVelocity=0.0)
    else:
        for jid, angle, vel in zip(joint_indices, q, qd):
            p.resetJointState(robot, jid, targetValue=float(angle), targetVelocity=float(vel))


def rollout_and_check_topple(robot, joint_indices, q_target, steps, thresh,
                             hold_pose=False, hold_force=200.0, pos_gain=0.3, vel_gain=1.0,
                             gui_sleep_s=0.0):
    """
    Run simulation for `steps` and return:
      toppled (bool), final (roll,pitch,yaw), min_z (base z over rollout)
    If hold_pose=True, apply POSITION_CONTROL each step to hold q_target.
    """
    toppled = False
    roll = pitch = yaw = 0.0
    min_z = 1e9

    if hold_pose:
        # Set the motor controller once; targets can be re-sent each step for robustness.
        pass

    for _ in range(int(steps)):
        if hold_pose:
            p.setJointMotorControlArray(
                bodyUniqueId=robot,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=q_target,
                positionGains=[pos_gain] * len(joint_indices),
                velocityGains=[vel_gain] * len(joint_indices),
                forces=[hold_force] * len(joint_indices),
            )

        p.stepSimulation()

        pos, orn = p.getBasePositionAndOrientation(robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        min_z = min(min_z, pos[2])

        if abs(roll) > thresh or abs(pitch) > thresh:
            toppled = True
            print("Toppled during rollout: roll={:.3f}, pitch={:.3f}".format(roll, pitch))
            break

        if gui_sleep_s > 0.0:
            time.sleep(gui_sleep_s)

    return toppled, (roll, pitch, yaw), min_z


def main():
    args = parse_args()

    # Import Panda class
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Panda_toppling import Panda

    # Initialize robot (handles PyBullet connection and URDF loading)
    connection_mode = p.GUI if args.gui else p.DIRECT
    panda_robot = Panda(
        stepsize=2e-3,
        realtime=0,
        connection_mode=connection_mode,
        urdf_path=args.urdf_path,
        use_fixed_base=False
    )
    robot = panda_robot.robot

    if args.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Optional: ignore a specific self-collision pair (your original line)
    p.setCollisionFilterPair(robot, robot, 8, 6, enableCollision=0)

    # Sample only the 7 arm joints defined by your Panda_toppling wrapper
    joint_indices = list(panda_robot.joints)
    if not joint_indices:
        raise RuntimeError(f"No movable joints found in URDF at {args.urdf_path}")

    # Build joint limits
    joint_position_limits = []
    joint_velocity_limits = []
    joint_torque_limits = []

    # (min,max) acceleration limits for the 7 arm joints
    joint_acceleration_limits = [(-15, 15), (-7.5, 7.5), (-10, 10), (-12.5, 12.5),
                                 (-15, 15), (-20, 20), (-20, 20)]
    if len(joint_acceleration_limits) != len(joint_indices):
        raise RuntimeError(
            f"joint_acceleration_limits has {len(joint_acceleration_limits)} entries, "
            f"but joint_indices has {len(joint_indices)} joints."
        )

    for jid in joint_indices:
        info = p.getJointInfo(robot, jid)
        joint_position_limits.append((info[8], info[9]))
        joint_velocity_limits.append((-info[11], info[11]))
        joint_torque_limits.append((-info[10], info[10]))

    # Output buffers
    n_samples = args.n_samples
    pos_samples = []
    vel_samples = []
    end_pos = []
    collision_flags = []
    topple_flags = []   # NEW: topple label (only meaningful if rollout_steps>0)
    N_collision = 0
    N_topple = 0

    progress_every = max(1, n_samples // 10)

    # For GUI: slow down a little if desired (keep small; rollout already costs time)
    gui_sleep = 0.0
    if args.gui:
        gui_sleep = 0.0  # set to 0.001 if you want to visually see motion

    for idx in range(n_samples):
        q, qd = sample_configuration(joint_position_limits, joint_velocity_limits, args)

        # Compute qe ("stopped" pose under bounded deceleration)
        qe = []
        for j, vel in enumerate(qd):
            a_max = joint_acceleration_limits[j][1]
            if vel == 0:
                qe_j = q[j]
            else:
                t_stop = abs(vel) / a_max
                delta = 0.5 * vel * t_stop
                qe_j = q[j] + delta
            qe.append(qe_j)

        # Optional: clip to joint limits to avoid absurd resets
        q_clipped = clip_to_limits(q, joint_position_limits)
        qe_clipped = clip_to_limits(qe, joint_position_limits)

        pos_samples.append(q_clipped)
        vel_samples.append(qd)
        end_pos.append(qe_clipped)

        # ---------------------------
        # 1) Test at pose q
        # ---------------------------
        reset_floating_base(robot, args.base_z)
        set_joints_state(robot, joint_indices, q_clipped)

        toppled_q = False
        if args.rollout_steps > 0:
            toppled_q, _, _ = rollout_and_check_topple(
                robot=robot,
                joint_indices=joint_indices,
                q_target=q_clipped,
                steps=args.rollout_steps,
                thresh=args.topple_thresh,
                hold_pose=args.hold_pose,
                hold_force=args.hold_force,
                pos_gain=args.pos_gain,
                vel_gain=args.vel_gain,
                gui_sleep_s=gui_sleep,
            )

        # Self-collision query (robot vs robot)
        contacts_q = p.getContactPoints(bodyA=robot, bodyB=robot)
        collision_q = len(contacts_q) > 0

        # ---------------------------
        # 2) Test at pose qe
        # ---------------------------
        reset_floating_base(robot, args.base_z)
        set_joints_state(robot, joint_indices, qe_clipped)

        toppled_qe = False
        if args.rollout_steps > 0:
            toppled_qe, _, _ = rollout_and_check_topple(
                robot=robot,
                joint_indices=joint_indices,
                q_target=qe_clipped,
                steps=args.rollout_steps,
                thresh=args.topple_thresh,
                hold_pose=args.hold_pose,
                hold_force=args.hold_force,
                pos_gain=args.pos_gain,
                vel_gain=args.vel_gain,
                gui_sleep_s=gui_sleep,
            )

        contacts_qe = p.getContactPoints(bodyA=robot, bodyB=robot)
        collision_qe = len(contacts_qe) > 0

        # Labels
        collision = collision_q or collision_qe
        toppled = toppled_q or toppled_qe

        # if collision:
        #     N_collision += 1
        #     if args.gui:
        #         print(f"[{idx}] Self-collision detected.")
        #         all_contacts = contacts_q + contacts_qe
        #         for c in all_contacts:
        #             p.addUserDebugLine(
        #                 c[5], [c[5][0], c[5][1], c[5][2] + 0.2],
        #                 [1, 0, 0], lineWidth=5, lifeTime=0.5
        #             )
        #         time.sleep(0.1)

        if toppled:
            N_topple += 1
            if args.gui:
                print(f"[{idx}] Topple detected (|roll| or |pitch| > {args.topple_thresh:.2f} rad).")
                time.sleep(0.1)

        collision_flags.append(bool(collision))
        topple_flags.append(bool(toppled))

        # Progress
        if (idx + 1) % progress_every == 0:
            if args.rollout_steps > 0:
                print(f"Sample {idx+1}/{n_samples}: collisions={N_collision}, topple={N_topple}")
            else:
                print(f"Sample {idx+1}/{n_samples}: collisions={N_collision}")

    # Write results to CSV
    if args.limit_sampling:
        out_file = f"collision_results_{args.limit_joints}_limit_sampling_new.csv"
    else:
        out_file = "collision_results_test.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            [f"joint_{i}_pos" for i in range(len(joint_indices))] +
            [f"joint_{i}_vel" for i in range(len(joint_indices))] +
            [f"joint_{i}_final_pos" for i in range(len(joint_indices))] +
            ["collision"] +
            ["toppled"]  # NEW
        )
        writer.writerow(header)

        for q_row, qd_row, qe_row, col_flag, top_flag in zip(
            pos_samples, vel_samples, end_pos, collision_flags, topple_flags
        ):
            writer.writerow(list(q_row) + list(qd_row) + list(qe_row) + [int(col_flag)] + [int(top_flag)])

    print(f"Done. Results saved to {out_file}")
    p.disconnect()


if __name__ == "__main__":
    main()
