#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import numpy as np
import csv
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample joint configurations and detect self-collisions on a Panda robot')
    parser.add_argument(
        '--limit_sampling', action='store_true',
        help='Enable near-limit sampling for a subset of joints')
    parser.add_argument(
        '--limit_fraction', type=float, default=0.05,
        help='Fraction of joint range to sample near limits')
    parser.add_argument(
        '--limit_joints', type=int, default=6,
        help='Number of joints to sample near their limits')
    parser.add_argument(
        '--n_samples', type=int, default=8000000,
        help='Total number of samples to generate')
    parser.add_argument(
        '--urdf_path', type=str,
        # default="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7toppling_with_stick.urdf",
        default="G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/base_fix.urdf",
        help='Path to the Panda URDF file')
    parser.add_argument(
        '--gui', action='store_true',
        help='Enable GUI visualization')
    parser.add_argument(
        '--collision_bias', action='store_true',
        help='Bias sampling towards self-collisions by forcing more joints to limits')
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
        if np.random.rand() < 0.5: # 70% chance to do aggressive limit sampling
            use_limit = True
            # Force 4 to all joints to limits
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


import time

def main():
    args = parse_args()

    # Import Panda class
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Panda_toppling import Panda

    # Initialize Panda robot (which handles PyBullet connection and URDF loading)
    connection_mode = p.GUI if args.gui else p.DIRECT
    panda_robot = Panda(stepsize=2e-3, realtime=0, connection_mode=connection_mode, urdf_path=args.urdf_path,use_fixed_base=False)
    robot = panda_robot.robot
    
    # # Print all joint/link info to identify the mobile base
    # print("="*50)
    # print(f"Robot Body ID: {robot}")
    # print("Link/Joint Info:")
    # # Base link is always -1
    # print(f"ID: -1, Link Name: Base Link (Root)")
    # for i in range(p.getNumJoints(robot)):
    #     info = p.getJointInfo(robot, i)
    #     # info[0] is jointIndex, info[1] is jointName, info[12] is linkName
    #     print(f"ID: {info[0]}, Joint Name: {info[1].decode('utf-8')}, Link Name: {info[12].decode('utf-8')}")
    # print("="*50)

    if args.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_CONTACT_POINTS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # p.setCollisionFilterPair(robot, robot, 4, 6, enableCollision=0)
    p.setCollisionFilterPair(robot, robot, 8, 6, enableCollision=0)


    # 获取关节索引和关节限制（只采样 Panda 类定义的 7 个手臂关节，匹配 14 维特征）
    joint_indices = list(panda_robot.joints)
    joint_position_limits = []
    joint_velocity_limits = []
    joint_torque_limits = []

    # (min,max) acceleration limits for the 7 arm joints
    joint_acceleration_limits = [(-15, 15), (-7.5, 7.5), (-10, 10), (-12.5, 12.5), (-15, 15), (-20, 20), (-20, 20)]
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
    # print("joint_position_limits", joint_position_limits)
    # print("joint_velocity_limits", joint_velocity_limits)
    # print("joint_acceleration_limits", joint_acceleration_limits)
    # print("joint_torque_limits", joint_torque_limits)
    if not joint_indices:
        raise RuntimeError(f'No movable joints found in URDF at {args.urdf_path}')

    # Generate samples and detect collisions
    n_samples = args.n_samples
    pos_samples = []
    vel_samples = []
    end_pos = []
    collision_flags = []
    N_collision = 0

    for idx in range(n_samples):
        q, qd = sample_configuration(joint_position_limits, joint_velocity_limits, args)
        pos_samples.append(q)
        vel_samples.append(qd)
        qe = []
        for j, vel in enumerate(qd):
            a_max = joint_acceleration_limits[j][1]
            if vel == 0:
                qe_j = q[j]
            else:
                t_stop = abs(vel) / a_max
                # distance under constant deceleration to zero
                delta = 0.5 * vel * t_stop
                qe_j = q[j] + delta
            qe.append(qe_j)
        end_pos.append(qe)

        # 1) Test at the sampled pose q
        for jid, angle in zip(joint_indices, q):
            p.resetJointState(robot, jid, angle)
        p.stepSimulation()
        if args.gui:
            time.sleep(0.01)
        contacts_q = p.getContactPoints(bodyA=robot, bodyB=robot)
        collision_q = len(contacts_q) > 0

        # 2) Test at the “stopped” pose qe
        for jid, angle in zip(joint_indices, qe):
            p.resetJointState(robot, jid, angle)
        p.stepSimulation()
        if args.gui:
            time.sleep(0.01)
        contacts_qe = p.getContactPoints(bodyA=robot, bodyB=robot)
        collision_qe = len(contacts_qe) > 0

        # 3) If either pose self‐collides, mark as collision
        collision = collision_q or collision_qe
        if collision:
            N_collision += 1
            if args.gui:
                print(f"Collision detected at sample {idx}")
                all_contacts = contacts_q + contacts_qe
                # Visualize contacts
                for c in all_contacts:
                    # c[5] is positionOnA, c[6] is positionOnB
                    p.addUserDebugLine(c[5], [c[5][0], c[5][1], c[5][2]+0.2], [1, 0, 0], lineWidth=5, lifeTime=1.0)
                    
                    # Highlight colliding links in red
                    linkA = c[3]
                    linkB = c[4]
                    if linkA >= -1: p.changeVisualShape(robot, linkA, rgbaColor=[1, 0, 0, 1])
                    if linkB >= -1: p.changeVisualShape(robot, linkB, rgbaColor=[1, 0, 0, 1])
                
                time.sleep(1.0)
                
                # Reset colors (simple reset to white/grey)
                for c in all_contacts:
                    linkA = c[3]
                    linkB = c[4]
                    if linkA >= -1: p.changeVisualShape(robot, linkA, rgbaColor=[1, 1, 1, 1])
                    if linkB >= -1: p.changeVisualShape(robot, linkB, rgbaColor=[1, 1, 1, 1])

        collision_flags.append(collision)

        # Periodically print progress
        if (idx + 1) % (n_samples // 10) == 0:
            print(f"Sample {idx+1}/{n_samples}: collisions so far = {N_collision}")

    # Write results to CSV
    if args.limit_sampling:
        out_file = f'collision_results_{args.limit_joints}_limit_sampling_new_8m.csv'
    else:
        out_file = 'collision_results_test.csv'
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"joint_{i}_pos" for i in range(len(joint_indices))] + [f"joint_{i}_vel" for i in range(len(joint_indices))]+ [f"joint_{i}_final_pos" for i in range(len(joint_indices))]+ ['collision']
        writer.writerow(header)
        for q, qd, qe, flag in zip(pos_samples, vel_samples, end_pos, collision_flags):
            writer.writerow(list(q) + list(qd) + list(qe) + [int(flag)])

    print(f"Done. Results saved to {out_file}")
    p.disconnect()


if __name__ == '__main__':
    main()
