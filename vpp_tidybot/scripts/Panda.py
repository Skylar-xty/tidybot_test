# Panda.py
import sys
sys.path.append('./src')

import numpy as np
import pybullet as p

class Panda:
    def __init__(self, stepsize=1e-3, realtime=0, connection_mode=p.GUI, urdf_path=None, use_fixed_base=True):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque"
        self.position_control_gain_p = [0.1]*7
        self.position_control_gain_d = [1.0]*7
        self.max_torque = [10000]*7

        self.cam_base_yaw = 30
        self.cam_pitch     = -20
        self.cam_dist      = 1.0
        self.cam_target    = [0, 0, 0.5]

        # connect pybullet
        if connection_mode == p.GUI:
            p.connect(connection_mode, options="--width=2048 --height=1536")
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=30,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.5],
            )
        else:
            p.connect(connection_mode)

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0, 0, 0)

        self.plane = p.loadURDF("../models/urdf/plane/plane.urdf", useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=.95)
        import os

        print("cwd =", os.getcwd())
        if urdf_path is None:
            # urdf_path = os.path.join("/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_fix.urdf")
            urdf_path = os.path.join("G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/base_fix.urdf")
        print("urdf_path =", urdf_path, "exists:", os.path.exists(urdf_path))

        # Temporarily switch to the model root directory so PyBullet can find meshes
        # referenced via package://iiwa_description
        # repo_root = "/Users/stav.42/f_lab/tidybot_model/"
        # repo_root = "/Users/stav.42/f_lab/vpp_tidybot_test/tidybot_model"
        repo_root = "G:/My Drive/PENN/vpp_tidybot_test/tidybot_model"
        old_cwd = os.getcwd()
        if os.path.exists(repo_root):
            os.chdir(repo_root)
        try:
            # NOTE: assuming this URDF has the same structure as the XML you pasted
            self.robot = p.loadURDF(
                urdf_path,
                useFixedBase=use_fixed_base,
                # flags=p.URDF_USE_SELF_COLLISION,
                flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            )
        finally:
            if os.path.exists(repo_root):
                os.chdir(old_cwd)
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)

        # figure out joints
        num_joints = p.getNumJoints(self.robot)
        # For your URDF: 0=floating_base, 1=arm_mount_joint, 2..8 = iiwa_joint_1..7
        self.base_fixed_joints = 2          # 0,1
        self.arm_dof           = 7          # iiwa has 7 revolute joints
        self.arm_start_idx     = self.base_fixed_joints          # 2
        self.arm_joint_indices = list(range(self.arm_start_idx,
                                            self.arm_start_idx + self.arm_dof))  # [2..8]
        self.dof               = self.arm_dof

        # end-effector link index (child of iiwa_joint_7 -> index 8)
        self.ee_link_index = self.arm_start_idx + self.arm_dof    # 2 + 7 = 9 ❌
        # careful: joints are 0..8, links are 0..8. Last joint index is 8, last link index is also 8.
        # So better:
        self.ee_link_index = self.arm_start_idx + self.arm_dof - 1  # 2 + 7 - 1 = 8

        # book-keeping
        self.joints        = []
        self.q_min         = []
        self.q_max         = []
        self.target_pos    = []
        self.target_torque = []

        # collect only the arm joints (ignore base-fixed joints)
        for j in self.arm_joint_indices:
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            # init at mid-range
            self.target_pos.append((self.q_min[-1] + self.q_max[-1]) / 2.0)
            self.target_torque.append(0.0)

        self.reset()

    def reset(self):
        self.t = 0.0
        self.control_mode = "torque"

        # these 7 values must correspond to iiwa_joint_1..7 (indices 2..8)
        self.target_pos = [0.669, -0.346, -0.742, -1.66, -0.367, 2.3, 1.99]

        for j_idx, q_des in zip(self.joints, self.target_pos):
            self.target_torque[self.joints.index(j_idx)] = 0.0
            p.resetJointState(self.robot, j_idx, targetValue=q_des)

        self.resetController()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self):
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * self.dof,
        )

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "velocity":
            self.control_mode = "velocity"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise ValueError("wrong control mode")

    def setTargetPositions(self, target_pos):
        assert len(target_pos) == self.dof
        self.target_pos = target_pos
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_pos,
        )

    def setTargetVelocities(self, target_vel):
        assert len(target_vel) == self.dof
        self.target_vel = target_vel
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=self.target_vel,
        )

    def setTargetTorques(self, target_torque):
        assert len(target_torque) == self.dof
        self.target_torque = target_torque
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.TORQUE_CONTROL,
            forces=self.target_torque,
        )

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel

    def solveInverseDynamics(self, pos, vel, acc):
        # pos/vel/acc are 7-vectors in joint space of iiwa_joint_1..7
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        # Use EE link index
        return list(p.calculateInverseKinematics(self.robot, self.ee_link_index, pos, ori))

    def solveForwardKinematics(self):
        pos, ori = p.getLinkState(self.robot, self.ee_link_index)[0:2]
        return pos, ori

    def getEndVelocity(self):
        return p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=True)[6]

    def getEndAngularVelocity(self):
        return p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=True)[7]

    def applyForce(self, force):
        p.applyExternalForce(
            self.robot,
            self.ee_link_index,
            force,
            p.getLinkState(self.robot, self.ee_link_index)[0],
            p.WORLD_FRAME,
        )

    def getJacobian(self):
        joint_states = p.getJointStates(self.robot, self.joints)
        jointpos = [js[0] for js in joint_states]
        jointvel = [0.0] * self.dof
        jointacc = [0.0] * self.dof

        # PyBullet returns (J_lin, J_ang)
        Jlin, Jang = p.calculateJacobian(
            self.robot,
            self.ee_link_index,
            [0, 0, 0],
            jointpos,
            jointvel,
            jointacc,
        )
        return Jlin  # shape (3, 7)

    def getJacobian_ori(self):
        joint_states = p.getJointStates(self.robot, self.joints)
        jointpos = [js[0] for js in joint_states]
        jointvel = [0.0] * self.dof
        jointacc = [0.0] * self.dof

        Jlin, Jang = p.calculateJacobian(
            self.robot,
            self.ee_link_index,
            [0, 0, 0],
            jointpos,
            jointvel,
            jointacc,
        )
        return Jang  # shape (3, 7)


    def getMassMatrix(self, jointstates):
        # jointstates should be a 7-vector matching the arm joints
        return p.calculateMassMatrix(self.robot, jointstates)

    def getClosestPoints(self, link1, link2):
        # here link1, link2 should be *link indices* (0..8), not joint indices
        return p.getClosestPoints(self.robot, self.robot, 1000, link1, link2)
