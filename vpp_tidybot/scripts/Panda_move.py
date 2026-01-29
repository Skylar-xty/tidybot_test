# Panda.py
import sys
sys.path.append('./src')

import numpy as np
import pybullet as p
import os

class Panda:
    def __init__(self, stepsize=1e-3, realtime=0, connection_mode=p.GUI, urdf_path=None):
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
                cameraDistance=1.5,
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

        # Load Plane
        p.setAdditionalSearchPath(os.path.dirname(urdf_path) if urdf_path else "")
        self.plane = p.loadURDF("plane.urdf", useFixedBase=True) if os.path.exists("plane.urdf") else None
        if self.plane is None:
            # Fallback internal plane if file not found
            import pybullet_data
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.plane = p.loadURDF("plane.urdf")
             
        p.changeDynamics(self.plane, -1, restitution=.95)

        print("cwd =", os.getcwd())
        if urdf_path is None:
            # UPDATE THIS PATH TO MATCH YOUR SYSTEM
            urdf_path = os.path.join("G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/models/urdf/tidybot/base_move.urdf")
            # urdf_path = os.path.join("/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_move.urdf")
        
        print("urdf_path =", urdf_path, "exists:", os.path.exists(urdf_path))

        # 计算 repo 根目录
        repo_root = "G:/My Drive/PENN/vpp_tidybot_test/tidybot_model"
        if os.path.exists(repo_root):
            os.chdir(repo_root)
        self.robot = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)
        p.setCollisionFilterPair(self.robot, self.plane, 1, -1, enableCollision=0)
        # --- JOINT LOGIC ---
        self.num_joints = p.getNumJoints(self.robot)
        
        self.base_dof = 2  # X, Y
        self.arm_dof = 7   # J1..J7
        self.dof = self.base_dof + self.arm_dof # 9 Total

        self.joints = []
        self.q_min = []
        self.q_max = []

        print("---------------- Joint Info ----------------")
        for j in range(self.num_joints):
            info = p.getJointInfo(self.robot, j)
            # info[1]: name, info[2]: type, info[8]: lower, info[9]: upper
            print(f"ID: {j}, Name: {info[1].decode('utf-8')}, Type: {info[2]}")
            
            # Skip Fixed Joints (Type 4)
            if info[2] != p.JOINT_FIXED:
                self.joints.append(j)
                self.q_min.append(info[8])
                self.q_max.append(info[9])

        self.ee_link_index = self.num_joints - 1
        
        self.target_pos = [0.0] * self.dof
        self.target_torque = [0.0] * self.dof

        self.reset()

    def reset(self):
        self.t = 0.0
        self.control_mode = "torque"

        # --- FIXED: Only 2 Base Coords (X, Y) ---
        base_target = [0.0, 0.0] 
        arm_target = [0.669, -0.346, -0.742, -1.66, -0.367, 2.3, 1.99]
        
        self.target_pos = base_target + arm_target
        
        # Safety check to prevent crashing
        if len(self.target_pos) != len(self.joints):
             print(f"ERROR: Target pos length ({len(self.target_pos)}) != Active Joints ({len(self.joints)})")

        for i, j_idx in enumerate(self.joints):
            self.target_torque[i] = 0.0
            p.resetJointState(self.robot, j_idx, targetValue=self.target_pos[i])

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

    def setTargetTorques(self, target_torque):
        # Ensure we pass a list to PyBullet (Mac fix)
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.TORQUE_CONTROL,
            forces=list(target_torque),
        )

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return np.array(joint_pos), np.array(joint_vel)

    def solveInverseDynamics(self, pos, vel, acc):
        # Pass lists to PyBullet
        torques = p.calculateInverseDynamics(self.robot, list(pos), list(vel), list(acc))
        return list(torques)

    def solveForwardKinematics(self):
        pos, ori = p.getLinkState(self.robot, self.ee_link_index)[0:2]
        return pos, ori

    def getEndVelocity(self):
        return p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=True)[6]

    def getJacobian(self):
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
        return np.array(Jlin)

    def getMassMatrix(self, jointstates):
        return np.array(p.calculateMassMatrix(self.robot, list(jointstates)))
    
    def getClosestPoints(self, link1, link2):
        # here link1, link2 should be *link indices* (0..8), not joint indices
        return p.getClosestPoints(self.robot, self.robot, 1000, link1, link2)
