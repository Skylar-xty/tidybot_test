# # Panda.py
# import sys
# sys.path.append('./src')

# import numpy as np
# import pybullet as p
# import os

# class Panda:
#     def __init__(self, stepsize=1e-3, realtime=0, connection_mode=p.GUI, urdf_path=None, use_fixed_base=True):
#         self.t = 0.0
#         self.stepsize = stepsize
#         self.realtime = realtime

#         self.control_mode = "torque"
#         self.position_control_gain_p = [0.1]*7
#         self.position_control_gain_d = [1.0]*7
#         self.max_torque = [10000]*7

#         self.cam_base_yaw = 30
#         self.cam_pitch     = -20
#         self.cam_dist      = 1.0
#         self.cam_target    = [0, 0, 0.5]

#         # connect pybullet
#         if connection_mode == p.GUI:
#             p.connect(connection_mode, options="--width=2048 --height=1536")
#             p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#             p.resetDebugVisualizerCamera(
#                 cameraDistance=1.5,
#                 cameraYaw=30,
#                 cameraPitch=-20,
#                 cameraTargetPosition=[0, 0, 0.5],
#             )
#         else:
#             p.connect(connection_mode)

#         p.resetSimulation()
#         p.setTimeStep(self.stepsize)
#         p.setRealTimeSimulation(self.realtime)
#         p.setGravity(0, 0, -9.81)

#         # Load Plane
#         p.setAdditionalSearchPath(os.path.dirname(urdf_path) if urdf_path else "")
#         self.plane = p.loadURDF("plane.urdf", useFixedBase=True) if os.path.exists("plane.urdf") else None
#         if self.plane is None:
#             # Fallback internal plane if file not found
#             import pybullet_data
#             p.setAdditionalSearchPath(pybullet_data.getDataPath())
#             self.plane = p.loadURDF("plane.urdf")
             
#         p.changeDynamics(self.plane, -1, restitution=.95)

#         print("cwd =", os.getcwd())
#         if urdf_path is None:
#             # UPDATE THIS PATH TO MATCH YOUR SYSTEM
#             urdf_path = os.path.join("G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7move.urdf")
#             # urdf_path = os.path.join("/Users/stav.42/f_lab/tidybot_model/model/stanford_tidybot/base_move.urdf")
        
#         print("urdf_path =", urdf_path, "exists:", os.path.exists(urdf_path))

#         # 计算 repo 根目录
#         repo_root = "G:/My Drive/PENN/vpp_tidybot_test/tidybot_model"
#         if os.path.exists(repo_root):
#             os.chdir(repo_root)
#         self.robot = p.loadURDF(
#             urdf_path,
#             useFixedBase=use_fixed_base,
#             # useFixedBase=True,
#             flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
#         )
#         p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)
#         p.setCollisionFilterPair(self.robot, self.plane, 1, -1, enableCollision=0)
#         # --- JOINT LOGIC ---
#         self.num_joints = p.getNumJoints(self.robot)

#         # Only treat the 7 iiwa arm joints as actuated DoFs. The base
#         # prismatic joints remain passive (can move under external forces).
#         self.joints = []  # indices of iiwa_joint_1..7
#         self.q_min = []
#         self.q_max = []

#         print("---------------- Joint Info ----------------")
#         for j in range(self.num_joints):
#             info = p.getJointInfo(self.robot, j)
#             # info[1]: name, info[2]: type, info[8]: lower, info[9]: upper
#             print(f"ID: {j}, Name: {info[1].decode('utf-8')}, Type: {info[2]}")

#             name = info[1]
#             joint_type = info[2]

#             # Select only revolute iiwa joints as actuated (exclude base sliders)
#             if joint_type != p.JOINT_FIXED and name.startswith(b"iiwa_joint"):
#                 self.joints.append(j)
#                 self.q_min.append(info[8])
#                 self.q_max.append(info[9])

#         # All active joints are arm DoFs
#         self.arm_dof = len(self.joints)
#         print("Active arm joint indices:", self.joints)
#         self.dof = self.arm_dof

#         self.ee_link_index = self.num_joints - 1

#         self.target_pos = [0.0] * self.dof
#         self.target_torque = [0.0] * self.dof

#         self.reset()

#     def reset(self):
#         self.t = 0.0
#         self.control_mode = "torque"

#         # Desired angles for the 7 iiwa arm joints
#         arm_target = [0.669, -0.346, -0.742, -1.66, -0.367, 2.3, 1.99]

#         self.target_pos = arm_target
        
#         # Safety check to prevent crashing
#         if len(self.target_pos) != len(self.joints):
#             print(f"ERROR: Target pos length ({len(self.target_pos)}) != Active Joints ({len(self.joints)})")

#         for i, j_idx in enumerate(self.joints):
#             self.target_torque[i] = 0.0
#             p.resetJointState(self.robot, j_idx, targetValue=self.target_pos[i])

#         self.resetController()

#     def step(self):
#         self.t += self.stepsize
#         p.stepSimulation()

#     def resetController(self):
#         p.setJointMotorControlArray(
#             bodyUniqueId=self.robot,
#             jointIndices=self.joints,
#             controlMode=p.VELOCITY_CONTROL,
#             forces=[0.0] * self.dof,
#         )

#     def setControlMode(self, mode):
#         if mode == "position":
#             self.control_mode = "position"
#         elif mode == "velocity":
#             self.control_mode = "velocity"
#         elif mode == "torque":
#             if self.control_mode != "torque":
#                 self.resetController()
#             self.control_mode = "torque"
#         else:
#             raise ValueError("wrong control mode")

#     def setTargetTorques(self, target_torque):
#         # Ensure we pass a list to PyBullet (Mac fix)
#         p.setJointMotorControlArray(
#             bodyUniqueId=self.robot,
#             jointIndices=self.joints,
#             controlMode=p.TORQUE_CONTROL,
#             forces=list(target_torque),
#         )

#     def getJointStates(self):
#         joint_states = p.getJointStates(self.robot, self.joints)
#         joint_pos = [x[0] for x in joint_states]
#         joint_vel = [x[1] for x in joint_states]
#         return np.array(joint_pos), np.array(joint_vel)

#     def solveInverseDynamics(self, pos, vel, acc):
#         """Compute inverse-dynamics torques for the 7 arm joints only.

#         PyBullet's calculateInverseDynamics expects full joint vectors for all
#         joints in the robot. Here we:
#         - read the full joint positions/velocities from Bullet
#         - build a full acceleration vector: base joints get 0 accel, arm
#           joints use the provided acc
#         - call calculateInverseDynamics on the full vectors
#         - return only the torques corresponding to the arm joints (self.joints)
#         """

#         # Full joint state (including base sliders and fixed joints)
#         js_all = p.getJointStates(self.robot, list(range(self.num_joints)))
#         q_full = [s[0] for s in js_all]
#         qd_full = [s[1] for s in js_all]

#         # Zero accelerations for all joints initially
#         qdd_full = [0.0] * self.num_joints

#         # Map provided arm accelerations into the corresponding joint indices
#         for local_idx, joint_idx in enumerate(self.joints):
#             if local_idx < len(acc):
#                 qdd_full[joint_idx] = float(acc[local_idx])

#         try:
#             tau_full = p.calculateInverseDynamics(self.robot, q_full, qd_full, qdd_full,flags=1)
#         except Exception:
#             # If inverse dynamics fails for any reason, fall back to zeros so
#             # the outer PD loop can still run.
#             return [0.0] * len(self.joints)

#         # Extract torques for the 7 arm joints
#         tau_arm = [tau_full[j_idx] for j_idx in self.joints]
#         return tau_arm

#     def solveForwardKinematics(self):
#         pos, ori = p.getLinkState(self.robot, self.ee_link_index)[0:2]
#         return pos, ori

#     def getEndVelocity(self):
#         return p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=True)[6]

#     def getJacobian(self):
#         joint_states = p.getJointStates(self.robot, self.joints)
#         jointpos = [js[0] for js in joint_states]
#         jointvel = [0.0] * self.dof
#         jointacc = [0.0] * self.dof

#         Jlin, Jang = p.calculateJacobian(
#             self.robot,
#             self.ee_link_index,
#             [0, 0, 0],
#             jointpos,
#             jointvel,
#             jointacc,
#         )
#         return np.array(Jlin)

#     def getMassMatrix(self, jointstates):
#         return np.array(p.calculateMassMatrix(self.robot, list(jointstates)))
    
#     def getClosestPoints(self, link1, link2):
#         # here link1, link2 should be *link indices* (0..8), not joint indices
#         return p.getClosestPoints(self.robot, self.robot, 1000, link1, link2)


# Panda_toppling.py  (UNIFIED 9-DoF interface: base(prismatic) + arm(revolute))
import sys
sys.path.append('./src')

import numpy as np
import pybullet as p
import os


class Panda:
    """
    Unified interface version:
      - self.joints includes ALL non-fixed joints (e.g., base sliders + iiwa arm)
      - getJointStates / setTargetTorques / solveInverseDynamics all use the same DoF dimension
      - you can keep the base "passive" by commanding 0 torque on base joints
    """

    def __init__(
        self,
        stepsize=1e-3,
        realtime=0,
        connection_mode=p.GUI,
        urdf_path=None,
        use_fixed_base=False,   # keep your original signature; you can set True/False from caller
    ):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque"

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

        # NOTE: your existing code uses zero gravity; keep it consistent.
        # If you want real toppling under gravity, change to p.setGravity(0,0,-9.81).
        p.setGravity(0, 0, 0)

        # Load Plane
        p.setAdditionalSearchPath(os.path.dirname(urdf_path) if urdf_path else "")
        self.plane = p.loadURDF("plane.urdf", useFixedBase=True) if os.path.exists("plane.urdf") else None
        if self.plane is None:
            import pybullet_data
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, restitution=.95)

        print("cwd =", os.getcwd())
        if urdf_path is None:
            urdf_path = os.path.join(
                "G:/My Drive/PENN/vpp_tidybot_test/vpp_tidybot/tidybot_iiwa7_urdf-master/tidybot_iiwa7move.urdf"
            )

        print("urdf_path =", urdf_path, "exists:", os.path.exists(urdf_path))

        # Optional: keep your repo_root behavior (meshes often rely on relative paths)
        repo_root = "G:/My Drive/PENN/vpp_tidybot_test/tidybot_model"
        if os.path.exists(repo_root):
            os.chdir(repo_root)

        self.robot = p.loadURDF(
            urdf_path,
            useFixedBase=use_fixed_base,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )

        # remove damping so the base sliders can move freely if unactuated
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0)

        # If your plane collision should ignore some link, keep your original choice
        # (You used: p.setCollisionFilterPair(self.robot, self.plane, 1, -1, enableCollision=0))
        # but note: link index depends on URDF ordering.
        try:
            p.setCollisionFilterPair(self.robot, self.plane, 1, -1, enableCollision=0)
        except Exception:
            pass

        # -------------------------
        # Joint discovery (UNIFIED)
        # -------------------------
        self.num_joints = p.getNumJoints(self.robot)

        self.joints = []          # all non-fixed joint indices in ascending joint index order
        self.joint_names = []     # same length as self.joints
        self.q_min = []
        self.q_max = []

        self.base_joints = []     # subset of joint indices for base prismatic joints
        self.arm_joints = []      # subset of joint indices for iiwa revolute joints

        # positions (indices) in the unified vector
        self.base_pos_idx = []    # positions in [0..dof-1] corresponding to base joints
        self.arm_pos_idx = []     # positions in [0..dof-1] corresponding to arm joints

        print("---------------- Joint Info ----------------")
        for j in range(self.num_joints):
            info = p.getJointInfo(self.robot, j)
            name = info[1].decode("utf-8")
            jtype = info[2]
            print(f"ID: {j}, Name: {name}, Type: {jtype}")

            if jtype == p.JOINT_FIXED:
                continue

            # add to unified list
            self.joints.append(j)
            self.joint_names.append(name)
            self.q_min.append(info[8])
            self.q_max.append(info[9])

        # build subsets + index maps (based on names)
        for k, j in enumerate(self.joints):
            name = self.joint_names[k]
            if name in ("world_to_base_x", "base_x_to_base_y"):
                self.base_joints.append(j)
                self.base_pos_idx.append(k)
            elif name.startswith("iiwa_joint"):
                self.arm_joints.append(j)
                self.arm_pos_idx.append(k)

        self.base_dof = len(self.base_joints)
        self.arm_dof = len(self.arm_joints)
        self.dof = len(self.joints)  # unified dof (expected 9 for your tidybot_iiwa7_move.urdf)

        # end-effector link index (keep your original convention)
        self.ee_link_index = self.num_joints - 1

        self.target_pos = [0.0] * self.dof
        self.target_torque = [0.0] * self.dof

        self.reset()

    def reset(self):
        self.t = 0.0
        self.control_mode = "torque"

        # default targets:
        # base: 0,0 ; arm: your existing 7-joint pose
        arm_default = {
            "iiwa_joint_1": 0.669,
            "iiwa_joint_2": -0.346,
            "iiwa_joint_3": -0.742,
            "iiwa_joint_4": -1.66,
            "iiwa_joint_5": -0.367,
            "iiwa_joint_6": 2.3,
            "iiwa_joint_7": 1.99,
        }
        base_default = {
            "world_to_base_x": 0.0,
            "base_x_to_base_y": 0.0,
        }

        # fill target_pos in unified order
        self.target_pos = [0.0] * self.dof
        for k, name in enumerate(self.joint_names):
            if name in base_default:
                self.target_pos[k] = base_default[name]
            elif name in arm_default:
                self.target_pos[k] = arm_default[name]
            else:
                self.target_pos[k] = 0.0

        # reset joint states
        for k, j_idx in enumerate(self.joints):
            self.target_torque[k] = 0.0
            p.resetJointState(self.robot, j_idx, targetValue=self.target_pos[k])

        self.resetController()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self):
        # Disable Bullet's default motors on ALL active joints so torque control is "clean".
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
        if len(target_torque) != self.dof:
            raise ValueError(f"target_torque length {len(target_torque)} != dof {self.dof}")
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
        return np.array(joint_pos, dtype=np.float32), np.array(joint_vel, dtype=np.float32)

    def solveInverseDynamics(self, pos, vel, acc):
        # Unified: pos/vel/acc must all be length dof
        if len(pos) != self.dof or len(vel) != self.dof or len(acc) != self.dof:
            raise ValueError(
                f"InverseDynamics expects (pos,vel,acc) length {self.dof}, "
                f"got ({len(pos)},{len(vel)},{len(acc)})"
            )
        torques = p.calculateInverseDynamics(self.robot, list(pos), list(vel), list(acc))
        return list(torques)

    def solveForwardKinematics(self):
        pos, ori = p.getLinkState(self.robot, self.ee_link_index)[0:2]
        return pos, ori

    def getEndVelocity(self):
        return p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=True)[6]

    def getJacobian(self):
        # Jacobian w.r.t. unified joint vector order
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
        return np.array(Jlin, dtype=np.float32)

    def getMassMatrix(self, jointstates):
        if len(jointstates) != self.dof:
            raise ValueError(f"MassMatrix expects jointstates length {self.dof}, got {len(jointstates)}")
        return np.array(p.calculateMassMatrix(self.robot, list(jointstates)), dtype=np.float32)

    def getClosestPoints(self, link1, link2):
        # link indices, not joint indices
        return p.getClosestPoints(self.robot, self.robot, 1000, link1, link2)
