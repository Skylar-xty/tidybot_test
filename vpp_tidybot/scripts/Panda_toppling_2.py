# Panda_toppling.py
import os
import time
import numpy as np
import pybullet as p
import pybullet_data


class Panda:
    """
    Unified wrapper that supports:
      A) Planar prismatic base in URDF (world_to_base_x, base_x_to_base_y) -> 9DoF (2 base + 7 arm)
      B) Free-base URDF (base_link is root, loaded with useFixedBase=False) -> 13DoF state (6 base + 7 arm)
         - Actuation is still only 7 arm joints; base is controlled by applying wrench (force/torque).
         - setTargetTorques supports:
             * 7-dim: arm torques
             * 9-dim: (only if planar base joints exist) base2 + arm7 torques
             * 13-dim: (only if free-base) base6 wrench + arm7 torques
    """

    def __init__(
        self,
        stepsize=1e-3,
        realtime=0,
        connection_mode=p.GUI,
        urdf_path=None,
        use_fixed_base=True,
        gravity=-9.81,
    ):
        self.t = 0.0
        self.stepsize = float(stepsize)
        self.realtime = int(realtime)

        # connect pybullet
        if connection_mode == p.GUI:
            p.connect(connection_mode, options="--width=2048 --height=1536")
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            p.connect(connection_mode)

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, float(gravity))
        p.setTimeStep(self.stepsize)

        if urdf_path is None:
            raise ValueError("urdf_path must be provided for Panda_toppling wrapper.")

        print("urdf_path =", urdf_path, "exists:", os.path.exists(urdf_path))

        self.robot = p.loadURDF(
            urdf_path,
            useFixedBase=bool(use_fixed_base),
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )

        self.num_joints = p.getNumJoints(self.robot)

        # ----------------------------
        # Detect base mode + indices
        # ----------------------------
        self.planar_base_joint_names = ("world_to_base_x", "base_x_to_base_y")
        name_to_jid = {}
        for j in range(self.num_joints):
            info = p.getJointInfo(self.robot, j)
            name_to_jid[info[1].decode("utf-8")] = j

        self.base_joint_ids = []
        for nm in self.planar_base_joint_names:
            if nm in name_to_jid:
                self.base_joint_ids.append(name_to_jid[nm])

        self.has_planar_base_joints = (len(self.base_joint_ids) == 2)

        # Arm joints: iiwa_joint_1..7 (revolute)
        self.arm_joint_ids = []
        self.q_min_arm = []
        self.q_max_arm = []

        print("---------------- Joint Info ----------------")
        for j in range(self.num_joints):
            info = p.getJointInfo(self.robot, j)
            jname = info[1].decode("utf-8")
            jtype = info[2]
            print(f"ID: {j}, Name: {jname}, Type: {jtype}")

            if jtype != p.JOINT_FIXED and jname.startswith("iiwa_joint"):
                self.arm_joint_ids.append(j)
                self.q_min_arm.append(info[8])
                self.q_max_arm.append(info[9])

        if len(self.arm_joint_ids) != 7:
            print(f"[WARN] Expected 7 arm joints, found {len(self.arm_joint_ids)}. Check URDF joint naming.")

        # Decide "actuated joint list" for motor commands
        if self.has_planar_base_joints:
            # Actuate base prismatic + arm joints
            self.base_mode = "planar_joint_base"
            self.base_dof = 2
            self.joints = list(self.base_joint_ids) + list(self.arm_joint_ids)  # 9 actuated DoF
            self.dof_actuated = len(self.joints)  # should be 9
            self.dof_state = self.dof_actuated    # 9 state as well (because base is joints)
        else:
            # Free base: no base joints exist in URDF; base is 6DoF rigid body
            self.base_mode = "free_base"
            self.base_dof = 6
            self.joints = list(self.arm_joint_ids)      # only 7 actuated joints
            self.dof_actuated = len(self.joints)        # should be 7
            self.dof_state = self.base_dof + self.dof_actuated  # 13

        print(f"[INFO] base_mode={self.base_mode}, dof_actuated={self.dof_actuated}, dof_state={self.dof_state}")

        # end-effector link index: last link is typically ee; keep consistent with your old code
        self.ee_link_index = self.num_joints - 1

        # controller mode
        self.control_mode = "torque"
        self.resetController()

    # ----------------------------
    # Core stepping
    # ----------------------------
    def step(self):
        if self.realtime:
            time.sleep(self.stepsize)
        p.stepSimulation()
        self.t += self.stepsize

    # ----------------------------
    # Controller setup
    # ----------------------------
    def resetController(self):
        """
        Disable default motor controllers so TORQUE_CONTROL actually applies.
        """
        # For any actuated joint in self.joints: disable velocity motors first
        for jid in self.joints:
            p.setJointMotorControl2(self.robot, jid, controlMode=p.VELOCITY_CONTROL, force=0)

        self.control_mode = "torque"

    def setControlMode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in ("position", "velocity", "torque"):
            raise ValueError("wrong control mode")
        if mode == "torque" and self.control_mode != "torque":
            self.resetController()
        self.control_mode = mode

    # ----------------------------
    # State getters (unified)
    # ----------------------------
    def getBaseState(self):
        """
        Returns base pose/vel in WORLD frame.
        pos: (3,), quat: (4,), linVel: (3,), angVel: (3,)
        """
        pos, quat = p.getBasePositionAndOrientation(self.robot)
        linVel, angVel = p.getBaseVelocity(self.robot)
        return (np.array(pos, dtype=np.float32),
                np.array(quat, dtype=np.float32),
                np.array(linVel, dtype=np.float32),
                np.array(angVel, dtype=np.float32))

    def getBaseRPY(self):
        _, quat, _, _ = self.getBaseState()
        rpy = p.getEulerFromQuaternion(quat.tolist())
        return np.array(rpy, dtype=np.float32)

    def getJointStates(self):
        """
        Backward-compatible: returns actuated joint states (length dof_actuated).
          - planar_joint_base: returns [base_x, base_y, arm1..7]
          - free_base: returns [arm1..7]
        """
        joint_states = p.getJointStates(self.robot, self.joints)
        q = [s[0] for s in joint_states]
        qd = [s[1] for s in joint_states]
        return np.array(q, dtype=np.float32), np.array(qd, dtype=np.float32)

    def getState(self):
        """
        Unified full state:
          - planar_joint_base: q,qd are 9D  (2 base joints + 7 arm)
          - free_base:         q,qd are 13D (base6 + arm7)
            q_base  = [x,y,z, roll,pitch,yaw]
            qd_base = [vx,vy,vz, wx,wy,wz]
        """
        if self.base_mode == "planar_joint_base":
            return self.getJointStates()

        # free base
        pos, quat, linVel, angVel = self.getBaseState()
        rpy = np.array(p.getEulerFromQuaternion(quat.tolist()), dtype=np.float32)

        q_arm, qd_arm = self.getJointStates()  # 7
        q = np.concatenate([pos, rpy, q_arm], axis=0)
        qd = np.concatenate([linVel, angVel, qd_arm], axis=0)
        return q, qd

    # ----------------------------
    # Actuation (unified)
    # ----------------------------
    def applyBaseWrench(self, force_world=(0, 0, 0), torque_world=(0, 0, 0)):
        """
        Apply force+torque on base (linkIndex=-1) in WORLD frame for ONE step.
        Call this every control tick if you want continuous wrench.
        """
        p.applyExternalForce(self.robot, -1, forceObj=list(force_world), posObj=[0, 0, 0], flags=p.WORLD_FRAME)
        p.applyExternalTorque(self.robot, -1, torqueObj=list(torque_world), flags=p.WORLD_FRAME)

    def setTargetTorques(self, target):
        """
        TORQUE command:
          - If len(target) == dof_actuated: apply directly to actuated joints.
          - If planar base and len(target)==9: same as above.
          - If free base and len(target)==13: interpret as [Fx,Fy,Fz,Tx,Ty,Tz, tau_arm(7)].
        """
        target = list(target)

        # free-base "one-piece": base6 wrench + arm7 torques
        if self.base_mode == "free_base" and len(target) == self.dof_state:
            Fx, Fy, Fz, Tx, Ty, Tz = target[0:6]
            tau_arm = target[6:]
            self.applyBaseWrench(force_world=(Fx, Fy, Fz), torque_world=(Tx, Ty, Tz))
            target = tau_arm  # now apply arm torques

        # planar base: 9D direct torque control (prismatic force for base joints)
        if len(target) != self.dof_actuated:
            raise ValueError(f"setTargetTorques expected length {self.dof_actuated} (or 13 in free_base), got {len(target)}")

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.TORQUE_CONTROL,
            forces=target,
        )

    # ----------------------------
    # Dynamics helpers
    # ----------------------------
    # def solveInverseDynamics(self, pos, vel, acc):
    #     """
    #     Inverse dynamics torque for actuated joints.

    #     For planar_joint_base: expects pos/vel/acc length 9, returns 9 torques.
    #     For free_base: expects pos/vel/acc length 7 (arm), returns 7 torques.
    #       - If you pass 13D (base+arm), we will use the last 7 as arm vectors.
    #     """
    #     pos = np.array(pos, dtype=np.float32).reshape(-1)
    #     vel = np.array(vel, dtype=np.float32).reshape(-1)
    #     acc = np.array(acc, dtype=np.float32).reshape(-1)

    #     if self.base_mode == "free_base":
    #         if pos.size == self.dof_state:
    #             pos = pos[6:]
    #             vel = vel[6:]
    #             acc = acc[6:]
    #         if pos.size != self.dof_actuated:
    #             raise ValueError(f"free_base solveInverseDynamics expects 7 (or 13), got {pos.size}")

    #     else:
    #         if pos.size != self.dof_actuated:
    #             raise ValueError(f"planar_base solveInverseDynamics expects 9, got {pos.size}")

    #     # PyBullet inverse dynamics expects vectors length == number of (non-fixed) joints in the body.
    #     # Here we provide only the actuated joint vectors (planar base joints + arm joints) consistent with self.joints.
    #     try:
    #         torques = p.calculateInverseDynamics(self.robot, list(pos), list(vel), list(acc), flags=1)
    #         return list(torques)
    #     except Exception:
    #         # Some Bullet builds can be finicky in certain configurations; safe fallback
    #         return [0.0] * int(pos.size)

    def solveInverseDynamics(self, pos, vel, acc):
        """
        Returns torques for actuated joints in a consistent way:
        - planar_joint_base: returns len(self.joints) = 9  (base2 + arm7)
        - free_base: returns 7 (arm only), even if Bullet returns [base6 + jointN]
        """
        pos = np.array(pos, dtype=np.float32).reshape(-1)
        vel = np.array(vel, dtype=np.float32).reshape(-1)
        acc = np.array(acc, dtype=np.float32).reshape(-1)

        # Decide what we pass into Bullet
        if self.base_mode == "free_base":
            # If user passes full state (13), use arm part (7) as joint vectors
            if pos.size == self.dof_state:
                pos_j = pos[6:]
                vel_j = vel[6:]
                acc_j = acc[6:]
            else:
                pos_j = pos
                vel_j = vel
                acc_j = acc

            if pos_j.size != self.dof_actuated:
                raise ValueError(f"free_base solveInverseDynamics expects 7 (or 13), got {pos.size}")

            try:
                out = p.calculateInverseDynamics(self.robot, list(pos_j), list(vel_j), list(acc_j),flags=1)
                out = np.asarray(out, dtype=np.float32).reshape(-1)

                # Bullet may return:
                #   (a) num_joints
                #   (b) 6 + num_joints   (base wrench + joint torques)
                if out.size == self.num_joints:
                    joint_tau_all = out
                elif out.size == 6 + self.num_joints:
                    joint_tau_all = out[6:]
                else:
                    # fallback heuristic: if longer than 6+num_joints, take the tail that matches joints
                    if out.size > 6 + self.num_joints:
                        joint_tau_all = out[-self.num_joints:]
                    else:
                        # unknown format: safest fallback
                        return [0.0] * 7

                # Map to arm joint torques by joint index
                tau_arm = [float(joint_tau_all[jid]) for jid in self.arm_joint_ids]
                return tau_arm

            except Exception:
                return [0.0] * 7

        # ---------------- planar base joints case ----------------
        # Here the actuated joints are explicit in the URDF; we want torques for self.joints (len 9)
        if pos.size != self.dof_actuated:
            raise ValueError(f"planar_base solveInverseDynamics expects {self.dof_actuated}, got {pos.size}")

        try:
            out = p.calculateInverseDynamics(self.robot, list(pos), list(vel), list(acc),flags=1)
            out = np.asarray(out, dtype=np.float32).reshape(-1)

            if out.size == self.num_joints:
                joint_tau_all = out
            elif out.size == 6 + self.num_joints:
                joint_tau_all = out[6:]
            else:
                if out.size > 6 + self.num_joints:
                    joint_tau_all = out[-self.num_joints:]
                else:
                    return [0.0] * self.dof_actuated

            tau_act = [float(joint_tau_all[jid]) for jid in self.joints]
            return tau_act

        except Exception:
            return [0.0] * self.dof_actuated

    def getMassMatrix(self, q):
        """
        Mass matrix for actuated joints.
          - planar_base: q is 9 -> returns 9x9
          - free_base: q is 7 (or 13, last 7 used) -> returns 7x7
        """
        q = np.array(q, dtype=np.float32).reshape(-1)
        if self.base_mode == "free_base" and q.size == self.dof_state:
            q = q[6:]
        return np.array(p.calculateMassMatrix(self.robot, list(q)), dtype=np.float32)

    def solveForwardKinematics(self):
        pos, ori = p.getLinkState(self.robot, self.ee_link_index)[0:2]
        return np.array(pos, dtype=np.float32), np.array(ori, dtype=np.float32)

    def getEndVelocity(self):
        # linkState[6] linear velocity; [7] angular velocity when computeLinkVelocity=True
        ls = p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=True)
        v = np.array(ls[6], dtype=np.float32)
        return v

    def getJacobian(self):
        """
        Returns linear Jacobian wrt actuated joints.
          - planar_base: 3x9
          - free_base:  3x7
        """
        q, qd = self.getJointStates()
        jointvel = [0.0] * int(q.size)
        jointacc = [0.0] * int(q.size)

        Jlin, _ = p.calculateJacobian(
            self.robot,
            self.ee_link_index,
            [0, 0, 0],
            list(q),
            jointvel,
            jointacc,
        )
        return np.array(Jlin, dtype=np.float32)

    def getClosestPoints(self, link1, link2, distance=1000.0):
        return p.getClosestPoints(self.robot, self.robot, distance, link1, link2)
