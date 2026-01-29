#!/usr/bin/env python3

import gc
import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
import rospkg
import ctypes
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_qp as functions_bounds
import functions as functions_collision
from rdf import query_sdf, query_sdf_batch



import rospy, tf2_ros
from geometry_msgs.msg import TransformStamped
def publish_world_alias():
    br = tf2_ros.StaticTransformBroadcaster()
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"          # parent
    t.child_frame_id = "panda_link0"  # child
    t.transform.translation.x = 0.0
    t.transform.translation.y = 0.0
    t.transform.translation.z = 0.0
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = 1.0
    br.sendTransform(t)



from visualization_msgs.msg import Marker
def make_sphere_marker(position, scale=0.05,
                       color=(0.1, 0.7, 1.0), alpha=0.95,
                       mid=0, ns="targets"):
    """
    Always create a sphere marker in the 'world' frame.

    Args:
        position: (x, y, z)
        scale: sphere diameter in meters
        color: (r, g, b) in [0, 1]
        alpha: transparency in [0, 1]
        mid: marker id
        ns: namespace string
    """
    m = Marker()
    m.header.frame_id = "panda_link0_sc"     # <- fixed to world
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(position[0])
    m.pose.position.y = float(position[1])
    m.pose.position.z = float(position[2])
    m.pose.orientation.w = 1.0
    m.scale.x = scale
    m.scale.y = scale
    m.scale.z = scale
    m.color.r, m.color.g, m.color.b = color
    m.color.a = alpha
    m.lifetime = rospy.Duration(0)   # 0 = forever
    m.frame_locked = True
    return m


def CVXsolver_basic(J_in, fc_in, M_in, tau_in, qdd_lb_in, qdd_ub_in):
    my_c_library = ctypes.CDLL('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/basic.so')
    my_c_library.set_defaults()
    my_c_library.setup_indexing()
    my_c_library.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double)]
    # Get reference to the global variable
    # solver_settings = ctypes.cast(
    #     ctypes.addressof(my_c_library.settings), ctypes.POINTER(Settings)
    # )

    # # Modify its values
    # solver_settings.contents.verbose = 1
    # solver_settings.contents.debug = 1
    my_c_library.CVXsolver.restype = None
    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    # print("J_in:", list(c_J_in))
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    # print("fc_in:", list(c_fc_in))
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    # print("M_in:", list(c_M_in))
    c_b_in = (ctypes.c_double * len(tau_in))(*tau_in)
    # print("b_in:", list(c_b_in))
    c_qdd_lb_in = (ctypes.c_double * len(qdd_lb_in))(*qdd_lb_in)
    # print("qdd_lb_in:", list(c_qdd_lb_in))
    c_qdd_ub_in = (ctypes.c_double * len(qdd_ub_in))(*qdd_ub_in)
    # print("qdd_ub_in:", list(c_qdd_ub_in))
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()
    my_c_library.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in,c_qdd_lb_in,  c_qdd_ub_in,
                           ctypes.byref(opt_x1), ctypes.byref(opt_x2),
                           ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))
    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x

def CVXsolver_sca(J_in, fc_in, M_in, tau_in, qdd_lb_in, qdd_ub_in, g_in, e_in, c_in):
    my_c_library = ctypes.CDLL('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/sca.so')
    my_c_library.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    my_c_library.CVXsolver.restype = None
    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    # print("J_in:", c_J_in)
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    # print("fc_in:", c_fc_in)
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    # print("M_in:", c_M_in)
    c_b_in = (ctypes.c_double * len(tau_in))(*tau_in)
    # print("b_in:", c_b_in)
    c_qdd_lb_in = (ctypes.c_double * len(qdd_lb_in))(*qdd_lb_in)
    # print("qdd_lb_in:", c_qdd_lb_in)
    c_qdd_ub_in = (ctypes.c_double * len(qdd_ub_in))(*qdd_ub_in)
    # print("qdd_ub_in:", c_qdd_ub_in)
    c_g_in = (ctypes.c_double * len(g_in))(*g_in)
    c_e_in = (ctypes.c_double * len(e_in))(*e_in)
    c_c_in = (ctypes.c_double * len(c_in))(*c_in)
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()
    my_c_library.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in, c_qdd_lb_in, c_qdd_ub_in, c_g_in, c_e_in, c_c_in,
                           ctypes.byref(opt_x1), ctypes.byref(opt_x2),
                           ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))
    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x

class subscriber_QP:
    def __init__(self):
        self.sub_joint = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=1)
        self.sub_ee = rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.QP_callback_sub_ee, queue_size=1)
        self.sub_state = rospy.Subscriber("/franka_state_controller/franka_model", Float32MultiArray, self.QP_callback_sub_state, queue_size=1)
        self.sub_obs1 = rospy.Subscriber("/natnet_ros/obs1/pose", PoseStamped, self.QP_callback_sub_obs1, queue_size=10)
        self.sub_obs2 = rospy.Subscriber("/natnet_ros/obs2/pose", PoseStamped, self.QP_callback_sub_obs2, queue_size=10)

        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_obs1 = [0, 0, 0]
        self.message_obs2 = [0, 0, 0]
        self.message_state = list(np.zeros(105))

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position = msg.position
        self.message_joint_velocity = msg.velocity

    def QP_callback_sub_ee(self, msg):
        self.message_ee = [msg.position.x, msg.position.y, msg.position.z]

    def QP_callback_sub_obs1(self, msg):
        # PoseStamped 里位置在 msg.pose.position
        self.message_obs1 = [msg.pose.position.x,
                             msg.pose.position.y,
                             msg.pose.position.z]

    def QP_callback_sub_obs2(self, msg):
        # PoseStamped 里位置在 msg.pose.position
        self.message_obs2 = [msg.pose.position.x,
                             msg.pose.position.y,
                             msg.pose.position.z]

    def QP_callback_sub_state(self, msg):
        self.message_state = msg.data
    
    def return_message(self):
        return 1, self.message_joint_position, self.message_joint_velocity, self.message_ee, self.message_state, self.message_obs1, self.message_obs2
    

if __name__ == "__main__":
    rospy.init_node("cvxpy_QP_all")
    rospy.logwarn("CVXPY QP Node initialized!")
    r = rospy.Rate(1500)  
    # gc.disable()
    publish_world_alias()
    Subscriber_QP = subscriber_QP()
    
    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size=1)
    # publisher (latched so it stays visible)
    tar_pub = rospy.Publisher("/target_marker", Marker, queue_size=1, latch=True)
    obs_pub1 = rospy.Publisher("/obstacle_marker1", Marker, queue_size=1, latch=True)
    obs_pub2 = rospy.Publisher("/obstacle_marker2", Marker, queue_size=1, latch=True)

    lambda1, lambda2, lambda3 = 9, 10, 10
    alpha = 1e-3
    
    # q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    # q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    q_min = np.array([-2.5, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8])
    q_max = np.array([ 2.5,  1.7,  2.8, -0.1,  2.8,  3.7,  2.8])
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])*0.8
    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    
    target_pos = np.array([0.6, 0.1, 0.3])
    target_pos = np.array([0.5, 0.0, 0.5])
    obs_pos = np.array([0.3, 0.0, 0.5])
    dt = 0.02  

    rospy.sleep(1.0)
    t0 = time.time()
    while not rospy.is_shutdown():
        # gc.enable()
        
        # if time.time() - t0 > 100:
        #     gc.collect()
        #     t0 = time.time()

        q = Subscriber_QP.return_message()[1]   
        q_dot = Subscriber_QP.return_message()[2]
        qe = functions_collision.compute_qe(q, q_dot)
        end_pos = Subscriber_QP.return_message()[3]
        obs_pos1 = Subscriber_QP.return_message()[5]-np.array([-2.5058, 1.0381, 0.888])
        obs_pos2 = Subscriber_QP.return_message()[6]-np.array([-2.5058, 1.0381, 0.888])
        print("obs_pos1:", obs_pos1)
        print("obs_pos2:", obs_pos2)
    
        
        Jacobian_raw = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
        Jacobian = Jacobian_raw[0:3, :].T 
        # print("Jacobian:", Jacobian)
        
        xdot = Jacobian.T @ np.array(q_dot)  

        MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
        Gravity = np.array(Subscriber_QP.return_message()[4][7:14])
        # target_frame = rospy.get_param("~target_frame", "panda_link0")
        # obs_pos[2] = 0.5 + 0.3 * np.sin(0.6 * time.time())

        # publish once (latched) or periodically
        tar_pub.publish(make_sphere_marker(target_pos, scale=0.04, color=(0.0, 1.0, 0.0)))
        obs_pub1.publish(make_sphere_marker(obs_pos1, scale=0.08, color=(1.0, 0.0, 0.0)))
        obs_pub2.publish(make_sphere_marker(obs_pos2, scale=0.08, color=(1.0, 0.0, 0.0)))

        # # --- Self-collision gamma and gradient ---
        # start = time.perf_counter()
        gamma_val, grad_gamma = functions_collision.compute_gamma_and_grad(
            np.array(q, dtype=np.float32), np.array(q_dot, dtype=np.float32), threshold=12.5
        )
        # print("gamma:", gamma_val)

        # gamma_val = functions_collision.compute_gamma(
        #     np.array(q, dtype=np.float32), np.array(q_dot, dtype=np.float32)
        # )
        # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
        # print("grad:", grad_gamma)
        # if gamma_val < 10.1:
        #     pass
        # # gamma_val = 11
        # grad_gamma = None
        soft = False
        pose = np.eye(4)
        theta_np = np.stack([q, qe], axis=0).astype(np.float32)   # (B=2, 7)
        points_np = np.stack([
            obs_pos1,
            # obs_pos2
        ], axis=0).astype(np.float32)                             # (N=2, 3)

        pose_np = np.broadcast_to(pose, (2, 4, 4)).astype(np.float32)  # (B=2, 4, 4)

        # ------ 一次批量查询：返回 (B,N) 与 (B,N,7) ------
        dsts, grad_qs = query_sdf_batch(points_np, pose_np, theta_np)  # need_index=False

        # ------ 对应回原来的四个变量 ------
        dst,  grad  = dsts[0, 0], grad_qs[0, 0]   # q,  obs1
        dst2, grad2 = dsts[1, 0], grad_qs[1, 0]   # qe, obs1
        # dst3, grad3 = dsts[0, 1], grad_qs[0, 1]   # q,  obs2
        # dst4, grad4 = dsts[1, 1], grad_qs[1, 1]   # qe, obs2
        dst3=dst4=dst
        grad3=grad4=grad

        dsts = [dst, dst2, dst3, dst4]
        grads = [grad, grad2, grad3, grad4]
        min_idx = int(np.argmin(dsts))
        dst_sdf = dsts[min_idx]
        # print("dst_sdf:", dst_sdf)
        grad_sdf = grads[min_idx]

        

        fx = -2* (np.array(end_pos) - target_pos)
        # print("fx:", np.linalg.norm(fx))
        # if np.linalg.norm(fx) > 1e-2 and np.linalg.norm(fx) < 5e-1 and gamma_val > 12.6 and dst_sdf > 0.151:
        #     fx1 = fx
        #     fx = fx / (np.linalg.norm(fx)**1.2)
        #     if np.linalg.norm(fx1) < 3e-1:
        #         fx = fx / (np.linalg.norm(fx)**1)
        # print("fx after:", np.linalg.norm(fx))
        e1 = fx / np.linalg.norm(fx)
        e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        Lambda = np.diag([lambda1, lambda2, lambda3])
        D = Q @ Lambda @ Q.T
        fc = -D @ (xdot - fx)
        
        start = time.time()
        qdd_lb, qdd_ub, flags = functions_bounds.compute_joint_acceleration_bounds_vec(
            q, q_dot, q_min, q_max, qd_lim, acc_max, dt*0.25, viability=True
        )
        # print("flags:", flags)
        
        if flags.any() != 0:
            # rospy.logwarn("Joint limits are active!")
            indices = np.nonzero(flags)[0]
            # rospy.logwarn("Active joint limits indices: %s", indices)
            for idx in indices:
                if flags[idx] == 1:
                    rospy.logwarn("Joint %d position limit is active.", idx)
                elif flags[idx] == 2:
                    rospy.logwarn("Joint %d velocity limit is active.", idx)
                elif flags[idx] == 3:
                    rospy.logwarn("Joint %d viability limit is active.", idx)
            # rospy.logwarn("Flags values: %s", flags[indices])


        M_inv = np.linalg.pinv(MassMatrix)
        qdd_zero = np.zeros(7)
        tau_id = MassMatrix @ qdd_zero + Coriolis #+ Gravity 

        
        
        
        
        for idx in range(7):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4

        # print("qdd_lb:", qdd_lb)
        # print("qdd_ub:", qdd_ub)
        
        # u = cp.Variable(7)
        # JT_pinv = np.linalg.pinv(Jacobian)
        # objective = cp.sum_squares(JT_pinv @ u - fc) + alpha * cp.sum_squares(u)
        # constraints = [
        #     M_inv @ u >= qdd_lb + M_inv @ tau_id,
        #     M_inv @ u <= qdd_ub + M_inv @ tau_id,
        # ]
        J_in = list(np.linalg.pinv(Jacobian).flatten('F'))
        fc_in = list(fc)
        M_in = list(np.array(M_inv).flatten('F'))
        b_in = list(tau_id)
        qdd_lb_in = list(qdd_lb)
        qdd_ub_in = list(qdd_ub)
        # qdd_lb_in = list(-acc_max)
        # qdd_ub_in = list(acc_max)


        if dst_sdf < 0.15:
            soft = True
            dt = 0.02
            # 1) 计算中间量
            c     = np.dot(grad_sdf, q_dot) * dt                    # grad·qd * dt
            g_eff = 0.5 * grad_sdf * dt**2                       # 0.5 * grad * dt^2
            # qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
            # tau_cmd = MassMatrix @ qdd_cmd + Coriolis
            # target_torque = list(tau_cmd)
            g_eff_in = list(g_eff)
            c_const = float(np.dot(np.asarray(grad_sdf, dtype=np.float64), np.asarray(q_dot, dtype=np.float64)) * dt)
            c_in = [c_const]
            # eps     = 0.5*(1.44 - (gamma_val*0.1)**2)
            eps     = (0.162- dst_sdf)**2
            e_in = [eps]
            try:
                target_torque = CVXsolver_sca(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in, list(g_eff), e_in, c_in)

            except Exception as e:
                rospy.logwarn("CVXsolver_eca failed")

            # print("Using soft constraint!")
            rospy.logwarn("External Collision Avoidance limits are active!")

        
        # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
        if not soft:
            if grad_gamma is not None:
                rospy.logwarn("Self Collision Avoidance limits are active!")
                
                grad_q  = grad_gamma[:7]
                grad_qd = grad_gamma[7:]
                g_eff   = 0.5 * grad_q * dt**2 + grad_qd * dt
                g_eff_in = list(g_eff)
                c_const = float(np.dot(np.asarray(grad_q, dtype=np.float64), np.asarray(q_dot, dtype=np.float64)) * dt)
                c_in = [c_const]
                # eps     = 0.5*(1.44 - (gamma_val*0.1)**2)
                eps     = 0.1*(1.25 - (gamma_val*0.1))
                e_in = [eps]
                try:
                    target_torque = CVXsolver_sca(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in, list(g_eff), e_in, c_in)

                except Exception as e:
                    rospy.logwarn("CVXsolver_sca failed")
                    # target_torque = list(Jacobian.T @ fc)
            else:
            # if True:
                try:
                    # start = time.time()
                    target_torque = CVXsolver_basic(J_in, fc_in, M_in, b_in, qdd_lb_in, qdd_ub_in)
                    # print(target_torque)
                    
                    # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
                    
                except Exception as e:
                    rospy.logwarn("CVXsolver_basic failed")
                    # target_torque = list(Jacobian.T @ fc)
    
        end = time.time()
            # target_torque = np.array(u.value)
            # print(f"Execution time: {(end - start) * 1e3:.2f} ms")
            # target_torque = [-14, -14, -14, -14, -2, -2, -2]
        target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
        msg_torque = Float32MultiArray()
        msg_torque.data = target_torque
        # print("torque:", target_torque)
        pub.publish(msg_torque)
        
        
    
    
        # r.sleep()

