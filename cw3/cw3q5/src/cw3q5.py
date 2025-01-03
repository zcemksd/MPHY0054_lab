#!/usr/bin/env python3

import rospy
import rosbag
import rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL
import numpy as np
import matplotlib.pyplot as plt


class JointAccelerationCalculator:
    def __init__(self):
        self.time_stamps = []
        self.acceleration_data = [np.array([]) for _ in range(7)]        

    def load_trajectory(self):
        """Load trajectory from the bagfile and publish it."""

        rospy.loginfo("Loading trajectory from bagfile...")
        joint_traj = JointTrajectory()
        rospack = rospkg.RosPack()
        bagfile_path = rospack.get_path('cw3q5') + '/bag/cw3q5.bag'

        try:
            with rosbag.Bag(bagfile_path, 'r') as bag:
                # Print message type, topic, and message count
                topics = bag.get_type_and_topic_info()

                for topic, msg, t in bag.read_messages(topics=['/iiwa/EffortJointInterface_trajectory_controller/command']):
                    joint_traj.header.stamp = rospy.Time.now()
                    joint_traj.joint_names = msg.joint_names

                    for point in msg.points:
                        point_obj = JointTrajectoryPoint()
                        point_obj.positions = list(point.positions)
                        point_obj.velocities = list(point.velocities)
                        point_obj.accelerations = list(point.accelerations)
                        point_obj.time_from_start = point.time_from_start
                        joint_traj.points.append(point_obj)
                    
            rospy.loginfo("Trajectory successfully loaded from bagfile.")
            return joint_traj


        except Exception as e:
            rospy.logerr(f"Error loading trajectory from bagfile: {e}")
            return None
        
    def calculate_acceleration(self, joint_state):
        """Calculate joint accelerations using dynamics."""
        q = np.array(joint_state.position)
        q_dot = np.array(joint_state.velocity)
        tau = np.array(joint_state.effort)

        print(f"q shape: {q.shape}")
        print(f"q_dot shape: {q_dot.shape}")
        print(f"tau shape: {tau.shape}")

        try:
            B = Iiwa14DynamicKDL.get_B(Iiwa14DynamicKDL(), q)
            C_qdot = Iiwa14DynamicKDL.get_C_times_qdot(Iiwa14DynamicKDL(), q, q_dot)
            G = Iiwa14DynamicKDL.get_G(Iiwa14DynamicKDL(), q)

            print(f"B shape: {B.shape}")
            print(f"C_qdot shape: {C_qdot.shape}")
            print(f"G shape: {G.shape}")

            q_ddot = np.linalg.inv(B).dot(tau - C_qdot - G)
            print(f"q_ddot shape: {q_ddot.shape}")

            time_stamp = rospy.Time.now().to_sec()
            self.time_stamps.append(time_stamp)

            for i in range(7):
                if self.acceleration_data[i].size == 0:
                    self.acceleration_data[i] = np.array([q_ddot[i]])
                else:
                    self.acceleration_data[i] = np.append(self.acceleration_data[i], q_ddot[i])
            
            self.plot_acceleration()

        except Exception as e:
            rospy.logerr(f"Error calculating accelerations: {e}")

    def plot_acceleration(self):
        """Plot joint accelerations as a function of time."""
        if len(self.time_stamps) < 2:
            return
        
        plt.clf()
        for i in range(7):
            plt.plot(self.time_stamps, self.acceleration_data[i], label=f"Joint {i+1}")

        plt.title("Joint Acceleration Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (rad/s^2)")
        plt.legend()
        plt.draw()
        plt.pause(1e-5)
        

if __name__ == "__main__":

    try:
        rospy.init_node("joint_acceleration_calculator", anonymous=True)
        calculator = JointAccelerationCalculator()

        joint_traj = calculator.load_trajectory()
        if joint_traj:
            traj_pub = rospy.Publisher('/iiwa/EffortJointInterface_trajectory_controller/command', JointTrajectory, queue_size=5)
            rospy.sleep(1)
            traj_pub.publish(joint_traj)
            rospy.loginfo("Trajectory published to topic.")

        rospy.Subscriber('/iiwa/joint_states', JointState, calculator.calculate_acceleration)

        plt.ion()
        plt.show()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node terminated.")
