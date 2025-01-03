#!/usr/bin/env python3

import rosbag
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import matplotlib.pyplot as plt
from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL


class JointAccelerationCalculator:
    def __init__(self, bagfile_path, robot_topic):
        self.bagfile_path = bagfile_path
        self.robot_topic = robot_topic
        self.kdl_solver = Iiwa14DynamicKDL
        self.joint_states = []
        self.time_stamps = []

    def load_bagfile(self):
        """ Loads the bagfile and extracts joint states and time stamps."""
        try:
            with rosbag.Bag(bagfile_path, 'r') as bag:
                topics = bag.get_type_and_topic_info().topics
                print(f"Bagfile contains {len(topics)} topics(s):")
                for topic, info in topics.items():
                    print(f"Topic: {topic}")
                    print(f"  - Message Type: {info.msg_type}")
                    print(f"  - Message Count: {info.message_count}")
                    print(f"  - Frenquency: {info.frequency} Hz")

                print("\nPreviewing first few messages:")

            for topic, msg, t in bag.read_messages(topics=[self.robot_topic]):
                print(f"Topic: {topic}, Time: {t.to_sec()}, Message: {msg}")
                break
        except Exception as e:
            print(f"Error reading bagfile: {e}")

if __name__ == "__main__":
    rospy.init_node("joint_acceleration_calculator")

    # File and topic details
    rospack = rospkg.RosPack()
    bagfile_path = rospack.get_path('cw3q5') + '/bag/cw3q5.bag'
    robot_topic = "/robot/joint_states"

    calculator = JointAccelerationCalculator(bagfile_path, robot_topic)

    # Load and process bagfile
    calculator.load_bagfile()

