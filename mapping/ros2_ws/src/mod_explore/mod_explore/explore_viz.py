#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import numpy as np
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation 

import csv
import os


def q2rpy(q):
    r = Rotation.from_quat(q)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return roll, pitch, yaw


class DummyViz(Node):
    """DummyViz"""
    def __init__(self):
        super().__init__("DummyViz")
        self.markers_cleared = False
        
        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        
        self.declare_parameter('is_gazebo', True)
        self.declare_parameter('sim_offset', [0.0, 0.0, 0.0])
        self.is_gazebo     = bool(self.get_parameter('is_gazebo').value)
        self.sim_offset     = (self.get_parameter('sim_offset').value)
        
        # STANDARD CALLBACKS
        self.leader_pos_sub = self.create_subscription(
            PoseStamped, 
            f'/leader/mavros/local_position/pose', 
            self.leader_pose_callback, 
            qos
        )
        self.leader_pos = np.array([])
        
        self.follower_pos_sub = self.create_subscription(
            PoseStamped, 
            f'/follower/mavros/local_position/pose', 
            self.follower_pose_callback, 
            qos
        )
        self.follower_pos = np.array([])
        self.follower_pose_msg = PoseStamped()
    
        
        # RELTAED TO ACOUSTIC SHIT
        self.acoustic_inference_sub = self.create_subscription(
            Point, 
            f'/acoustic/inference/machine', 
            self.acoustic_inference_callback, 
            qos
        )
        self.leader_via_inference = np.array([0,0])
        
        self.acoustic_inference_pub = self.create_publisher(
            PointStamped,
            '/acoustic/guess',
            10
        )
        self.lala_i = 0
        
        self.acoustic_pf_inference_pub = self.create_publisher(
            PointStamped,
            '/acoustic/pf_guess',
            10
        )
        self.acoustic_guess_pf = PointStamped()

        
        
        
        # VIZ TOOLS
        # self.marker_spring_pub = self.create_publisher(
        #     Marker, 
        #     '/acoustic/tracker_viz', 
        #     10
        # )
        self.marker_spring_pts = Marker()
        
        self.marker_trajs_pub = self.create_publisher(
            Marker, 
            '/acoustic/trajs', 
            10
        )
        self.marker_leader_traj = Marker()
        self.marker_follower_traj = Marker()
        self.traj_pts_leader = []
        self.traj_pts_follower = []
        self.traj_pts_follower_desired = []
        self.traj_estimated = []
        
        # Hard reset all markers at the very beginning
        self.get_logger().info("Performing hard reset - clearing all old markers...")
        self.clear_all_markers()
        self.markers_cleared = True
        
        self.leader_ego_pos_pub = self.create_publisher(
            PoseStamped, 
            '/acoustic/leader_pose', 
            10
        )
        self.follower_ego_pos_pub = self.create_publisher(
            PoseStamped, 
            '/acoustic/follower_pose', 
            10
        )
        self.follower_desired_pos_sub = self.create_subscription(
            PoseStamped, 
            '/gazebo_follower/mavros/local_position/pose', 
            self.follower_desired_pose_callback, 
            qos
        )
        self.follower_desired = np.array([0.0,0.0,0.0])
        
        self.follower_pose_rviz = PoseStamped()
        self.leader_pose_rviz = PoseStamped()
        self.viz_start = False
        self.leader_pos_start = False
        self.time_start = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9
    
    
    def set_markers(self):
        
        
        self.marker_leader_traj.header.frame_id = "map"
        self.marker_leader_traj.header.stamp = self.get_clock().now().to_msg()
        self.marker_leader_traj.ns = "trajs"
        self.marker_leader_traj.id = 0
        self.marker_leader_traj.type = Marker.LINE_STRIP
        self.marker_leader_traj.action = Marker.ADD
        self.marker_leader_traj.scale.x = 0.05
        self.marker_leader_traj.color = ColorRGBA(a=1.0, r=0.0, g=1.0, b=0.0)
        self.marker_leader_traj.points = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.traj_pts_leader
        ]

        self.marker_follower_traj.header.frame_id = "map"
        self.marker_follower_traj.header.stamp = self.get_clock().now().to_msg()
        self.marker_follower_traj.ns = "trajs"
        self.marker_follower_traj.id = 1
        self.marker_follower_traj.type = Marker.LINE_STRIP
        self.marker_follower_traj.action = Marker.ADD
        self.marker_follower_traj.scale.x = 0.05
        self.marker_follower_traj.color = ColorRGBA(a=1.0, r=0.0, g=0.0, b=1.0)
        self.marker_follower_traj.points = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.traj_pts_follower
        ]
   
    def leader_pose_callback(self, msg):  
        self.viz_start = True
        self.leader_pose_rviz = msg
        self.leader_pose = msg
        
        if self.is_gazebo:
            self.leader_pos = np.array([
                msg.pose.position.x + self.sim_offset[0],
                msg.pose.position.y + self.sim_offset[1],
                msg.pose.position.z + self.sim_offset[2]
            ]) 
            self.leader_pose_rviz.pose.position.x = self.leader_pose_rviz.pose.position.x + self.sim_offset[0] 
            self.leader_pose_rviz.pose.position.y = self.leader_pose_rviz.pose.position.y + self.sim_offset[1]
            self.leader_pose_rviz.pose.position.z = self.leader_pose_rviz.pose.position.z + self.sim_offset[2]
        else:
            self.leader_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]) 
        
        self.leader_pos_start = True
        self.traj_pts_leader.append(self.leader_pos)                        
        
    def acoustic_inference_callback(self, msg: Point):           
        
        if not self.viz_start or not self.leader_pos_start or not self.follower_pos.size == 3:
            return       
        
        bearing = msg.x  # Same convention as acoustic model: positive = left, negative = right
        range = msg.y
        
        r_rel_B = np.array([
            range * np.cos(bearing / 180.0 * np.pi), 
            range * np.sin(bearing / 180.0 * np.pi),
            0
        ])
        
        _, _, ego_yaw = q2rpy([
            self.follower_pose_msg.pose.orientation.x,
            self.follower_pose_msg.pose.orientation.y,
            self.follower_pose_msg.pose.orientation.z,
            self.follower_pose_msg.pose.orientation.w
        ])    
                                               
        self.lala_i = self.lala_i + 1                    
        
    
    def follower_desired_pose_callback(self, msg):  
        if not self.viz_start:
            return
        self.follower_desired[0] = msg.pose.position.x
        self.follower_desired[1] = msg.pose.position.y
        self.follower_desired[2] = self.follower_pose_msg.pose.position.z
        
        self.traj_pts_follower_desired.append(self.follower_desired)
    
    def follower_pose_callback(self, msg):      
        
        if not self.viz_start:
            return
        
        self.follower_pose_msg = msg
        
        self.follower_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]) 
        
        self.traj_pts_follower.append(self.follower_pos)
        

            

        self.set_markers()
        
        # self.marker_spring_pub.publish(self.marker_spring_pts)
        
        # Publish leader and follower trajectories
        self.marker_trajs_pub.publish(self.marker_leader_traj)
        self.marker_trajs_pub.publish(self.marker_follower_traj)
        
        self.follower_pose_rviz = msg
        self.follower_ego_pos_pub.publish(self.follower_pose_rviz)
        self.leader_ego_pos_pub.publish(self.leader_pose_rviz)

        
    
    def clear_all_markers(self):
        """Delete all old markers to clean up visualization - hard reset"""
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        
        # Clear all possible marker namespaces
        namespaces = [
            # "spring", 
            "trajs", 
            "trajs_points", 
            "trajs_lines", 
            "current_positions", 
            "acoustic_estimates",
            "leader",
            "follower"
        ]
        
        for ns in namespaces:
            delete_marker.ns = ns
            # Publish delete command on both publishers to ensure cleanup
            # self.marker_spring_pub.publish(delete_marker)
            self.marker_trajs_pub.publish(delete_marker)
        
        self.get_logger().info("Cleared all old markers - hard reset complete")
# ----
# Main
# ----
def main(args=None):
    rclpy.init(args=args)
    dummy = DummyViz()
    rclpy.spin(dummy)
    dummy.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()