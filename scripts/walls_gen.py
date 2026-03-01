#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import csv
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import buffer, transform_listener
from tf2_geometry_msgs import do_transform_pose_stamped
import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from enum import Enum
from rclpy.duration import Duration

class Wall(Enum):
    INNER = 0
    OUTER = 1

class WallLogger(Node):
    def __init__(self):
        super().__init__('WallLogger')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.inner_waypoints = None
        self.outer_waypoints = None
        self.angle_max = np.pi * 3.0/4.0
        self.angle_min = -1 * np.pi * 3.0/4.0
        self.tf_buffer = buffer.Buffer(node=self)
        self.tf_listener = transform_listener.TransformListener(self.tf_buffer, self, qos=10)

    def lidar_callback(self, msg : LaserScan):
        target_angle_rad = np.pi/2.0
        inner_wall_idx = int(target_angle_rad/self.angle_max * 540.0) + 540
        outer_wall_idx = int(target_angle_rad/self.angle_min * 540.0) + 540
        range_inner_wall = msg.ranges[inner_wall_idx]
        range_outer_wall = msg.ranges[outer_wall_idx]
        # inner wall pose

        self.get_logger().info(
            str(self.tf_buffer.can_transform(
                'map',
                'ego_racecar/base_link',
                Time()
        )))

        # self.get_logger().info(
        #     str(self.tf_buffer.can_transform(
        #         'ego_racecar/base_link',
        #         'ego_racecar/laser',
        #         Time()
        # )))

        pose_inner_wall = PoseStamped()
        pose_inner_wall.header.frame_id = 'ego_racecar/laser'
        pose_inner_wall.pose.position.x = -1.0 * range_inner_wall
        pose_inner_wall.pose.position.y = 0.0
        pose_inner_wall.pose.position.z = 0.0
        # outer wall pose
        pose_outer_wall = PoseStamped()
        pose_outer_wall.header.frame_id = 'ego_racecar/laser'
        pose_outer_wall.pose.position.x = 1.0 * range_outer_wall
        pose_outer_wall.pose.position.y = 0.0
        pose_outer_wall.pose.position.z = 0.0
        

        pose_outer_wall_map = self.tf_buffer.transform(pose_outer_wall, 'map', timeout=Duration(seconds=1))
        pose_inner_wall_map = self.tf_buffer.transform(pose_inner_wall, 'map', timeout=Duration(seconds=1))
        
        if self.update_waypoints(pose_outer_wall_map, Wall.OUTER):
            self.append_to_csv(pose_outer_wall_map, Wall.OUTER)
        if self.update_waypoints(pose_inner_wall_map, Wall.INNER):
            self.append_to_csv(pose_inner_wall_map, Wall.INNER)
        
    def update_waypoints(self, new_pose : PoseStamped, wall : Enum):
        if wall == Wall.INNER:
            if self.inner_waypoints is not None:
                distance_squared = (new_pose.pose.position.x - self.inner_waypoints[-1, 0])**2 + (new_pose.pose.position.y - self.inner_waypoints[-1, 1])**2
                if distance_squared > (0.1):
                    self.inner_waypoints = np.append(self.inner_waypoints, np.array([new_pose.pose.position.x, new_pose.pose.position.y]).reshape(1, 2), axis=0)
                    return True
            else:
                self.inner_waypoints = np.array([new_pose.pose.position.x, new_pose.pose.position.y]).reshape(1, 2)
                return True
        elif wall == Wall.OUTER:
            if self.outer_waypoints is not None:
                distance_squared = (new_pose.pose.position.x - self.outer_waypoints[-1, 0])**2 + (new_pose.pose.position.y - self.outer_waypoints[-1, 1])**2
                if distance_squared > (0.1 ** 2):
                    self.outer_waypoints = np.append(self.outer_waypoints, np.array([new_pose.pose.position.x, new_pose.pose.position.y]).reshape(1, 2), axis=0)
                    return True
            else:
                self.outer_waypoints = np.array([new_pose.pose.position.x, new_pose.pose.position.y]).reshape(1, 2)
                return True
        return False

    def append_to_csv(self, new_pose : PoseStamped, wall : Enum):
        if wall == Wall.INNER:
            with open("/sim_ws/src/lidar_processing/scripts/inner_wall.csv", 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([new_pose.pose.position.x, new_pose.pose.position.y])
                self.get_logger().info(f'x: {new_pose.pose.position.x}, y: {new_pose.pose.position.y}')
        elif wall == Wall.OUTER:
            with open('/sim_ws/src/lidar_processing/scripts/outer_wall.csv', 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([new_pose.pose.position.x, new_pose.pose.position.y])
                self.get_logger().info(f'x: {new_pose.pose.position.x}, y: {new_pose.pose.position.y}')

def main(args=None):
    rclpy.init(args=args)
    node = WallLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()