#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import csv 
import numpy as np
from scipy.interpolate import make_interp_spline, splprep, splev
from enum import Enum

class MapSplines(Enum):
    CENTER = 0
    OUTER = 1
    INNER = 2


class GlobalRacelinePublisher(Node):
    def __init__(self):
        super().__init__('GlobalRacelinePublisher')
        self.raceline_publisher = self.create_publisher(Path, '/global_raceline', 10)
        self.timer =  self.create_timer(1.0, self.publish_raceline)
        self.coordinates = [None, None, None]
        # discretized splines (1000 points)
        self.map_splines = [None, None, None]
        # spline params (can be evaluated to get any number of points (x, y) over 0 <= u <= 1)
        self.spline_params = [None, None, None]        
        self.get_raceline_from_csv()

    
    def get_raceline_from_csv(self):
        with open('/sim_ws/src/lidar_processing/scripts/recorded_waypoints.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            coordinates = []
            for line in reader:
                coordinate = [float(value) for value in line]
                coordinates.append(coordinate)
            coordinates = np.array(coordinates)
            print(coordinates.shape)
        self.coordinates[MapSplines.CENTER.value] = coordinates
        x = self.coordinates[MapSplines.CENTER.value][:,0]
        y = self.coordinates[MapSplines.CENTER.value][:,1]
        tck, u = splprep([x, y], s=1.1)
        u_smooth = np.linspace(0, 1, 1000)
        x_smooth, y_smooth  = splev(u_smooth, tck)
        x_smooth = np.reshape(x_smooth, (len(x_smooth), 1))
        y_smooth = np.reshape(y_smooth, (len(y_smooth), 1))
        self.global_raceline = np.hstack((x_smooth, y_smooth))

    def get_walls_from_csv(self):
        with open('/sim_ws/src/lidar_processing/scripts/inner_wall.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            coordinates = []
            for line in reader:
                coordinate = [float(value) for value in line]
                coordinates.append(coordinate)
            coordinates = np.array(coordinates)
            print(coordinates.shape)
            self.coordinates[MapSplines.INNER.value] = coordinates
            x = self.coordinates[MapSplines.INNER.value][:,0]
            y = self.coordinates[MapSplines.INNER.value][:,1]
            tck, u = splprep([x, y], s=1.1)
            u_smooth = np.linspace(0, 1, 1000)
            x_smooth, y_smooth  = splev(u_smooth, tck)
            x_smooth = np.reshape(x_smooth, (len(x_smooth), 1))
            y_smooth = np.reshape(y_smooth, (len(y_smooth), 1))
            self.inner_wall = np.hstack((x_smooth, y_smooth))
        with open('/sim_ws/src/lidar_processing/scripts/outer_wall.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            coordinates = []
            for line in reader:
                coordinate = [float(value) for value in line]
                coordinates.append(coordinate)
            coordinates = np.array(coordinates)
            print(coordinates.shape)
            self.outer_wall_coordinates = coordinates
            x = self.coordinates[MapSplines.OUTER.value][:,0]
            y = self.coordinates[MapSplines.OUTER.value][:,1]
            tck, u = splprep([x, y], s=1.1)
            u_smooth = np.linspace(0, 1, 1000)
            x_smooth, y_smooth  = splev(u_smooth, tck)
            x_smooth = np.reshape(x_smooth, (len(x_smooth), 1))
            y_smooth = np.reshape(y_smooth, (len(y_smooth), 1))
            self.outer_wall = np.hstack((x_smooth, y_smooth))
        

    def publish_raceline(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.poses = []
        for i in range(len(self.global_raceline)):
            current_pose = PoseStamped()
            current_pose.header = path_msg.header
            current_pose.pose.position.x = self.global_raceline[i, 0]
            current_pose.pose.position.y = self.global_raceline[i, 1]
            path_msg.poses.append(current_pose)
        self.raceline_publisher.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GlobalRacelinePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()