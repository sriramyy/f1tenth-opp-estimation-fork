#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator
from std_msgs.msg import Float64MultiArray, String
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.time import Time
import csv

class IMMNode(Node):
    def __init__(self):
        super().__init__('imm_predictor')
        
        # Initial dt - updated dynamically in callback
        self.dt = 0.05
        self.first_callback = True
        self.last_odom_pub_time = self.get_clock().now()
        self.filter_counts = np.zeros(3)
        self.num_callbacks = 0
        self.frequencies = np.empty(3, dtype=np.float64)

        # Create the kalman filters: [x, vx, ax, y, vy, ay]
        kf_cv = self.create_kf_cv(self.dt)
        kf_ca = self.create_kf_ca(self.dt)
        kf_ct = self.create_kf_ct(self.dt, w=0.5)
        
        filters = [kf_cv, kf_ca, kf_ct]
        mu = [0.33, 0.33, 0.34]
        trans = np.array([[0.98, 0.01, 0.01],   
                          [0.01, 0.98, 0.01], 
                          [0.01, 0.01, 0.98]])
        
        self.imm_model = IMMEstimator(filters, mu, trans)
        
        # Set testing=False to use detections from Ego Car LiDAR (/state_vector)
        # Set testing=True for simulator ground truth debugging
        self.testing = True 
    
        if not self.testing:
            self.state_sub = self.create_subscription(
                Float64MultiArray, '/state_vector', self.state_callback, 10)
        else:
            self.odom_sub = self.create_subscription(
                Odometry, '/opp_racecar/odom', self.odom_callback, 10)
        
        self.traj_pub = self.create_publisher(Path, '/imm_path', 10)
        self.chosen_filter_pub = self.create_publisher(String, '/chosen_filter', 10)
        self.heatmap_pub = self.create_publisher(MarkerArray, '/imm_heatmap', 10)

        self.last_publish_time = self.get_clock().now()
        
    def create_kf_cv(self, dt):
        kf = KalmanFilter(dim_x=6, dim_z=2)
        kf.F = np.array([[1, dt, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, dt, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        kf.R, kf.P = np.eye(2) * 0.05, np.eye(6) * 1.0
        kf.Q = np.diag([0.01, 0.2, 1.0, 0.01, 0.2, 1.0])
        kf.x = np.zeros(6)
        return kf

    def create_kf_ca(self, dt):
        kf = KalmanFilter(dim_x=6, dim_z=2)
        kf.F = np.array([[1, dt, 0.5*dt**2, 0, 0, 0], [0, 1, dt, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, dt, 0.5*dt**2], [0, 0, 0, 0, 1, dt], [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        kf.R, kf.P = np.eye(2) * 0.05, np.eye(6) * 2.0
        kf.Q = np.diag([0.01, 0.3, 1.0, 0.01, 0.3, 1.0])
        kf.x = np.zeros(6)
        return kf

    def create_kf_ct(self, dt, w):
        kf = KalmanFilter(dim_x=6, dim_z=2)
        if w == 0: w = 0.01
        c, s = np.cos(w*dt), np.sin(w*dt)
        kf.F = np.array([[1, s/w, (1-c)/w**2, 0, 0, 0], [0, c, s/w, 0, 0, 0], [0, -w*s, c, 0, 0, 0],
                         [0, 0, 0, 1, s/w, (1-c)/w**2], [0, 0, 0, 0, c, s/w], [0, 0, 0, 0, -w*s, c]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        kf.R, kf.P = np.eye(2) * 0.05, np.eye(6) * 1.0
        kf.Q = np.diag([0.01, 0.5, 1.0, 0.01, 0.5, 1.0])
        kf.x = np.zeros(6)
        return kf

    def update_filter_matrices(self, dt, w):
        self.imm_model.filters[0].F = np.array([[1, dt, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, dt, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        self.imm_model.filters[1].F = np.array([[1, dt, 0.5*dt**2, 0, 0, 0], [0, 1, dt, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, dt, 0.5*dt**2], [0, 0, 0, 0, 1, dt], [0, 0, 0, 0, 0, 1]])
        wdt = w * dt
        c, s = np.cos(wdt), np.sin(wdt)
        sw, lhs = (s/w, (1-c)/w**2) if abs(w) >= 0.001 else (dt, 0.5*dt**2)
        self.imm_model.filters[2].F = np.array([[1, sw, lhs, 0, 0, 0], [0, c, sw, 0, 0, 0], [0, -w*s, c, 0, 0, 0],
                                               [0, 0, 0, 1, sw, lhs], [0, 0, 0, 0, c, sw], [0, 0, 0, 0, -w*s, c]])

    def state_callback(self, msg):
        raw_dt, x, y, vx, vy = msg.data
        dt = raw_dt / 1000.0 if raw_dt > 0 else 0.150
        if dt > 0.16: return
        if self.first_callback:
            self.first_callback = False
            for kf in self.imm_model.filters:
                kf.x[0], kf.x[3], kf.x[1], kf.x[4] = x, y, vx, vy
            self.imm_model.x = self.imm_model.mu @ np.array([f.x for f in self.imm_model.filters])
            return
        
        vel_mag = np.sqrt(self.imm_model.x[1]**2 + self.imm_model.x[4]**2)
        accel_mag = np.sqrt(self.imm_model.x[2]**2 + self.imm_model.x[5]**2)
        w = np.clip(accel_mag/vel_mag, -0.3, 0.3) if vel_mag >= 0.01 else 0.0
        
        self.update_filter_matrices(dt, w)
        self.imm_model.predict()
        self.imm_model.update(np.array([x, y]))
        
        self.imm_model.x[2] = np.clip(self.imm_model.x[2], -3.0, 3.0)
        self.imm_model.x[5] = np.clip(self.imm_model.x[5], -3.0, 3.0)

        pred = self.generate_prediction(steps=15, dt=dt)
        self.publish_path(pred.tolist())
        self.publish_heatmap(steps=15)
        self.prev_x, self.prev_y = x, y

    def odom_callback(self, msg : Odometry):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        current_time = self.get_clock().now()
        dt = 0.150
        if (current_time - self.last_odom_pub_time).nanoseconds/(1e9) > dt:
            if self.first_callback:
                self.first_callback = False
                for kf in self.imm_model.filters: kf.x[0], kf.x[3] = x, y
                self.imm_model.x = self.imm_model.mu @ np.array([f.x for f in self.imm_model.filters])
                return

            vel_mag = np.sqrt(self.imm_model.x[1]**2 + self.imm_model.x[4]**2)
            accel_mag = np.sqrt(self.imm_model.x[2]**2 + self.imm_model.x[5]**2)
            w = np.clip(accel_mag/vel_mag, -0.3, 0.3) if vel_mag >= 0.01 else 0.0
            self.update_filter_matrices(dt, w)
            self.imm_model.predict()
            self.imm_model.update(np.array([x, y]))

            pred = self.generate_prediction(steps=45, dt=(dt/3))
            self.publish_path(pred.tolist())
            self.publish_heatmap(steps=45)
            self.last_odom_pub_time = current_time

    def publish_heatmap(self, steps):
        """Monte Carlo Heatmap: 30 samples with jittered velocity and acceleration to show curvature"""
        marker_array = MarkerArray()
        best_idx = np.argmax(self.imm_model.mu)
        num_samples_per_model = 10 
        vel_noise = 0.15   # Velocity jitter (m/s)
        accel_noise = 0.1  # Acceleration jitter (m/s^2) - creates curves
        
        for model_idx, kf in enumerate(self.imm_model.filters):
            for sample_idx in range(num_samples_per_model):
                marker = Marker()
                marker.header.frame_id, marker.header.stamp = "map", self.get_clock().now().to_msg()
                marker.ns, marker.id = f"model_{model_idx}", (model_idx * num_samples_per_model) + sample_idx
                marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
                
                temp_state = self.imm_model.x.copy()
                # Apply noise to create a fan of curved projections
                temp_state[1] += np.random.normal(0, vel_noise)   # vx
                temp_state[2] += np.random.normal(0, accel_noise) # ax
                temp_state[4] += np.random.normal(0, vel_noise)   # vy
                temp_state[5] += np.random.normal(0, accel_noise) # ay
                
                if model_idx == best_idx:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 0.6 / num_samples_per_model
                    marker.scale.x = 0.08
                else:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 0.2 / num_samples_per_model
                    marker.scale.x = 0.03

                for _ in range(steps):
                    temp_state = np.dot(kf.F, temp_state)
                    p = Point()
                    p.x, p.y, p.z = float(temp_state[0]), float(temp_state[3]), 0.15 # Elevated path
                    marker.points.append(p)
                marker_array.markers.append(marker)
        self.heatmap_pub.publish(marker_array)

    def generate_prediction(self, steps, dt):
        best_idx = np.argmax(self.imm_model.mu)
        curr_state = self.imm_model.x.copy()
        prediction = np.zeros((steps, 2))
        for i in range(steps):
            curr_state = np.dot(self.imm_model.filters[best_idx].F, curr_state)
            prediction[i] = [curr_state[0], curr_state[3]]
        return np.array(prediction)

    def publish_path(self, points):
        path_msg = Path()
        path_msg.header.frame_id, path_msg.header.stamp = "map", self.get_clock().now().to_msg()
        for pt in points:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.orientation.w = float(pt[0]), float(pt[1]), 1.0
            path_msg.poses.append(ps)
        self.traj_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IMMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()