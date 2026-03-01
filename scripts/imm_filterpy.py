#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator
from std_msgs.msg import Float64MultiArray, String
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Pose
from rclpy.time import Time


class IMMNode(Node):
    def __init__(self):
        super().__init__('imm_predictor')
        
        # Initial dt - will be updated dynamically in callback based on rate of received messages
        self.dt = 0.05
        self.prev_deg = 0.00
        self.prev_w = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.first_callback = True

        # Create the kalman filters
        # State vector: [x, vx, ax, y, vy, ay]
        kf_cv = self.create_kf_cv(self.dt)
        kf_ca = self.create_kf_ca(self.dt)
        kf_ct = self.create_kf_ct(self.dt, w=0.5)
        
        # imm filters and initial probabilities of model
        filters = [kf_cv, kf_ca, kf_ct]
        mu = [0.33, 0.33, 0.34]

        # Transition matrix between models
        trans = np.array([[0.98, 0.01, 0.01],   
                          [0.01, 0.98, 0.01], 
                          [0.01, 0.01, 0.98]])
        
    
        self.imm_model = IMMEstimator(filters, mu, trans)
        self.testing = False
    
        if not self.testing:
            self.state_sub = self.create_subscription(
                Float64MultiArray, '/state_vector', self.state_callback, 10)
        else:
            self.odom_sub = self.create_subscription(
                Odometry, '/opp_racecar/odom', self.odom_callback, 10)
        self.traj_pub = self.create_publisher(Path, '/imm_path', 10)
        self.wait_count = 0
        self.chosen_filter_pub = self.create_publisher(String, '/chosen_filter', 10)

        self.last_publish_time = self.get_clock().now()
        self.publish_interval = 0.005
        
    def create_kf_cv(self, dt):
        kf = KalmanFilter(dim_x=6, dim_z=2) # Observe x, y
        # kf.F = np.array([
        #         [1, dt, 0, 0, 0, 0],
        #         [0, 1, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0],
        #         [0, 0, 1, dt, 0, 0],
        #         [0, 0, 0, 1, 0, 0], 
        #         [0, 0, 0, 0, 0, 0] 
        #     ])
        kf.F = np.array([ # change to use vy instead of acceleraton
            [1, dt, 0,  0,  0,  0],
            [0,  1, 0,  0,  0,  0], 
            [0,  0, 1,  0,  0,  0], 
            [0,  0, 0,  1, dt,  0],
            [0,  0, 0,  0,  1,  0],  
            [0,  0, 0,  0,  0,  1],  
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ])
        
        kf.R = np.eye(2) * 0.05

        q_pos = 0.01
        q_vel = 0.2
        q_accel = 1.0

        kf.Q = np.diag([
            q_pos, q_vel, q_accel,
            q_pos, q_vel, q_accel
        ])

        kf.P = np.eye(6) * 1.0
        kf.x = np.zeros(6)
        return kf

    def create_kf_ca(self, dt):
        """Creates constant-acceleration Kalman Filter """
        kf = KalmanFilter(dim_x=6, dim_z=2)
        kf.F = np.array([
            [1, dt, 0.5 * dt**2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt**2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        # kf.R = np.eye(4) * 0.10
        # kf.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.1, 0.1])
        # kf.P *= 1.0
        # kf.x = np.zeros(6)
        kf.R = np.eye(2) * 0.05

        q_pos = 0.01
        q_vel = 0.3
        q_accel = 1.0

        kf.Q = np.diag([
            q_pos, q_vel, q_accel,
            q_pos, q_vel, q_accel
        ])

        kf.P = np.eye(6) * 2.0
        kf.x = np.zeros(6)
        
        return kf

    def create_kf_ct(self, dt, w):
        """Creates constant-turning Kalman filter with a timestep and an initial value for angular velocity (updated dynamically based on vx and vy)"""
        kf = KalmanFilter(dim_x=6, dim_z=2)
        c, s = np.cos(w*dt), np.sin(w*dt)
        if w == 0:
            w = 0.01

        kf.F = np.array([
            [1, s/w, (1 - c)/(w**2), 0, 0, 0],
            [0, c,   s/w, 0, 0, 0],
            [0, -1 * w * s, c, 0, 0, 0],
            [0, 0, 0, 1, s/w, (1 - c)/(w**2)],
            [0, 0, 0, 0, c, s/w],
            [0, 0, 0, 0, -1 * w * s, c]
        ])


        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        # kf.R = np.eye(4) * 0.10
        # kf.Q = np.eye(6) * 0.001
        # kf.P *= 1.0

        q_pos = 0.01
        q_vel = 0.5
        q_accel = 1.0

        kf.Q = np.diag([
            q_pos, q_vel, q_accel,
            q_pos, q_vel, q_accel
        ])

        kf.R = np.eye(2) * 0.05
        kf.P = np.eye(6) * 1.0
        kf.x = np.zeros(6)
        return kf

    def update_filter_matrices(self, dt, w):
        """Update state transition matrix F accordingly for each model"""
        # CV Model
        self.imm_model.filters[0].F = np.array([
            [1, dt, 0,  0,  0,  0],
            [0,  1, 0,  0,  0,  0],
            [0,  0, 1,  0,  0,  0],
            [0,  0, 0,  1, dt,  0],
            [0,  0, 0,  0,  1,  0],
            [0,  0, 0,  0,  0,  1],
        ])
        
        
        # CA Model
        self.imm_model.filters[1].F = np.array(
        [
            [1, dt, 0.5 * dt**2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt**2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
        


        # deg = np.degrees(np.arctan2(vy, vx))
        # w = np.radians((deg - self.prev_deg)/dt)
        # w = np.clip(w, -0.3, 0.3)
        # w = 0.35 * self.prev_w + 0.65 * w
        # w = np.clip(w, -0.3, 0.3)
        # self.prev_deg = deg
        # self.prev_w = w
        
        # if deg > 90:
        #     w = 0.5
        # else:
        #     w = -0.5

        # 

        wdt = w * dt
        c = np.cos(wdt)
        s = np.sin(wdt)

        if abs(w) < 0.001:
            # if w is too small, use CV matrix
            # small angle approximation
            sw = dt 
            lhs = 1/2 * dt^2
        else: 
            sw = s/w
            lhs = (1 - c) / (w**2)

        f_ct = np.array([
            [1, sw, lhs, 0,    0,      0],
            [0, c,   sw, 0,    0,      0],
            [0, -w*s, c, 0,    0,      0],
            [0, 0,    0, 1,    sw,    lhs],
            [0, 0,    0, 0,    c,      sw],
            [0, 0,    0, 0,    -w*s,   c]
            ])
        self.imm_model.filters[2].F = f_ct

    def state_callback(self, msg):
        # msg: [dt_ms, x, y, vx, vy]
        raw_dt, x, y, vx, vy = msg.data
        dt = raw_dt / 1000.0 if raw_dt > 0 else 0.150
        # handle large lapses in data
        if dt > 0.16:
         return

        if self.first_callback:
            self.first_callback = False
            for kf in self.imm_model.filters:
                kf.x[0] = x
                kf.x[3] = y
                kf.x[1] = vx
                kf.x[4] = vy
            self.imm_model.x = self.imm_model.mu @ [f.x for f in self.imm_model.filters]
            return

        cross_product = np.cross(np.array([self.imm_model.x[1], self.imm_model.x[4], 0]), np.array([self.imm_model.x[2], self.imm_model.x[5], 0]))

        vel_mag = np.sqrt(self.imm_model.x[1]**2 + self.imm_model.x[4]**2)
        accel_mag = np.sqrt(self.imm_model.x[2]**2 + self.imm_model.x[5]**2)

        # state vector is (x, vx, ax, y, vy, ay)
        if vel_mag < 0.01:
            w = 0.0
        else:
            w = np.clip(accel_mag/vel_mag, -0.3, 0.3)
        
        w *= np.sign(cross_product[2])
        
            
        # dist_from_prev_squared = (x - self.prev_x)**2 + (y - self.prev_y)**2
        # if not self.first_callback and dist_from_prev_squared > 3:
        #     self.wait_count += 1
        #     if self.wait_count <= 10:
        #         return 
        #     else:
        #         self.wait_count = 0
                
            
        self.update_filter_matrices(dt, w)
        
        z = np.array([x, y])
        self.imm_model.predict()
        self.imm_model.update(z)

        # [x, vx, ax, y, vy, ay]
        # Clamp acceleration to reasonable bounds just in case the taj gen goes off map
        self.imm_model.x[2] = np.clip(self.imm_model.x[2], -3.0, 3.0)  # ax
        self.imm_model.x[5] = np.clip(self.imm_model.x[5], -3.0, 3.0)  # ay
        
        # Also clamp velocity if needed
        self.imm_model.x[1] = np.clip(self.imm_model.x[1], -10.0, 10.0)  # vx
        self.imm_model.x[4] = np.clip(self.imm_model.x[4], -10.0, 10.0)  # vy


        # reducing prediction steps to less lag 
        pred = self.generate_prediction(steps=15, dt=(dt/10))
        min_x = np.min(pred[:,0])
        max_x = np.max(pred[:,0])
        min_y = np.min(pred[:,1])
        max_y = np.max(pred[:,1])
        # distance_squared = (max_x - min_x)*(max_x - min_x) + (max_y - min_y)*(max_y-min_y)

        # filter out trajectories that are too long / go off the map (likely formed due to noisy data or incorrect identification of opponent cluster)
        # if distance_squared <= 20:
            #publishing less frequently 
            # current_time = self.get_clock().now()
            # time_diff = (current_time - self.last_publish_time).nanoseconds / 1e9
            # Uncomment if want to test other filters besides CA and CA is dying 
            # if time_diff >= self.publish_interval:
            # self.publish_path(pred.tolist())
            # self.last_publish_time = current_time

        self.publish_path(pred.tolist())
        self.get_logger().info("Published!")
        self.prev_x = x
        self.prev_y = y
        self.wait_count = 0

    def odom_callback(self, msg : Odometry):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
    
    def generate_prediction(self, steps, dt):
        """Generate predicted trajectory by forward-propagation of the best model"""
        curr_state = self.imm_model.x.copy()
        best_idx = np.argmax(self.imm_model.mu)
        filter_names = ['cv', 'ca', 'ct']
        chosen_filter = filter_names[best_idx]
        print(chosen_filter)
        filter_info = String()
        filter_info.data = chosen_filter
        self.chosen_filter_pub.publish(filter_info)
        F_best = self.imm_model.filters[best_idx].F
        
        prediction = np.zeros((steps, 2))
        for i in range(steps):
            curr_state = np.dot(F_best, curr_state)
            prediction[i] = [curr_state[0], curr_state[3]]
            
        return np.array(prediction)

    def publish_path(self, points):
        """Publish path message to indicate opponent trajectory"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for pt in points:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(pt[0])
            ps.pose.position.y = float(pt[1])
            ps.pose.orientation.w = 1.0
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
