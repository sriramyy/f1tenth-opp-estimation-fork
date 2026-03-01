#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.h"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"
#include <chrono>

using std::placeholders::_1;

class bounding_box : public rclcpp::Node {
    public:
        bounding_box() : Node("bounding_box") {
            filtered_pc_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("/filtered_cluster", 10, std::bind(&bounding_box::bounding_box_callback, this, std::placeholders::_1));
            //tf buffer and listener are used for transforming received point cloud data from reference frame of lidar to map reference frame
            tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener = std::make_unique<tf2_ros::TransformListener>(*tf_buffer);
            state_vector_pub = this->create_publisher<std_msgs::msg::Float64MultiArray>("/state_vector", 10);
        }
        

        
    private:
        // subscribes to cluster that we have identified as the car
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pc_sub;
        // publishes opponent state vector information
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_vector_pub;
        std::unique_ptr<tf2_ros::Buffer> tf_buffer;
        std::unique_ptr<tf2_ros::TransformListener> tf_listener;
        

        // center values fo bounding box created from point cloud
        float x_curr_ctr = 0.0;
        float y_curr_ctr = 0.0;
        float z_curr_ctr = 0.0;
        float vx_curr = 0.0;
        float vy_cur = 0.0;

        // in ms
        int dt = 50; 
        int steps = 3;
        int num_of_clusters = 0;

        std::chrono::milliseconds elapsed_time_ms;
        std::chrono::high_resolution_clock::time_point last_recorded_time_ms = std::chrono::high_resolution_clock::now();

        // average values of x, y, and z across multiple clusters (z always stuck at 0 because we use simulation lidar data)
        float x_avg = 0.0;
        float y_avg = 0.0;
        float z_avg = 0.0;

        float prev_x = 0.0;
        float prev_y = 0.0;
        float prev_z = 0.0;

        float vx = 0.0;
        float vy = 0.0;

        float x_tot = 0.0;
        float y_tot = 0.0;
        float z_tot = 0.0;


        void bounding_box_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            std::size_t count = 0;

            sensor_msgs::msg::PointCloud2 point_cloud_map_frame;
            

            tf_buffer->transform<sensor_msgs::msg::PointCloud2>(*msg, point_cloud_map_frame, "map", tf2::Duration(std::chrono::seconds(1)));
            

            sensor_msgs::PointCloud2Iterator<float> iterX(point_cloud_map_frame, "x");
            sensor_msgs::PointCloud2Iterator<float> iterY(point_cloud_map_frame, "y");
            sensor_msgs::PointCloud2Iterator<float> iterZ(point_cloud_map_frame, "z");
            
            float x_ctr = 0.0;
            float y_ctr = 0.0;
            float z_ctr = 0.0;



            for(; iterX != iterX.end(); ++iterX, ++iterY, ++iterZ) {
                    x_ctr += *iterX;
                    y_ctr += *iterY;
                    z_ctr += *iterZ;
                    count++;
            }

            x_ctr /= static_cast<double>(count);
            y_ctr /= static_cast<double>(count);
            z_ctr /= static_cast<double>(count);

            x_curr_ctr = x_ctr;
            y_curr_ctr = y_ctr;
            z_curr_ctr = z_ctr;


            if(num_of_clusters <= steps)
            {
                //add centers 
                x_tot += x_curr_ctr;
                y_tot += y_curr_ctr;
                z_tot += z_curr_ctr;
                
                // gets an average of steps num of clusters
                num_of_clusters++;

            }
            else {
                // getting average of 3 clusters 
                x_avg = x_tot / steps;
                y_avg = y_tot / steps;
                z_avg = z_tot / steps;
                x_tot = 0;
                y_tot = 0;
                z_tot = 0;
                num_of_clusters = 0;
            }
            
            elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last_recorded_time_ms);
    

            if(elapsed_time_ms.count() >= dt)
            {
                vx = (x_avg - prev_x) * 1000.0 / static_cast<float>(elapsed_time_ms.count());
                vy = (y_avg - prev_y) * 1000.0 / static_cast<float>(elapsed_time_ms.count());

                std_msgs::msg::Float64MultiArray state_array;
                std_msgs::msg::MultiArrayDimension dim;
                dim.label="arr";
                dim.size = 5;
                dim.stride = 5;
                state_array.layout.dim = {dim};
            
                // Publish centers on "/state_vector" topic if elapsed time > dt, publish dt as well here [dt, x, y, vx, vy]
                state_array.data = {static_cast<float>(elapsed_time_ms.count()), x_curr_ctr, y_curr_ctr, vx, vy};
                state_vector_pub->publish(state_array);
                RCLCPP_INFO(this->get_logger(), "Centers: x_avg - %f, prev_x - %f, z - %f, vx - %f, vy - %f", x_avg, prev_x, z_curr_ctr, vx, vy);

                elapsed_time_ms = std::chrono::milliseconds(0);
                last_recorded_time_ms = std::chrono::high_resolution_clock::now();

                //sets to last seen cluster state
                prev_x = x_avg;
                prev_y = y_avg;
                prev_z = z_avg; 
            }    
        }

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<bounding_box>());
    rclcpp::shutdown();
    return 0;

}