#include "rclcpp/rclcpp.hpp"
#include "referee_pkg/srv/hit_armor.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/polygon.hpp"
#include "std_msgs/msg/header.hpp"
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mutex>

using namespace std;
using namespace rclcpp;

class shooter_node : public rclcpp::Node
{
private:
    const double v = 1.5;
    const float g = 9.8;
    rclcpp::Service<referee_pkg::srv::HitArmor>::SharedPtr hit_armor_service_;
    rclcpp::Subscription<geometry_msgs::msg::Polygon>::SharedPtr armor_corners_sub_;
    
    // 固定的相机内参（从Gazebo模型计算）
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    // 存储最新的装甲板角点
    std::vector<cv::Point2f> latest_armor_corners_;
    std::mutex corners_mutex_;

    void handle_hit_armor(
        const std::shared_ptr<referee_pkg::srv::HitArmor::Request> request,
        const std::shared_ptr<referee_pkg::srv::HitArmor::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "收到击打请求，使用PNP计算目标位置...");

        try {
            geometry_msgs::msg::Point center = calculate_center_pnp(request->modelpoint);
            
            RCLCPP_INFO(this->get_logger(), "装甲板世界坐标: (%.3f, %.3f, %.3f)", 
                       center.x, center.y, center.z);

            // 计算欧拉角（外旋ZYX顺序）
            calculateAngle(center, response);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "计算失败: %s", e.what());
            // 返回默认值避免服务调用失败
            response->yaw = 0.0;
            response->pitch = 0.0;
            response->roll = 0.0;
        }
    }

    // 使用PNP算法计算装甲板在世界坐标系中的中心点
    geometry_msgs::msg::Point calculate_center_pnp(const std::vector<geometry_msgs::msg::Point> &model_points)
    {
        // 步骤1: 获取装甲板在图像中的2D角点坐标
        std::vector<cv::Point2f> image_points = get_armor_image_points();
        
        if (image_points.size() != 4) {
            throw std::runtime_error("未能获取到4个装甲板角点，当前数量: " + std::to_string(image_points.size()));
        }

        // 步骤2: 将ROS消息中的模型点转换为OpenCV格式
        std::vector<cv::Point3f> object_points;
        for (const auto& point : model_points) {
            object_points.push_back(cv::Point3f(point.x, point.y, point.z));
        }

        // 步骤3: 使用PNP算法求解
        cv::Mat rvec, tvec;
        solvePnP(object_points, image_points, camera_matrix_, 
                                   dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_EPNP);

        // 步骤4: 将结果转换为世界坐标系中的点
        geometry_msgs::msg::Point center;
        center.x = tvec.at<double>(0);  // X: 相机前方
        center.y = tvec.at<double>(1);  // Y: 相机左方  
        center.z = tvec.at<double>(2);  // Z: 相机上方

        return center;
    }

    // 从视觉节点获取装甲板角点像素坐标
    std::vector<cv::Point2f> get_armor_image_points()
    {
        std::lock_guard<std::mutex> lock(corners_mutex_);
        return latest_armor_corners_;
    }

    // 订阅视觉节点发布的装甲板角点
    void armor_corners_callback(const geometry_msgs::msg::Polygon::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(corners_mutex_);
        latest_armor_corners_.clear();
        
        for (const auto& point : msg->points) {
            latest_armor_corners_.push_back(cv::Point2f(point.x, point.y));
        }
        
        RCLCPP_DEBUG(this->get_logger(), "收到装甲板角点数据，数量: %zu", latest_armor_corners_.size());
    }

    // 从Gazebo相机模型计算内参
    void initialize_camera_parameters()
    {
        // 从Gazebo相机模型提取的参数
        double width = 640.0;
        double height = 480.0;
        double horizontal_fov = 1.3962634; // 弧度，约80度
        
        // 计算焦距
        double fx = width / (2 * tan(horizontal_fov / 2));
        double fy = fx;
        double cx = width / 2;
        double cy = height / 2;
        
        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1);
        
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);  // 仿真中通常无畸变
    }

public:
    shooter_node() : Node("shooter_node")
    {
        // 初始化相机参数
        initialize_camera_parameters();
        
        // 创建击打服务
        hit_armor_service_ = this->create_service<referee_pkg::srv::HitArmor>(
            "/referee/hit_arror",
            std::bind(&shooter_node::handle_hit_armor, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        // 订阅视觉节点发布的装甲板角点
        armor_corners_sub_ = this->create_subscription<geometry_msgs::msg::Polygon>(
            "/vision/armor_corners", 10,
            std::bind(&shooter_node::armor_corners_callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "击打节点已启动");
    }

    void calculateAngle(geometry_msgs::msg::Point centers, const std::shared_ptr<referee_pkg::srv::HitArmor::Response> response)
    {
        double horizontal_distance = sqrt(centers.x * centers.x + centers.y * centers.y);
        double height = centers.z;
        double delta = pow(horizontal_distance, 2) - pow(g, 2) * pow(horizontal_distance, 5) - 2 * g * height * pow(horizontal_distance, 3) / pow(v, 2);
        if (delta < 0)
        {
            RCLCPP_WARN(this->get_logger(), "击打失败，无解");
        }
        else
        {
            response->yaw = std::atan2(centers.y, centers.x);
            double optimal_pitch = calculate_optimal_pitch(horizontal_distance, height, v, g, delta);
            double z = horizontal_distance * tan(optimal_pitch);
            response->pitch = std::atan2(z, centers.y);
            response->roll = std::atan2(z, centers.x);
            RCLCPP_INFO(this->get_logger(), "计算完成: yaw=%.3f, pitch=%.3f, roll=%.3f",
                        response->yaw, response->pitch, response->roll);
        }
    }

    double calculate_optimal_pitch(double d, double h, double v, double g, double delta)
    {
        double angle1 = acosf128((-pow(h, 2) - g * h * d / v + sqrt(delta)) / pow(h, 2) + d) / 2;
        double angle2 = acosf128((-pow(h, 2) - g * h * d / v - sqrt(delta)) / pow(h, 2) + d) / 2;
        double time1 = d / v * cos(angle1);
        double time2 = d / v * cos(angle2);
        return time1 < time2 ? angle1 : angle2;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<shooter_node>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}