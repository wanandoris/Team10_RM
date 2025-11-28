#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <thread>
#include <atomic>

using namespace std;
using namespace rclcpp;
using namespace cv;
using sensor_msgs::msg::Image;

class TrainingNode : public rclcpp::Node
{
private:
    Subscription<Image>::SharedPtr image_sub_;
    cv_bridge::CvImagePtr cv_ptr_;
    
    // 训练数据保存路径
    string output_dir_;
    string images_dir_;
    string labels_dir_;
    
    int image_counter_;
    
    // 类别信息
    string current_class_;
    vector<string> class_names_;
    int current_class_index_;
    
    // 自动识别相关
    std::atomic<bool> auto_detection_enabled_;
    std::thread console_thread_;
    bool running_;
    
    // 帧计数器
    int frame_counter_;
    int save_interval_;

public:
    TrainingNode() : Node("training_node"), image_counter_(0), 
                    auto_detection_enabled_(false), running_(true),
                    frame_counter_(0), save_interval_(5)
    {
        // 初始化参数
        output_dir_ = "training_data";
        images_dir_ = output_dir_ + "/images";
        labels_dir_ = output_dir_ + "/labels";
        
        // 创建输出目录
        filesystem::create_directories(images_dir_);
        filesystem::create_directories(labels_dir_);
        
        // 装甲板类别
        class_names_ = {"armor_red_1", "armor_red_2", "armor_red_3", "armor_red_4", "armor_red_5"};
        current_class_index_ = 0;
        current_class_ = class_names_[current_class_index_];
        
        // 订阅摄像头话题
        image_sub_ = this->create_subscription<Image>(
            "/camera/image_raw", 10,
            bind(&TrainingNode::image_callback, this, placeholders::_1));
        
        // 启动控制台线程
        console_thread_ = std::thread(&TrainingNode::console_interface, this);
        
        RCLCPP_INFO(this->get_logger(), "训练节点已启动");
    
        print_instructions();
    }

    ~TrainingNode()
    {
        running_ = false;
        if (console_thread_.joinable()) {
            console_thread_.join();
        }
    }

    void print_instructions()
    {
        cout << "  auto: 开启/关闭自动识别模式" << endl;
        cout << "  quit: 退出程序" << endl;
        cout << "\n当前类别: " << current_class_ << endl;
        cout << "自动识别: " << (auto_detection_enabled_ ? "开启" : "关闭") << endl;
        cout << "自动保存间隔: " << save_interval_ << " 帧" << endl;
    }

    void console_interface()
    {
        while (running_ && rclcpp::ok()) {
            cout << "\n训练节点> ";
            string command;
            getline(cin, command);
            
            if (command.empty()) continue;
            
            // 转换为小写
            transform(command.begin(), command.end(), command.begin(), ::tolower);
            
            if (command == "quit") {
                RCLCPP_INFO(this->get_logger(), "退出训练节点");
                rclcpp::shutdown();
                running_ = false;
                break;
            }
            else if (command == "auto") {
                // 切换自动识别模式
                auto_detection_enabled_ = !auto_detection_enabled_;
                cout << "自动识别模式: " << (auto_detection_enabled_ ? "开启" : "关闭") << endl;
                RCLCPP_INFO(this->get_logger(), "自动识别模式: %s", 
                           auto_detection_enabled_ ? "开启" : "关闭");
            }
            else {
                cout << "未知命令: " << command << endl;
            }
        }
    }

    void image_callback(const Image::SharedPtr msg)
    {
        try
        {
            // 转换图像消息为OpenCV格式
            cv_ptr_ = cv_bridge::toCvCopy(msg, "bgr8");
            Mat image = cv_ptr_->image.clone();
            
            if (image.empty())
            {
                RCLCPP_WARN(this->get_logger(), "收到空图像");
                return;
            }
            
            // 帧计数器递增
            frame_counter_++;
            
            // 处理自动识别
            if (auto_detection_enabled_) {
                process_simple_detection(image);
            }
            
            // 处理用户输入
            handle_user_input(image);
            
            // 显示图像
            display_image(image);
            
        }
        catch (const cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "图像转换错误: %s", e.what());
        }
    }

    void process_simple_detection(cv::Mat& image)
    {
        cv::Mat gray, edges;
        
        // 转换为灰度图
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        // 使用简单的阈值分割
        cv::Mat binary;
        cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY_INV);
        
        // 形态学操作，连接相邻区域
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
        
        // 查找轮廓
        vector<vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        vector<cv::Rect> detected_rects;
        
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            
            // 面积筛选
            if (area > 100 && area < 10000) {
                
                // 获取边界矩形
                cv::Rect rect = cv::boundingRect(contour);
                float aspect_ratio = static_cast<float>(rect.width) / rect.height;
                
                // 简单的宽高比筛选
                if (aspect_ratio > 0.5f && aspect_ratio < 3.0f) {
                    detected_rects.push_back(rect);
                    
                    // 绘制边界框
                    cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
                    
                    // 显示类别标签
                    string label = current_class_ + " (auto)";
                    cv::putText(image, label, cv::Point(rect.x, rect.y - 10),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }
            }
        }
        
        // 每隔指定帧数自动保存检测到的目标
        if (auto_detection_enabled_ && frame_counter_ % save_interval_ == 0 && !detected_rects.empty()) {
            save_auto_detections(image, detected_rects);
        }
    }

    void save_auto_detections(const cv::Mat& image, const vector<cv::Rect>& rects)
    {
        // 保存图像
        string image_filename = "auto_armor_" + to_string(image_counter_) + ".jpg";
        string image_path = images_dir_ + "/" + image_filename;
        
        if (!imwrite(image_path, image))
        {
            RCLCPP_ERROR(this->get_logger(), "保存自动检测图像失败: %s", image_path.c_str());
            return;
        }
        
        // 保存YOLO格式标注
        string label_filename = "auto_armor_" + to_string(image_counter_) + ".txt";
        string label_path = labels_dir_ + "/" + label_filename;
        
        ofstream label_file(label_path);
        if (label_file.is_open())
        {
            // 保存所有检测到的矩形
            for (const auto& rect : rects) {
                // YOLO格式: class_id x_center y_center width height (归一化)
                float x_center = (rect.x + rect.width / 2.0) / image.cols;
                float y_center = (rect.y + rect.height / 2.0) / image.rows;
                float width = rect.width / (float)image.cols;
                float height = rect.height / (float)image.rows;
                
                label_file << current_class_index_ << " " 
                          << x_center << " " << y_center << " " 
                          << width << " " << height << endl;
            }
            
            label_file.close();
            
            RCLCPP_INFO(this->get_logger(), "自动保存第 %d 帧的 %zu 个检测目标 (类别: %s)", 
                       frame_counter_, rects.size(), current_class_.c_str());
            image_counter_++;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "无法打开自动检测标注文件: %s", label_path.c_str());
        }
    }

    void handle_user_input(cv::Mat& image)
    {
        int key = waitKey(1) & 0xFF;
        
        switch (key)
        {
            case '1': case '2': case '3': case '4': case '5': 
                if (key - '1' < static_cast<int>(class_names_.size())) {
                    current_class_index_ = key - '1';
                    current_class_ = class_names_[current_class_index_];
                    RCLCPP_INFO(this->get_logger(), "切换到类别: %s", current_class_.c_str());
                    cout << "切换到类别: " << current_class_ << endl;
                }
                break;
        }
    }

    void display_image(cv::Mat& image)
    {
        cv::Mat display_image = image.clone();
        
        // 显示当前类别信息
        string class_text = "Class: " + current_class_;
        putText(display_image, class_text, cv::Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        // 显示图像计数
        string count_text = "Saved: " + to_string(image_counter_);
        putText(display_image, count_text, cv::Point(10, 70), 
                FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        imshow("Training Data Collection", display_image);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = make_shared<TrainingNode>();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    destroyAllWindows();
    return 0;
}