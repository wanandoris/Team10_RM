#include "geometry_msgs/msg/polygon.hpp"
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
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace rclcpp;
using namespace cv;

class Stage1Node : public rclcpp::Node
{
private:
    // 定义结构体，以便绑定识别到的物体的信息
    struct ObjectInfo
    {
        string type;             // 类型
        vector<Point2f> corners; // 角点
        double area;             // 面积
        int id;
        Point2f center;
        float radius;
        float length;
        float width;
        double circularity;
        // 初始化
        ObjectInfo(string t, vector<Point2f> c, double a, int i,
                   Point2f cen = Point2f(0, 0), float r = 0,
                   float l = 0, float w = 0, double circ = 0)
            : type(t), corners(c), area(a), id(i), center(cen),
              radius(r), length(l), width(w), circularity(circ) {}
    };

    vector<ObjectInfo> object_info_list;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    string objectType;
    Point2f center;
    float radius;
    double circularity;
    float length, width;
    vector<Point2f> Point_V;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub_;
    rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr Target_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Polygon>::SharedPtr armor_corners_pub_;

    // ONNX Runtime 相关变量
    Ort::Env onnx_env;
    Ort::Session onnx_session{nullptr};
    bool dl_model_loaded;
    std::vector<std::string> armor_classes;
    float conf_threshold;
    float nms_threshold;
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_shape;

    void callback_camera(sensor_msgs::msg::Image::SharedPtr msg)
    {
        DetectTheObjects(msg);
    }

    ObjectInfo createObjectInfo(const string &type,
                                const vector<Point2f> &corners,
                                double area, int id,
                                const Point2f &center = Point2f(0, 0),
                                float radius = 0,
                                float length = 0,
                                float width = 0,
                                double circularity = 0)
    {
        return ObjectInfo(type, corners, area, id, center, radius, length, width, circularity);
    }

    referee_pkg::msg::Object toRosMessage(const ObjectInfo &obj)
    {
        referee_pkg::msg::Object ros_obj;
        ros_obj.target_type = obj.type;

        for (const auto &corner : obj.corners)
        {
            geometry_msgs::msg::Point point;
            point.x = corner.x;
            point.y = corner.y;
            point.z = 0.0;
            ros_obj.corners.push_back(point);
        }

        return ros_obj;
    }
    // 预处理函数，采用掩码进行预处理
    Mat preProcessImage(const Mat &img)
    {
        if (img.empty())
        {
            RCLCPP_WARN(this->get_logger(), "没有准备处理的图像");
            return Mat();
        }
        Mat hsv, mask_red1, mask_red2, mask_red, mask_blue, mask_green, mask_cyan, mask_black, mask_final;
        try
        {
            cvtColor(img, hsv, COLOR_BGR2HSV);
            // 处理红、蓝、绿、青、黑
            inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask_red1);
            inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask_red2);
            mask_red = mask_red1 | mask_red2;

            inRange(hsv, Scalar(100, 120, 70), Scalar(130, 255, 255), mask_blue);
            inRange(hsv, Scalar(40, 120, 70), Scalar(80, 255, 255), mask_green);
            inRange(hsv, Scalar(85, 120, 70), Scalar(100, 255, 255), mask_cyan);
            inRange(hsv, Scalar(0, 0, 0), Scalar(180, 255, 50), mask_black);

            mask_final = mask_red | mask_blue | mask_green | mask_cyan | mask_black;
            // 进行开闭运算
            Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
            Mat kernel_open = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

            morphologyEx(mask_final, mask_final, MORPH_OPEN, kernel_open);
            morphologyEx(mask_final, mask_final, MORPH_CLOSE, kernel_close);
            // 进行侵蚀运算
            Mat kernel_dilate = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
            dilate(mask_final, mask_final, kernel_dilate, Point(-1, -1), 1);

            return mask_final;
        }
        catch (const cv::Exception &e)
        {
            return Mat();
        }
    }

    void detectSphereAndRectangles(const Mat &raw_img)
    {
        Mat mask = preProcessImage(raw_img);
        if (mask.empty())
        {
            RCLCPP_DEBUG(this->get_logger(), "Mask is empty");
            return;
        }
        // 寻找轮廓
        vector<vector<Point>> contours; // 存储轮廓
        vector<Vec4i> hierarchy;        // findContours所用核
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);

        if (contours.empty())
        {
            RCLCPP_DEBUG(this->get_logger(), "No contours found");
            return;
        }
        // 遍历轮廓
        for (size_t i = 0; i < contours.size(); i++)
        {
            if (contours[i].empty())
                continue;

            double area = contourArea(contours[i]);

            float peri = arcLength(contours[i], true); // 计算周长
            if (peri <= 0)
                continue;

            vector<Point> approx;
            approxPolyDP(contours[i], approx, 0.02 * peri, true); // 利用周长逼近多边形

            if (approx.empty())
                continue;

            int objCor = (int)approx.size(); // 寻找角点
            double circularity = 4 * CV_PI * area / (peri * peri);

            bool object_detected = false;
            vector<Point2f> corners;
            double calculated_area = 0.0;
            Point2f center;
            float radius = 0, length = 0, width = 0;
            string objectType;
            // 圆形度检测
            if (circularity > 0.85)
            {
                objectType = "sphere";
                radius = peri / (2 * CV_PI);
                Moments m = moments(approx);
                if (m.m00 != 0)
                {
                    center.x = static_cast<float>(m.m10 / m.m00);
                    center.y = static_cast<float>(m.m01 / m.m00);
                }
                // 计算与圆相关的信息
                object_detected = true;
                corners = getSphereCorners(center, radius);
                calculated_area = CV_PI * radius * radius;

                RCLCPP_DEBUG(this->get_logger(), "Detected sphere: center(%.1f,%.1f), radius=%.1f",
                             center.x, center.y, radius);
            }
            else
            {
                // 长宽比检测
                Rect rect = boundingRect(approx);
                float aspect_ratio = (float)rect.width / rect.height;

                if (aspect_ratio >= 2.5)
                {
                    // 长方形相关检测
                    objectType = detectMovingRectangle(center, contours[i]);
                    length = rect.width;
                    width = rect.height;
                    center.x = rect.x + length / 2;
                    center.y = rect.y + width / 2;
                    object_detected = true;

                    corners = getRectangleCorners(center, length, width);
                    calculated_area = length * width;

                    RCLCPP_DEBUG(this->get_logger(), "Detected rectangle: %s, center(%.1f,%.1f), size=%.1fx%.1f",
                                 objectType.c_str(), center.x, center.y, length, width);
                }
            }

            if (object_detected && !corners.empty())
            {
                // 创建obj_info绑定物体信息
                ObjectInfo obj_info = createObjectInfo(
                    objectType, corners, calculated_area, object_info_list.size(),
                    center, radius, length, width, circularity);
                object_info_list.push_back(obj_info);
            }
        }
    }
    string detectMovingRectangle(const Point2f &current_center, const vector<Point> &contour)
    {
        static map<int, Point2f> prev_centers;
        static map<int, int> move_counters;
        static int frame_count = 0;
        // 利用哈希值跟踪轮廓
        int contour_hash = hashContour(contour);

        // 每10帧重置一次
        if (frame_count % 10 == 0)
        {
            move_counters.clear();
        }
        frame_count++;

        if (prev_centers.find(contour_hash) != prev_centers.end())
        {
            Point2f prev_center = prev_centers[contour_hash];
            float movement = norm(current_center - prev_center);

            // 如果移动距离超过阈值，增加移动计数
            if (movement > 5.0f)
            { // 像素移动阈值
                move_counters[contour_hash]++;
            }

            // 如果连续多帧都在移动，判定为移动矩形
            if (move_counters[contour_hash] >= 3)
            {
                prev_centers[contour_hash] = current_center;
                return "rect_move";
            }
        }

        prev_centers[contour_hash] = current_center;
        return "rect";
    }
    // 辅助函数：计算轮廓哈希值
    int hashContour(const vector<Point> &contour)
    {
        int hash = 0;
        for (const auto &point : contour)
        {
            hash = hash * 31 + point.x + point.y;
        }
        return abs(hash);
    }
    void detectArmor(const cv::Mat &image)
    {
        if (!dl_model_loaded)
        {
            RCLCPP_DEBUG(this->get_logger(), "ONNX模型未加载，跳过装甲板检测");
            return;
        }

        try
        {
            // 预处理图像
            cv::Mat resized_image;
            if (input_shape.size() >= 4)
            {
                cv::resize(image, resized_image, cv::Size(input_shape[3], input_shape[2]));
            }
            else
            {
                cv::resize(image, resized_image, cv::Size(640, 640));
            }

            cv::Mat float_image;
            resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

            // 将图像从HWC转换为CHW格式
            std::vector<cv::Mat> channels(3);
            cv::split(float_image, channels);

            // 准备输入数据
            std::vector<float> input_tensor_values;
            for (int c = 0; c < 3; ++c)
            {
                input_tensor_values.insert(input_tensor_values.end(),
                                           (float *)channels[c].data, (float *)channels[c].data + channels[c].total());
            }

            // 创建输入张量
            size_t input_tensor_size = input_tensor_values.size();
            std::vector<Ort::Value> input_tensors;

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

            // 使用正确的形状创建张量
            std::vector<int64_t> current_shape = input_shape;
            if (current_shape.empty())
            {
                current_shape = {1, 3, 640, 640}; // 默认形状
            }

            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_size,
                current_shape.data(), current_shape.size()));

            // 运行推理
            if (!input_names.empty() && !output_names.empty())
            {
                std::vector<Ort::Value> output_tensors = onnx_session.Run(
                    Ort::RunOptions{nullptr},
                    input_names.data(),
                    input_tensors.data(),
                    input_tensors.size(),
                    output_names.data(),
                    output_names.size());

                // 处理输出
                processYOLOOutput(output_tensors, image.size());
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "输入或输出名称为空");
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "ONNX推理失败: %s", e.what());
        }
    }

    void sortDetectedObjects()
    {
        // 先按形状排序
        auto getShapePriority = [](const string &type) -> int
        {
            if (type.find("armor") != string::npos)
                return 0;
            if (type == "rect" || type == "rect_move")
                return 1;
            if (type == "sphere")
                return 2;
            return 3;
        };
        // 利用sort函数，排序方法为先排形状，形状相同再按面积排
        sort(object_info_list.begin(), object_info_list.end(),
             [&](const ObjectInfo &a, const ObjectInfo &b)
             {
                 int priority_a = getShapePriority(a.type);
                 int priority_b = getShapePriority(b.type);

                 if (priority_a != priority_b)
                 {
                     return priority_a < priority_b;
                 }

                 return a.area > b.area;
             });
        // 重新修改ID
        for (size_t i = 0; i < object_info_list.size(); i++)
        {
            object_info_list[i].id = i;
        }
    }

    void drawObjectInfo(Mat &img, const ObjectInfo &obj)
    {
        Scalar contour_color, center_color, corner_color;
        bool draw_contour = false;
        bool draw_center = false;
        bool draw_radius_text = false;
        // 分别判断大家都要画什么
        if (obj.type == "sphere")
        {
            contour_color = Scalar(0, 255, 0);
            center_color = Scalar(0, 0, 255);
            draw_contour = true;
            draw_center = true;
            draw_radius_text = true;
        }
        else if (obj.type == "rect" || obj.type == "rect_move")
        {
            contour_color = Scalar(0, 0, 255);
            center_color = Scalar(0, 0, 0);
            draw_contour = true;
            draw_center = false;
            draw_radius_text = false;
        }
        else if (obj.type.find("armor") != string::npos)
        {
            contour_color = Scalar(0, 0, 0);
            center_color = Scalar(0, 0, 0);
            draw_contour = false;
            draw_center = false;
            draw_radius_text = false;
        }
        // 定义角点颜色
        vector<Scalar> corner_colors;
        if (obj.type == "sphere" || obj.type == "rect" || obj.type == "rect_move")
        {
            corner_colors = {
                Scalar(255, 0, 0),
                Scalar(0, 255, 0),
                Scalar(0, 255, 255),
                Scalar(255, 0, 255)};
        }
        else
        {
            corner_colors = {Scalar(0, 0, 255), Scalar(0, 0, 255), Scalar(0, 0, 255), Scalar(0, 0, 255)};
        }
        // 圆形矩形画边框
        if (draw_contour)
        {
            if (obj.type == "sphere")
            {
                circle(img, obj.center, obj.radius, contour_color, 2);
            }
            else if (obj.type == "rect" || obj.type == "rect_move")
            {
                Rect rect(obj.center.x - obj.length / 2, obj.center.y - obj.width / 2,
                          obj.length, obj.width);
                rectangle(img, rect, contour_color, 2);
            }
        }
        // 圆画中心
        if (draw_center)
        {
            circle(img, obj.center, 3, center_color, -1);
        }

        for (int j = 0; j < 4 && j < obj.corners.size(); j++)
        {
            if (obj.type.find("armor") != string::npos)
            {
                circle(img, obj.corners[j], 6, corner_colors[j], 1.5);
            }
            else
            {
                circle(img, obj.corners[j], 6, corner_colors[j], -1);
                circle(img, obj.corners[j], 6, Scalar(0, 0, 0), 2);
            }

            string point_text = to_string(j + 1);
            if (obj.type.find("armor") != string::npos)
            {
                putText(img, point_text,
                        Point(obj.corners[j].x + 10, obj.corners[j].y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
            }
            else
            {
                putText(img, point_text,
                        Point(obj.corners[j].x + 10, obj.corners[j].y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 3);
                putText(img, point_text,
                        Point(obj.corners[j].x + 10, obj.corners[j].y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.6, corner_colors[j], 2);
            }
        }

        if (draw_radius_text)
        {
            string radius_text = "R:" + to_string((int)obj.radius);
            putText(img, radius_text,
                    Point(obj.center.x - 15, obj.center.y + 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        }

        string info_text = "ID:" + to_string(obj.id);
        putText(img, info_text,
                Point(obj.center.x - 25, obj.center.y - 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
    }

    void publishDetectedObjects(sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!object_info_list.empty())
        {
            // 对应内容
            referee_pkg::msg::MultiObject msg_object;
            msg_object.header = msg->header;
            msg_object.num_objects = object_info_list.size();

            for (const auto &obj_info : object_info_list)
            {
                msg_object.objects.push_back(toRosMessage(obj_info));
            }
            // 发布
            Target_pub_->publish(msg_object);
            // 发布信息
            string sorted_info = "Published " + to_string(object_info_list.size()) + " objects: ";
            for (const auto &obj : object_info_list)
            {
                sorted_info += obj.type + "[ID:" + to_string(obj.id) + "] ";
            }
            RCLCPP_INFO(this->get_logger(), sorted_info.c_str());
        }
    }

    vector<Point2f> getSphereCorners(const Point2f &center, float radius)
    {
        vector<Point2f> corners;
        corners.push_back(Point2f(center.x - radius, center.y));
        corners.push_back(Point2f(center.x, center.y + radius));
        corners.push_back(Point2f(center.x + radius, center.y));
        corners.push_back(Point2f(center.x, center.y - radius));
        return corners;
    }

    vector<Point2f> getRectangleCorners(const Point2f &center, float length, float width)
    {
        vector<Point2f> corners;
        corners.push_back(Point2f(center.x - length / 2, center.y + width / 2));
        corners.push_back(Point2f(center.x + length / 2, center.y + width / 2));
        corners.push_back(Point2f(center.x + length / 2, center.y - width / 2));
        corners.push_back(Point2f(center.x - length / 2, center.y - width / 2));
        return corners;
    }

    vector<Point2f> getArmorCorners(const Point2f &center, float length, float width)
    {
        const float k = 0.4971;
        vector<Point2f> corners;
        corners.push_back(Point2f(center.x - length / 2, center.y - width * k / 2));
        corners.push_back(Point2f(center.x + length / 2, center.y - width * k / 2));
        corners.push_back(Point2f(center.x + length / 2, center.y + width * k / 2));
        corners.push_back(Point2f(center.x - length / 2, center.y + width * k / 2));
        return corners;
    }

    bool isValidArmorDetection(const cv::Rect &box, float confidence, int class_id)
    {
        // 检查边界框尺寸
        float aspect_ratio = static_cast<float>(box.width) / box.height;
        if (aspect_ratio < 1.5 || aspect_ratio > 4.0)
        {
            RCLCPP_DEBUG(this->get_logger(), "无效的宽高比: %.2f", aspect_ratio);
            return false;
        }

        // 检查边界框面积
        int area = box.width * box.height;
        if (area < 500 || area > 50000)
        { // 根据实际情况调整
            RCLCPP_DEBUG(this->get_logger(), "无效的面积: %d", area);
            return false;
        }

        // 检查置信度
        if (confidence < conf_threshold)
        {
            RCLCPP_DEBUG(this->get_logger(), "置信度过低: %.3f", confidence);
            return false;
        }

        return true;
    }

    bool loadONNXModel(const std::string &model_path)
    {
        try
        {
            // 检查文件是否存在
            std::ifstream file(model_path, std::ios::binary | std::ios::ate);
            if (!file.is_open())
            {
                RCLCPP_ERROR(this->get_logger(), "无法打开模型文件: %s", model_path.c_str());
                return false;
            }

            std::streamsize file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            RCLCPP_INFO(this->get_logger(), "模型文件大小: %ld bytes", file_size);

            if (file_size <= 0)
            {
                RCLCPP_ERROR(this->get_logger(), "模型文件为空或无效");
                return false;
            }

            // 初始化ONNX Runtime环境
            onnx_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ArmorDetection");

            // 设置会话选项
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // 加载ONNX模型
            std::string absolute_path = model_path;
            if (model_path[0] != '/')
            {
                // 如果是相对路径，转换为绝对路径
                char cwd[1024];
                if (getcwd(cwd, sizeof(cwd)) != nullptr)
                {
                    absolute_path = std::string(cwd) + "/" + model_path;
                }
            }

            RCLCPP_INFO(this->get_logger(), "尝试加载模型: %s", absolute_path.c_str());

            onnx_session = Ort::Session(onnx_env, absolute_path.c_str(), session_options);

            // 获取输入输出信息 - 修复名称获取问题
            size_t num_input_nodes = onnx_session.GetInputCount();
            size_t num_output_nodes = onnx_session.GetOutputCount();

            RCLCPP_INFO(this->get_logger(), "输入节点数: %zu, 输出节点数: %zu", num_input_nodes, num_output_nodes);

            // 清空之前的名称
            input_names_str.clear();
            output_names_str.clear();
            input_names.clear();
            output_names.clear();

            // 获取输入名称
            for (size_t i = 0; i < num_input_nodes; i++)
            {
                Ort::AllocatedStringPtr input_name_ptr = onnx_session.GetInputNameAllocated(i, allocator);
                std::string input_name = input_name_ptr.get();
                input_names_str.push_back(input_name);
                RCLCPP_INFO(this->get_logger(), "输入名称[%zu]: %s", i, input_name.c_str());
            }

            // 获取输出名称
            for (size_t i = 0; i < num_output_nodes; i++)
            {
                Ort::AllocatedStringPtr output_name_ptr = onnx_session.GetOutputNameAllocated(i, allocator);
                std::string output_name = output_name_ptr.get();
                output_names_str.push_back(output_name);
                RCLCPP_INFO(this->get_logger(), "输出名称[%zu]: %s", i, output_name.c_str());
            }

            // 转换为C字符串数组
            for (const auto &name : input_names_str)
            {
                input_names.push_back(name.c_str());
            }
            for (const auto &name : output_names_str)
            {
                output_names.push_back(name.c_str());
            }

            // 获取输入形状
            if (num_input_nodes > 0)
            {
                auto input_type_info = onnx_session.GetInputTypeInfo(0);
                auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                input_shape = tensor_info.GetShape();

                // 处理动态维度
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    if (input_shape[i] < 0)
                    {
                        input_shape[i] = 1;
                    }
                }
            }

            armor_classes = {"armor_red_1", "armor_red_2", "armor_red_3", "armor_red_4", "armor_red_5"};
            conf_threshold = 0.5f;
            nms_threshold = 0.4f;
            dl_model_loaded = true;

            RCLCPP_INFO(this->get_logger(), "ONNX模型加载成功");
            if (!input_shape.empty())
            {
                RCLCPP_INFO(this->get_logger(), "输入形状: [%ld, %ld, %ld, %ld]",
                            input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
            }
            return true;
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "加载ONNX模型失败: %s", e.what());
            dl_model_loaded = false;
            return false;
        }
    }

    void processYOLOOutput(const std::vector<Ort::Value> &outputs, const cv::Size &img_size)
    {
        if (outputs.empty())
            return;

        try
        {
            // 获取输出张量
            const float *output_data = outputs[0].GetTensorData<float>();
            auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

            RCLCPP_DEBUG(this->get_logger(), "输出形状: [%ld, %ld, %ld]",
                         output_shape[0], output_shape[1], output_shape[2]);

            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;

            const int num_classes = armor_classes.size();
            const int num_proposals = output_shape[2];
            const int dimensions = output_shape[1];

            // 解析YOLO输出
            for (int i = 0; i < num_proposals; ++i)
            {
                const float *detection = output_data + i * dimensions;

                // 获取类别分数
                float max_confidence = 0.0f;
                int class_id = -1;

                for (int j = 4; j < dimensions; ++j)
                {
                    if (detection[j] > max_confidence)
                    {
                        max_confidence = detection[j];
                        class_id = j - 4;
                    }
                }

                float box_confidence = detection[4];
                float total_confidence = box_confidence * max_confidence;

                if (class_id >= 0 && class_id < num_classes && total_confidence > conf_threshold)
                {
                    // 提取边界框 (归一化坐标)
                    float x_center = detection[0] * img_size.width;
                    float y_center = detection[1] * img_size.height;
                    float width = detection[2] * img_size.width;
                    float height = detection[3] * img_size.height;

                    int left = static_cast<int>(x_center - width / 2);
                    int top = static_cast<int>(y_center - height / 2);

                    // 确保边界框在图像范围内
                    left = std::max(0, left);
                    top = std::max(0, top);
                    int right = std::min(img_size.width, left + static_cast<int>(width));
                    int bottom = std::min(img_size.height, top + static_cast<int>(height));

                    cv::Rect box(left, top, right - left, bottom - top);

                    class_ids.push_back(class_id);
                    confidences.push_back(total_confidence);
                    boxes.push_back(box);
                }
            }

            // 应用非极大值抑制
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

            for (size_t i = 0; i < indices.size(); ++i)
            {
                int idx = indices[i];
                cv::Rect box = boxes[idx];
                int class_id = class_ids[idx];

                // 使用自定义的装甲板角点计算函数
                cv::Point2f center(box.x + box.width / 2, box.y + box.height / 2);
                std::vector<cv::Point2f> corners = getArmorCorners(center, box.width, box.height);

                ObjectInfo obj_info = createObjectInfo(
                    armor_classes[class_id], corners, box.width * box.height,
                    object_info_list.size(),
                    center,
                    0, box.width, box.height, 0);

                object_info_list.push_back(obj_info);
                if (isValidArmorDetection(box, confidences[i], class_id))
                {
                    // 有效的装甲板检测，发布角点
                    cv::Point2f center(box.x + box.width / 2, box.y + box.height / 2);
                    std::vector<cv::Point2f> corners = getArmorCorners(center, box.width, box.height);

                    ObjectInfo obj_info = createObjectInfo(
                        armor_classes[class_id], corners, box.width * box.height,
                        object_info_list.size(),
                        center, 0, box.width, box.height, 0);

                    object_info_list.push_back(obj_info);
                    publish_armor_corners(corners);
                }
                if (!corners.empty())
                {
                    publish_armor_corners(corners);
                }
                RCLCPP_DEBUG(this->get_logger(), "检测到装甲板: %s, 置信度: %.2f",
                             armor_classes[class_id].c_str(), confidences[idx]);
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "处理YOLO输出失败: %s", e.what());
        }
    }

    void publish_armor_corners(const std::vector<cv::Point2f> &corners)
    {
        geometry_msgs::msg::Polygon corners_msg;
        for (const auto &corner : corners)
        {
            geometry_msgs::msg::Point32 point;
            point.x = corner.x;
            point.y = corner.y;
            point.z = 0.0;
            corners_msg.points.push_back(point);
        }
        armor_corners_pub_->publish(corners_msg);
    }

public:
    Stage1Node(string name) : Node(name),
                              onnx_env(ORT_LOGGING_LEVEL_WARNING, "ArmorDetection"),
                              dl_model_loaded(false),
                              conf_threshold(0.5f), nms_threshold(0.4f)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Stage1Node");

        Image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            bind(&Stage1Node::callback_camera, this, std::placeholders::_1));

        Target_pub_ = this->create_publisher<referee_pkg::msg::MultiObject>(
            "/vision/target", 10);

        armor_corners_pub_ = this->create_publisher<geometry_msgs::msg::Polygon>(
            "/vision/armor_corners", 10);

        cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);

        // 加载ONNX模型
        string model_path = "src/player_pkg/models/best.onnx";
        if (loadONNXModel(model_path))
        {
            RCLCPP_INFO(this->get_logger(), "ONNX模型加载成功");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "ONNX模型加载失败，将仅使用传统方法");
        }

        RCLCPP_INFO(this->get_logger(), "Stage1Node initialized successfully");
    }

    void DetectTheObjects(sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            // 通过cv_bridge转化
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }

            cv::Mat raw_img = cv_ptr->image;

            if (raw_img.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Received empty image after conversion");
                return;
            }

            Mat result_image = raw_img.clone();

            object_info_list.clear();

            detectSphereAndRectangles(raw_img);

            detectArmor(raw_img);

            if (!object_info_list.empty())
            {
                sortDetectedObjects();

                for (auto &obj : object_info_list)
                {
                    drawObjectInfo(result_image, obj);
                }

                publishDetectedObjects(msg);
            }

            imshow("Detection Result", result_image);
            waitKey(1);
        }
        catch (const cv::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "OpenCV Exception in DetectTheObjects: %s", e.what());
            RCLCPP_ERROR(this->get_logger(), "Error in file: %s, line: %d", e.file.c_str(), e.line);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Exception in DetectTheObjects: %s", e.what());
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Stage1Node>("Stage1Node");
    RCLCPP_INFO(node->get_logger(), "Starting Stage1Node");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}