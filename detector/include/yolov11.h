//
// Created by ubuntu on 2/8/23.
//
#ifndef DETECT_NORMAL_YOLOV11_HPP
#define DETECT_NORMAL_YOLOV11_HPP
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "common.hpp"
#include "preprocess.h"
#include "postprocess.h"
#include <fstream>
#include <algorithm>

using namespace det;

const std::vector<std::string> CLASS_NAMES = {
    "Ferry",              // 1 = Ferry (渡轮)
    "Buoy",               // 2 = Buoy (浮标)
    "Vessel/ship",        // 3 = Vessel/ship (船舶)
    "Speed boat",         // 4 = Speed boat (快艇)
    "Boat",               // 5 = Boat (小船)
    "Kayak",              // 6 = Kayak (皮划艇)
    "Sail boat",          // 7 = Sail boat (帆船)
    "Swimming person",    // 8 = Swimming person (游泳者)
    "Flying bird/plane",  // 9 = Flying bird/plane (飞鸟/飞机)
    "Other"               // 10 = Other (其他)
};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   // Ferry - 蓝色
    {217, 83, 25},   // Buoy - 橙色
    {237, 177, 32},  // Vessel/ship - 黄色
    {126, 47, 142},  // Speed boat - 紫色
    {119, 172, 48},  // Boat - 绿色
    {77, 190, 238},  // Kayak - 浅蓝色
    {162, 20, 47},   // Sail boat - 红色
    {76, 76, 76},    // Swimming person - 灰色
    {153, 153, 153}, // Flying bird/plane - 浅灰色
    {255, 0, 0}      // Other - 纯红色
};


class YOLOv11 {
public:
    explicit YOLOv11(const std::string& engine_file_path);
    ~YOLOv11();

    void                 make_pipe(bool warmup = true);
    // void                 copy_from_Mat_GPU(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& siWze);
    void                 preprocessGPU(const cv::Mat& image);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocessGPU(std::vector<Object>& objs,
        float                score_thres = 0.25f,
        float                iou_thres = 0.65f,
        int                  topk = 100);
    void                 postprocess(std::vector<Object>& objs,
        float                score_thres = 0.25f,
        float                iou_thres = 0.65f,
        int                  topk = 100);
    static void          draw_objects(const cv::Mat& image,
        cv::Mat& res,
        const std::vector<Object>& objs,
        const std::vector<std::string>& CLASS_NAMES,
        const std::vector<std::vector<unsigned int>>& COLORS);

private:
    int                  num_bindings;
    int                  num_inputs = 0;
    int                  num_outputs = 0;
    int                  dst_h = 640;
    int                  dst_w = 640;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;
    PreParam pparam;

    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream = nullptr;
    Logger                       gLogger{ nvinfer1::ILogger::Severity::kERROR };
};

#endif  // DETECT_NORMAL_YOLOV11_HPP