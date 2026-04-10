#include "../detector/include/yolov11.h"
#include <OCSort.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <queue>
#include <mutex>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::mutex queueMutex;
std::condition_variable cond;
std::queue<cv::Mat> frameQueue;
bool finished = false;

// ----------- FPS parameters-----------
double displayFps = 0.0;      // 显示帧率
double inferenceFps = 0.0;    // 推理帧率
int frameCount = 0;
int inferenceCount = 0;

// ----------- Object Detection parameters--------
cv::Size size = cv::Size{640, 640};
int topk = 100;
float score_thres = 0.25f;
float iou_thres = 0.65f;

/**
@brief Convert Vector to Matrix
*/
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        for (int j = 0; j < data[0].size(); ++j)
            matrix(i, j) = data[i][j];
    return matrix;
}


//更新卡尔曼滤波器 并绘制图像
void processFrame(cv::Mat& frame, std::vector<Object>& output, ocsort::OCSort& tracker) {
    std::vector<std::vector<float>> data;

    for (int i = 0; i < output.size(); ++i) {
        Object obj = output[i];
        cv::Rect rect = obj.rect;

        data.push_back({
            (float)rect.x,
            (float)rect.y,
            (float)(rect.x + rect.width),
            (float)(rect.y + rect.height),
            obj.prob,
            (float)obj.label
        });
    }

    if (!data.empty()) {
        // 更新卡尔曼滤波器
        std::vector<Eigen::RowVectorXf> res = tracker.update(Vector2Matrix(data));
        // 绘制矩形框
        for (auto& j : res) {
            int ID = int(j[4]);
            int class_id = int(j[5]);
            float conf = j[6];   // 置信度，范围通常为 [0,1]

            // 确保class_id在COLORS数组范围内（0-9）
            int color_index = class_id;
            if (color_index < 0) color_index = 0;
            if (color_index >= static_cast<int>(COLORS.size())) {
                color_index = color_index % COLORS.size();
            }
            const std::vector<unsigned int>& rgb = COLORS[color_index];
            cv::Scalar color(rgb[2], rgb[1], rgb[0]); // BGR order

            cv::rectangle(frame, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1),
                          color, 2);

            // 确保class_id在CLASS_NAMES数组范围内（0-9）
            int name_index = class_id;
            if (name_index < 0) name_index = 0;
            if (name_index >= static_cast<int>(CLASS_NAMES.size())) {
                name_index = 0; // 使用第一个类别作为默认
            }
            // 添加置信度显示（保留两位小数）
            std::string labelText = CLASS_NAMES[name_index] + cv::format(" ID:%d conf:%.2f", ID, conf);
            // 若希望显示百分比，可改为：cv::format(" ID:%d conf:%.0f%%", ID, conf * 100)

            cv::putText(frame, labelText, cv::Point(j[0], j[1] - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        }
    }

    // ----------- 绘制半透明背景 -----------
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(10, 10), cv::Point(300, 110), 
                  cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.6, frame, 0.4, 0, frame);

    // ----------- 绘制 FPS 信息 -----------
    // 显示帧率 (整体速度)
    cv::putText(frame, cv::format("Display FPS: %.1f", displayFps), 
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(0, 255, 255), 2);
    
    // 推理帧率 (模型处理速度)
    cv::putText(frame, cv::format("Inference FPS: %.1f", inferenceFps), 
                cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 2);
    
    // 检测到的对象数量
    cv::putText(frame, cv::format("Objects: %d", (int)data.size()), 
                cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(255, 255, 0), 2);
}

//进行目标检测 并记录推理时长与帧率
void videoODThread(const std::string& videofile, YOLOv11* yolov11, ocsort::OCSort* tracker) {
    cv::VideoCapture capture(videofile);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open video file: " << videofile << std::endl;
        finished = true;
        cond.notify_all();
        return;
    }

    std::vector<Object> output;
    cv::Mat frame;

    // 用于计算推理帧率
    auto inferenceTimer = std::chrono::high_resolution_clock::now();
    double inferenceElapsed = 0.0;

    while (true) {
        if (!capture.read(frame)) break;

        // 记录推理开始时间
        auto t_start = std::chrono::high_resolution_clock::now();

        // object detection
        yolov11->preprocessGPU(frame);
        yolov11->infer();
        output.clear();
        yolov11->postprocessGPU(output, score_thres, iou_thres, topk);
      
        // object tracking
        processFrame(frame, output, *tracker);

        // 记录推理结束时间
        auto t_end = std::chrono::high_resolution_clock::now();
        
        // 计算推理帧率
        inferenceCount++;
        inferenceElapsed += std::chrono::duration<double>(t_end - t_start).count();
        
        // 每秒更新一次推理帧率
        auto now = std::chrono::high_resolution_clock::now();
        double totalElapsed = std::chrono::duration<double>(now - inferenceTimer).count();
        
        if (totalElapsed >= 1.0) {
            inferenceFps = inferenceCount / totalElapsed;
            inferenceCount = 0;
            inferenceTimer = now;
            inferenceElapsed = 0.0;
        }

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            frameQueue.push(frame.clone());
        }
        cond.notify_one();
    }

    finished = true;
    cond.notify_all();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [save_video]" << std::endl;
        std::cerr << "  <video_path>  : path to input video file" << std::endl;
        std::cerr << "  [save_video]  : optional, 1 to save output video, 0 or omit to not save (default: 0)" << std::endl;
        std::cerr << "Example: ./yolov11_ocsort /path/to/video.mp4 1" << std::endl;
        return -1;
    }

    std::string videofile = argv[1];
    bool saveVideo = false;
    
    if (argc >= 3) {
        saveVideo = (std::stoi(argv[2]) == 1);
    }
    
    std::cout << "=====================================" << std::endl;
    std::cout << "YOLOv11 + OCSort Tracker" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Video file: " << videofile << std::endl;
    std::cout << "Save output: " << (saveVideo ? "Yes" : "No") << std::endl;

    // initiate VideoWriter
    cv::VideoWriter videoWriter;
    std::string outputPath = videofile;
    if(videofile.length() >= 4){
        outputPath = outputPath.insert(videofile.length() - 4, "_out");
    }else{
        std::cerr << "Vedio should be .avi format" << std::endl;
    }

    // cuda
    cudaSetDevice(0);

    // ---------- model---------------
    std::cout << "\nInitializing YOLOv11 model..." << std::endl;
    
    std::unique_ptr<YOLOv11> yolov11(new YOLOv11("../weights/11n_custom.engine"));
    
    yolov11->make_pipe(true);
    std::cout << "YOLOv11 model loaded successfully." << std::endl;
    
    std::cout << "Initializing OCSort tracker..." << std::endl;
    //检测阈值 最大允许丢失帧数 轨迹确认最小连续匹配次数 交并比阈值 时间差阈值 关联函数类型 惯性因子 是否使用bytetrack
    ocsort::OCSort tracker = ocsort::OCSort(0.3, 500, 1, 0.1, 1, "giou", 0.3941737016672115, true);
    std::cout << "OCSort tracker initialized successfully." << std::endl;

    if (saveVideo) {
        cv::VideoCapture tempCap(videofile);
        if (tempCap.isOpened()) {
            int frameWidth = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_WIDTH));
            int frameHeight = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double videoFps = tempCap.get(cv::CAP_PROP_FPS);
            tempCap.release();
            
            std::cout << "\nVideo properties:" << std::endl;
            std::cout << "  Resolution: " << frameWidth << "x" << frameHeight << std::endl;
            std::cout << "  FPS: " << videoFps << std::endl;
            
            // H.264 encoder
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            videoWriter.open(outputPath, fourcc, videoFps, cv::Size(frameWidth, frameHeight));
            
            if (!videoWriter.isOpened()) {
                std::cerr << "Error: Could not open video writer for output file: " << outputPath << std::endl;
                return -1;
            }
            std::cout << "Output will be saved to: " << outputPath << std::endl;
        } else {
            std::cerr << "Error: Could not open video file to get properties" << std::endl;
            return -1;
        }
    }

    std::cout << "\n=====================================" << std::endl;
    std::cout << "Starting video processing..." << std::endl;
    std::cout << "Press ESC to stop." << std::endl;
    std::cout << "=====================================" << std::endl;

    std::thread videoThread(videoODThread, videofile, yolov11.get(), &tracker);

    cv::Mat frame;
    int processedFrames = 0;
    
    // 用于计算显示帧率
    auto displayTimer = std::chrono::high_resolution_clock::now();
    
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        cond.wait(lock, [] { return !frameQueue.empty() || finished; });

        if (!frameQueue.empty()) {
            frame = frameQueue.front();
            frameQueue.pop();
            lock.unlock();

            // save frame to video
            if (saveVideo && videoWriter.isOpened()) {
                videoWriter.write(frame);
            }

            // cv::imshow("YOLOv11 + OCSORT", frame);
            processedFrames++;
            
            // 计算显示帧率
            frameCount++;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - displayTimer).count();
            
            if (elapsed >= 1.0) {
                displayFps = frameCount / elapsed;
                frameCount = 0;
                displayTimer = now;
                
                // 在控制台输出统计信息
                std::cout << "\rProcessed: " << processedFrames 
                          << " | Display FPS: " << std::fixed << std::setprecision(1) << displayFps
                          << " | Inference FPS: " << inferenceFps << std::flush;
            }
            
            if (cv::waitKey(1) == 27) {  // ESC key
                std::cout << "\n\nESC pressed, stopping..." << std::endl;
                finished = true;
                break;
            }
        } else if (finished) {
            break;
        }
    }

    videoThread.join();
    
    // release VideoWriter
    if (videoWriter.isOpened()) {
        videoWriter.release();
        std::cout << "\n=====================================" << std::endl;
        std::cout << "Video saved successfully!" << std::endl;
        std::cout << "  Output file: " << outputPath << std::endl;
        std::cout << "  Total frames: " << processedFrames << std::endl;
        std::cout << "=====================================" << std::endl;
    }
    
    std::cout << "\nProgram finished." << std::endl;
    return 0;
}
