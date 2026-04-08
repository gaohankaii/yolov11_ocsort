#include "../include/OCSort.hpp"
#include "iomanip"
#include <utility>

namespace ocsort {
    template<typename Matrix>
    std::ostream& operator<<(std::ostream& os, const std::vector<Matrix>& v) {
        os << "{";
        for (auto it = v.begin(); it != v.end(); ++it) {
            os << "(" << *it << ")\n";
            if (it != v.end() - 1) os << ",";
        }
        os << "}\n";
        return os;
    }

    OCSort::OCSort(float det_thresh_, int max_age_, int min_hits_, float iou_threshold_, int delta_t_, std::string asso_func_, float inertia_, bool use_byte_) {
        max_age = max_age_;
        min_hits = min_hits_;
        iou_threshold = iou_threshold_;
        trackers.clear();
        frame_count = 0;
        det_thresh = det_thresh_;
        delta_t = delta_t_;
        std::unordered_map<std::string, std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)>> ASSO_FUNCS{
            {"iou", iou_batch},
            { "giou", giou_batch }};
        ;
        std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> asso_func = ASSO_FUNCS[asso_func_];
        inertia = inertia_;
        use_byte = use_byte_;
        KalmanBoxTracker::count = 0;
    }
    std::ostream& precision(std::ostream& os) {
        os << std::fixed << std::setprecision(2);
        return os;
    }
std::vector<Eigen::RowVectorXf> OCSort::update(Eigen::MatrixXf dets) {
    // ------------------------- 1. 预处理检测框 -------------------------
    frame_count += 1;
    // 提取前4列：x1, y1, x2, y2（边界框坐标）
    Eigen::Matrix<float, Eigen::Dynamic, 4> xyxys = dets.leftCols(4);
    // 第5列：置信度分数
    Eigen::Matrix<float, 1, Eigen::Dynamic> confs = dets.col(4);
    // 第6列：类别ID
    Eigen::Matrix<float, 1, Eigen::Dynamic> clss = dets.col(5);
    Eigen::MatrixXf output_results = dets;   // 原始检测结果（6列）

    // 按置信度分层：低分检测（0.1 < conf < det_thresh）和高分检测（conf > det_thresh）
    auto inds_low   = confs.array() > 0.1;
    auto inds_high  = confs.array() < det_thresh;   // det_thresh 通常为0.5或0.6
    auto inds_second = inds_low && inds_high;       // 中等置信度检测（用于二次匹配）
    Eigen::Matrix<float, Eigen::Dynamic, 6> dets_second; // 存储中等置信度检测框
    Eigen::Matrix<bool, 1, Eigen::Dynamic> remain_inds = (confs.array() > det_thresh);
    Eigen::Matrix<float, Eigen::Dynamic, 6> dets_first;  // 存储高置信度检测框

    for (int i = 0; i < output_results.rows(); i++) {
        if (true == inds_second(i)) {
            dets_second.conservativeResize(dets_second.rows() + 1, Eigen::NoChange);
            dets_second.row(dets_second.rows() - 1) = output_results.row(i);
        }
        if (true == remain_inds(i)) {
            dets_first.conservativeResize(dets_first.rows() + 1, Eigen::NoChange);
            dets_first.row(dets_first.rows() - 1) = output_results.row(i);
        }
    }

    // ------------------------- 2. 卡尔曼滤波预测 -------------------------
    // trks 矩阵：每个跟踪器的预测框 (x1,y1,x2,y2) + 占位符（速度标记）
    Eigen::MatrixXf trks = Eigen::MatrixXf::Zero(trackers.size(), 5);
    std::vector<int> to_del;   // 待删除跟踪器索引（未使用，可能为冗余）
    std::vector<Eigen::RowVectorXf> ret; // 最终输出结果

    // 对每个现有跟踪器调用 predict()，更新卡尔曼滤波的状态
    for (int i = 0; i < trks.rows(); i++) {
        Eigen::RowVectorXf pos = trackers[i].predict();
        trks.row(i) << pos(0), pos(1), pos(2), pos(3), 0;
    }

    // 收集跟踪器的运动信息：速度、上一次观测框、基于观测次数和年龄计算的加权观测值
    Eigen::MatrixXf velocities = Eigen::MatrixXf::Zero(trackers.size(), 2);
    Eigen::MatrixXf last_boxes = Eigen::MatrixXf::Zero(trackers.size(), 5);
    Eigen::MatrixXf k_observations = Eigen::MatrixXf::Zero(trackers.size(), 5);
    for (int i = 0; i < trackers.size(); i++) {
        velocities.row(i) = trackers[i].velocity;
        last_boxes.row(i) = trackers[i].last_observation;
        k_observations.row(i) = k_previous_obs(trackers[i].observations, trackers[i].age, delta_t);
    }

    // ------------------------- 3. 第一次关联（高置信度检测 vs 跟踪器） -------------------------
    // 使用增强的 IoU 匹配（融合速度、观测、惯性）
    std::vector<Eigen::Matrix<int, 1, 2>> matched;
    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_trks;
    auto result = associate(dets_first, trks, iou_threshold, velocities, k_observations, inertia);
    matched = std::get<0>(result);
    unmatched_dets = std::get<1>(result);
    unmatched_trks = std::get<2>(result);

    // 对匹配上的跟踪器执行卡尔曼滤波更新（使用高置信度检测框）
    for (auto m : matched) {
        Eigen::Matrix<float, 5, 1> tmp_bbox;
        tmp_bbox = dets_first.block<1, 5>(m(0), 0); // 取前5列（x1,y1,x2,y2,conf）
        trackers[m(1)].update(&(tmp_bbox), dets_first(m(0), 5));
    }

    // ------------------------- 4. Byte 二次匹配（中等置信度检测 vs 未匹配的跟踪器） -------------------------
    if (true == use_byte && dets_second.rows() > 0 && unmatched_trks.size() > 0) {
        // 提取未匹配跟踪器的预测框
        Eigen::MatrixXf u_trks(unmatched_trks.size(), trks.cols());
        int index_for_u_trks = 0;
        for (auto i : unmatched_trks) {
            u_trks.row(index_for_u_trks++) = trks.row(i);
        }

        // 计算中等置信度检测框与这些跟踪器的 GIoU 矩阵
        Eigen::MatrixXf iou_left = giou_batch(dets_second, u_trks);
        if (iou_left.maxCoeff() > iou_threshold) {
            // 转换为代价矩阵（负 GIoU），执行线性分配（LAPJV）
            std::vector<std::vector<float>> iou_matrix(iou_left.rows(), std::vector<float>(iou_left.cols()));
            for (int i = 0; i < iou_left.rows(); i++) {
                for (int j = 0; j < iou_left.cols(); j++) {
                    iou_matrix[i][j] = -iou_left(i, j);
                }
            }
            std::vector<int> rowsol, colsol;
            float MIN_cost = execLapjv(iou_matrix, rowsol, colsol, true, 0.01, true);

            // 构建匹配对
            std::vector<std::vector<int>> matched_indices;
            for (int i = 0; i < rowsol.size(); i++) {
                if (rowsol.at(i) >= 0) {
                    matched_indices.push_back({ colsol.at(rowsol.at(i)), rowsol.at(i) });
                }
            }

            // 对通过 IoU 阈值筛选的匹配执行更新
            std::vector<int> to_remove_trk_indices;
            for (auto m : matched_indices) {
                int det_ind = m[0];
                int trk_ind = unmatched_trks[m[1]];
                if (iou_left(m[0], m[1]) < iou_threshold) continue;

                Eigen::Matrix<float, 5, 1> tmp_box;
                tmp_box = dets_second.block<1, 5>(det_ind, 0);
                trackers[trk_ind].update(&tmp_box, dets_second(det_ind, 5));
                to_remove_trk_indices.push_back(trk_ind);
            }
            // 从未匹配跟踪器列表中移除已成功二次匹配的跟踪器
            std::vector<int> tmp_res1(unmatched_trks.size());
            sort(unmatched_trks.begin(), unmatched_trks.end());
            sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());
            auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
                                       to_remove_trk_indices.begin(), to_remove_trk_indices.end(),
                                       tmp_res1.begin());
            tmp_res1.resize(end1 - tmp_res1.begin());
            unmatched_trks = tmp_res1;
        }
    }

    // ------------------------- 5. 再次匹配（针对第一次关联后剩余的未匹配检测和跟踪器） -------------------------
    // 使用“上一次观测框”作为预测框，再次进行 IoU 匹配（对遮挡或短暂消失的目标更鲁棒）
    if (unmatched_dets.size() > 0 && unmatched_trks.size() > 0) {
        Eigen::MatrixXf left_dets(unmatched_dets.size(), 6);
        int inx_for_dets = 0;
        for (auto i : unmatched_dets) {
            left_dets.row(inx_for_dets++) = dets_first.row(i);
        }
        Eigen::MatrixXf left_trks(unmatched_trks.size(), last_boxes.cols());
        int indx_for_trk = 0;
        for (auto i : unmatched_trks) {
            left_trks.row(indx_for_trk++) = last_boxes.row(i);
        }

        Eigen::MatrixXf iou_left = giou_batch(left_dets, left_trks);
        if (iou_left.maxCoeff() > iou_threshold) {
            // 构建代价矩阵并求解线性分配
            std::vector<std::vector<float>> iou_matrix(iou_left.rows(), std::vector<float>(iou_left.cols()));
            for (int i = 0; i < iou_left.rows(); i++) {
                for (int j = 0; j < iou_left.cols(); j++) {
                    iou_matrix[i][j] = -iou_left(i, j);
                }
            }
            std::vector<int> rowsol, colsol;
            float MIN_cost = execLapjv(iou_matrix, rowsol, colsol, true, 0.01, true);
            std::vector<std::vector<int>> rematched_indices;
            for (int i = 0; i < rowsol.size(); i++) {
                if (rowsol.at(i) >= 0) {
                    rematched_indices.push_back({ colsol.at(rowsol.at(i)), rowsol.at(i) });
                }
            }

            // 执行更新并从未匹配列表中移除成功匹配项
            std::vector<int> to_remove_det_indices;
            std::vector<int> to_remove_trk_indices;
            for (auto i : rematched_indices) {
                int det_ind = unmatched_dets[i.at(0)];
                int trk_ind = unmatched_trks[i.at(1)];
                if (iou_left(i.at(0), i.at(1)) < iou_threshold) continue;
                Eigen::Matrix<float, 5, 1> tmp_bbox;
                tmp_bbox = dets_first.block<1, 5>(det_ind, 0);
                trackers.at(trk_ind).update(&tmp_bbox, dets_first(det_ind, 5));
                to_remove_det_indices.push_back(det_ind);
                to_remove_trk_indices.push_back(trk_ind);
            }
            // 更新 unmatched_dets
            std::vector<int> tmp_res(unmatched_dets.size());
            sort(unmatched_dets.begin(), unmatched_dets.end());
            sort(to_remove_det_indices.begin(), to_remove_det_indices.end());
            auto end = set_difference(unmatched_dets.begin(), unmatched_dets.end(),
                                      to_remove_det_indices.begin(), to_remove_det_indices.end(),
                                      tmp_res.begin());
            tmp_res.resize(end - tmp_res.begin());
            unmatched_dets = tmp_res;
            // 更新 unmatched_trks
            std::vector<int> tmp_res1(unmatched_trks.size());
            sort(unmatched_trks.begin(), unmatched_trks.end());
            sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());
            auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
                                       to_remove_trk_indices.begin(), to_remove_trk_indices.end(),
                                       tmp_res1.begin());
            tmp_res1.resize(end1 - tmp_res1.begin());
            unmatched_trks = tmp_res1;
        }
    }

    // ------------------------- 6. 对最终未匹配的跟踪器进行无检测更新 -------------------------
    for (auto m : unmatched_trks) {
        trackers.at(m).update(nullptr, 0);   // 没有对应检测，仅预测，不更新观测
    }

    // ------------------------- 7. 为未匹配的高分检测框创建新跟踪器 -------------------------
    for (int i : unmatched_dets) {
        Eigen::RowVectorXf tmp_bbox = dets_first.block(i, 0, 1, 5);
        int cls_ = int(dets(i, 5));
        KalmanBoxTracker trk = KalmanBoxTracker(tmp_bbox, cls_, delta_t);
        trackers.push_back(trk);
    }

    // ------------------------- 8. 生成最终输出，并移除过时跟踪器 -------------------------
    for (int i = trackers.size() - 1; i >= 0; i--) {
        Eigen::Matrix<float, 1, 4> d;
        int last_observation_sum = trackers.at(i).last_observation.sum();
        if (last_observation_sum < 0) {
            // 如果从未更新过，使用卡尔曼滤波的状态
            d = trackers.at(i).get_state();
        } else {
            // 否则使用上一次观测框（更可靠）
            d = trackers.at(i).last_observation.block(0, 0, 1, 4);
        }
        // 满足最小命中次数或处于初始帧才输出
        if (trackers.at(i).time_since_update < 1 && 
            ((trackers.at(i).hit_streak >= min_hits) || (frame_count <= min_hits))) {
            Eigen::RowVectorXf tracking_res(7);
            tracking_res << d(0), d(1), d(2), d(3), 
                            trackers.at(i).id + 1, 
                            trackers.at(i).cls, 
                            trackers.at(i).conf;
            ret.push_back(tracking_res);
        }
        // 移除超过最大存活帧数（max_age）的跟踪器
        if (trackers.at(i).time_since_update > max_age) {
            trackers.erase(trackers.begin() + i);
        }
    }
    return ret;
}    
}// namespace ocsort