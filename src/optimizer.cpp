#include "cfsd/optimizer.hpp"

namespace cfsd {

Optimizer::Optimizer(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<FeatureTracker>& pFeatureTracker, const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose)
    : _pMap(pMap), _pFeatureTracker(pFeatureTracker), _pImuPreintegrator(pImuPreintegrator), _pCameraModel(pCameraModel), _verbose(verbose), 
      _delta_bg(Eigen::Vector3d::Zero()), _delta_ba(Eigen::Vector3d::Zero()) {}

void Optimizer::motionOnlyBA(const cv::Mat& img) {
    double delta_pose[WINDOWSIZE][6];
    double delta_v_dbga[WINDOWSIZE][9];
    for (int i = 0; i < WINDOWSIZE; i++) {
        for (int j = 0; j < 6; j++)
            delta_pose[i][j] = 0;
        for (int j = 0; j < 9; j++)
            delta_v_dbga[i][j] = 0;
    }

    int actualSize = (_pMap->_R.size() < WINDOWSIZE) ? _pMap->_R.size() : WINDOWSIZE;
    
    // Build the problem.
    ceres::Problem problem;

    // Set up imu cost function.
    int numImuConstraint = _pMap->_imuConstraint.size();
    for (int j = 1; j < actualSize; j++) {
        ceres::CostFunction* preintegrationCost = new ImuCostFunction(_pMap, actualSize - j);
        problem.AddResidualBlock(preintegrationCost, nullptr, delta_pose[j-1], delta_v_dbga[j-1], delta_pose[j], delta_v_dbga[j]);
    }

    // // Current frame ID.
    // int curFrameID =  _pFeatureTracker->_features[_pFeatureTracker->_matchedFeatureIDs[0]]->seenByFrames.back(); 

    // cv::Mat img0 = img.clone();
    // // Set up reprojection cost function (a.k.a. residuals).
    // for (int i = 0; i < _pFeatureTracker->_matchedFeatureIDs.size(); i++) {
    //     cfsd::Ptr<Feature> f = _pFeatureTracker->_features[_pFeatureTracker->_matchedFeatureIDs[i]];
    //     for (int j = 0; j < f->seenByFrames.size(); j++) {
    //         // Check if the frame is within the local window.
    //         int idx = f->seenByFrames[j] - curFrameID + actualSize - 1;
    //         if (idx >= 0) {
    //             // Reproject to left image.
    //             // double landmark[3] = {f->position(0), f->position(1), f->position(2)};
    //             // problem.AddParameterBlock(landmark, 3);
    //             // Keep landmark constant, perform motion-only optimization.
    //             // problem.SetParameterBlockConstant(landmark);
    //             ceres::CostFunction* reprojectCost = new ImageCostFunction(_pCameraModel, f->position, f->pixelsL[j]);
    //             problem.AddResidualBlock(reprojectCost, nullptr, delta_pose[idx]);
    //         }

    //         // Show landmarks in image.
    //         if (f->seenByFrames[j] == curFrameID) {
    //             Eigen::Vector3d r_j(_pose[actualSize-1][0], _pose[actualSize-1][1], _pose[actualSize-1][2]);
    //             Eigen::Vector3d p_j(_pose[actualSize-1][3], _pose[actualSize-1][4], _pose[actualSize-1][5]);
    //             // T_WB = (r_j, p_j), is transformation from body to world frame.
    //             // T_BW = T_WB.inverse()
    //             Sophus::SE3d T_BW = Sophus::SE3d(Sophus::SO3d::exp(r_j), p_j).inverse();
    //             Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * T_BW * f->position;
    //             Eigen::Matrix<double,4,1> point_homo;
    //             point_homo << point_wrt_cam(0), point_wrt_cam(1), point_wrt_cam(2), 1;
    //             Eigen::Vector3d pixel_homo;
    //             pixel_homo = _pCameraModel->_P_L * point_homo;

    //             cv::circle(img0, f->pixelsL[j], 3, cv::Scalar(255,0,0));
    //             cv::circle(img0, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(0,0,255));
    //         }
    //     }
    // }
    // cv::imshow("before optimization", img0);

    // Set the solver.
    ceres::Solver::Options options;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.max_num_iterations = 8;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR; // bundle adjustment problems have a special sparsity structure that can be solved much more efficiently using Schur-based solvers.
    options.minimizer_progress_to_stdout = true;
    // options.check_gradients = true; // default: false
    // options.gradient_check_relative_precision = 1e-6; // default: 1e-8
    ceres::Solver::Summary summary;

    // std::cout << "Before optimizing:\npose: " << _pose[actualSize-1][3] << ", " << _pose[actualSize-1][4] << ", "<< _pose[actualSize-1][5] << std::endl;
    // // Run the solver.
    ceres::Solve(options, &problem, &summary);
    // std::cout << "After optimizing:\npose: " << _pose[actualSize-1][3] << ", " << _pose[actualSize-1][4] << ", "<< _pose[actualSize-1][5] << std::endl;

    // if (_verbose) {
        // Show the report.
        // std::cout << summary.BriefReport() << std::endl;
        std::cout << summary.FullReport() << std::endl;
    // }

    _pMap->updateStates(delta_pose, delta_v_dbga);
    _pMap->checkKeyframe();
    _pImuPreintegrator->updateBias();

    // for (int i = 0; i < _pFeatureTracker->_matchedFeatureIDs.size(); i++) {
    //     cfsd::Ptr<Feature> f = _pFeatureTracker->_features[_pFeatureTracker->_matchedFeatureIDs[i]];
    //     for (int j = 0; j < f->seenByFrames.size(); j++) {
    //         // Show landmarks in image.
    //         if (f->seenByFrames[j] == curFrameID) {
    //             Eigen::Vector3d r_j(_pose[actualSize-1][0], _pose[actualSize-1][1], _pose[actualSize-1][2]);
    //             Eigen::Vector3d p_j(_pose[actualSize-1][3], _pose[actualSize-1][4], _pose[actualSize-1][5]);
    //             // T_WB = (r_j, p_j), is transformation from body to world frame.
    //             // T_BW = T_WB.inverse()
    //             Sophus::SE3d T_BW = Sophus::SE3d(Sophus::SO3d::exp(r_j), p_j).inverse();
    //             Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * T_BW * f->position;
    //             Eigen::Matrix<double,4,1> point_homo;
    //             point_homo << point_wrt_cam(0), point_wrt_cam(1), point_wrt_cam(2), 1;
    //             Eigen::Vector3d pixel_homo;
    //             pixel_homo = _pCameraModel->_P_L * point_homo;

    //             cv::circle(img, f->pixelsL[j], 3, cv::Scalar(255,0,0));
    //             cv::circle(img, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(0,0,255));
    //         }
    //     }
    // }
    cv::imshow("after optimization", img);
    cv::waitKey(0);
}

void Optimizer::initialGyrBias() {
    double delta_dbg[3] = {0,0,0};

    ceres::Problem problem;

    ceres::CostFunction* gyrCost = new BiasGyrCostFunction(_pImuPreintegrator->_ic, _delta_bg);
    problem.AddResidualBlock(gyrCost, NULL, delta_dbg);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    // options.check_gradients = true;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;
    
    _delta_bg = _delta_bg + Eigen::Vector3d(delta_dbg[0], delta_dbg[1], delta_dbg[2]);
    // std::cout << "delta_bg:\n" << _delta_bg << std::endl;

    _pImuPreintegrator->setInitialGyrBias(_delta_bg);
}

void Optimizer::initialAlignment(bool isBiasConstant) {
    double delta_r[3] = {0,0,0};
    double delta_dba[3] = {0,0,0};
    
    ceres::Problem problem; // acc initial rotation and bias

    ceres::CostFunction* alignmentCost = new AlignmentCostFunction(_pImuPreintegrator->_ic, _delta_bg, _delta_ba, _pMap->_gravity);
    problem.AddResidualBlock(alignmentCost, NULL, delta_r, delta_dba);

    if (isBiasConstant) problem.SetParameterBlockConstant(delta_dba);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    // options.check_gradients = true;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;

    _pMap->setInitialRotation(Eigen::Vector3d(delta_r[0], delta_r[1], delta_r[2]));

    _delta_ba = _delta_ba + Eigen::Vector3d(delta_dba[0], delta_dba[1], delta_dba[2]);
    _pImuPreintegrator->setInitialAccBias(_delta_ba);
}

} // namespace cfsd