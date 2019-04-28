#include "cfsd/optimizer.hpp"

namespace cfsd {

Optimizer::Optimizer(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<FeatureTracker>& pFeatureTracker, const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose)
    : _pMap(pMap), _pFeatureTracker(pFeatureTracker), _pImuPreintegrator(pImuPreintegrator), _pCameraModel(pCameraModel), _verbose(verbose) { 
}

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
    
    int n = _pMap->_frames.size() - actualSize;
    
    // Build the problem.
    ceres::Problem problem;

    // Set up imu cost function.
    for (int i = 0; i < actualSize-1; i++) {
        ceres::CostFunction* preintegrationCost = new ImuCostFunction(_pMap, n+i);
        problem.AddResidualBlock(preintegrationCost, nullptr, delta_pose[i], delta_v_dbga[i], delta_pose[i+1], delta_v_dbga[i+1]);
    }

    // historical information...................

    // Set up reprojection cost function (a.k.a. residuals).
    for (int i = 0; i < actualSize; i++) {
        for (int j = 0; j < _pMap->_frames[n+i].size(); j++) {
            auto pixel_position = _pMap->_frames[n+i][j];
            // ceres::CostFunction* reprojectCost = new ImageCostFunction(_pCameraModel, pixel_position.first, pixel_position.second, _pMap->_R[n+i], _pMap->_p[n+i]);
            ceres::CostFunction* reprojectCost = new ceres::AutoDiffCostFunction<AutoDiffImageCostFunction, 2, 6>(new AutoDiffImageCostFunction(_pCameraModel, pixel_position.first, pixel_position.second, _pMap->_R[n+i], _pMap->_p[n+i]));
            problem.AddResidualBlock(reprojectCost, new ceres::HuberLoss(1.0), delta_pose[i]);
        }
    }

    // Show pixels and reprojected pixels before optimization.
    cv::Mat img0 = img.clone();
    for (int i = actualSize-1, j = 0; j < _pMap->_frames[n+i].size(); j++) {
        auto pixel_position = _pMap->_frames[n+i][j];
        Eigen::Vector3d pixel_homo = _pCameraModel->_P_L.block<3,3>(0,0) * (_pCameraModel->_T_CB * (_pMap->_R[n+i].inverse() * (pixel_position.second - _pMap->_p[n+i]))) + _pCameraModel->_P_L.block<3,1>(0,3);
        cv::circle(img0, pixel_position.first, 3, cv::Scalar(255,0,0));
        cv::circle(img0, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(0,0,255));
    }
    cv::imshow("before optimization", img0);

    // Set the solver.
    ceres::Solver::Options options;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR; // bundle adjustment problems have a special sparsity structure that can be solved much more efficiently using Schur-based solvers.
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = Config::get<int>("max_num_iterations");
    options.max_solver_time_in_seconds = Config::get<double>("max_solver_time_in_seconds");
    options.num_threads = Config::get<int>("num_threads");
    options.check_gradients = Config::get<bool>("check_gradients"); // default: false
    // options.gradient_check_relative_precision = 1e-6; // default: 1e-8
    ceres::Solver::Summary summary;

    // Run the solver.
    ceres::Solve(options, &problem, &summary);

    // if (_verbose) {
        // Show the report.
        // std::cout << summary.BriefReport() << std::endl;
        std::cout << summary.FullReport() << std::endl;
    // }

    _pMap->updateStates(delta_pose, delta_v_dbga);
    _pMap->checkKeyframe();
    _pImuPreintegrator->updateBias();

    // Show pixels and reprojected pixels after optimization.
    for (int i = actualSize-1, j = 0; j < _pMap->_frames[n+i].size(); j++) {
        auto pixel_position = _pMap->_frames[n+i][j];
        Eigen::Vector3d pixel_homo = _pCameraModel->_P_L.block<3,3>(0,0) * (_pCameraModel->_T_CB * (_pMap->_R[n+i].inverse() * (pixel_position.second - _pMap->_p[n+i]))) + _pCameraModel->_P_L.block<3,1>(0,3);
        cv::circle(img, pixel_position.first, 3, cv::Scalar(255,0,0));
        cv::circle(img, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(0,0,255));
    }
    cv::imshow("after optimization", img);
    cv::waitKey(0);
}

void Optimizer::initialGyrBias() {
    double delta_dbg[3] = {0,0,0};
    
    ceres::Problem problem;

    for (int i = 0; i < WINDOWSIZE-1; i++) {
        ceres::CostFunction* gyrCost = new BiasGyrCostFunction(_pMap->_imuConstraint[i], _pMap->_R[i], _pMap->_R[i+1]);
        // problem.AddResidualBlock(gyrCost, nullptr, delta_dbg);
        problem.AddResidualBlock(gyrCost, new ceres::HuberLoss(1.0), delta_dbg);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    // options.check_gradients = Config::get<bool>("check_gradients");
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;
    
    Eigen::Vector3d delta_bg(delta_dbg[0], delta_dbg[1], delta_dbg[2]);
    _pImuPreintegrator->setInitialGyrBias(delta_bg);
    _pMap->repropagate(0, delta_bg, Eigen::Vector3d::Zero());
}

void Optimizer::initialGravityVelocity() {
    // Estimate the gracity in the initial body frame.
    double delta_g[3] = {0,0,0};
    double delta_v[WINDOWSIZE][3];
    for (int i = 0; i < WINDOWSIZE; i++) {
        delta_v[i][0] = 0;
        delta_v[i][1] = 0;
        delta_v[i][2] = 0;
    }

    ceres::Problem problem;

    for (int i = 0; i < WINDOWSIZE-1; i++) {
        ceres::CostFunction* gravityVelocityCost = new GravityVelocityCostFunction(_pMap->_imuConstraint[i], _pMap->_R[i], _pMap->_p[i], _pMap->_p[i+1]);
        // problem.AddResidualBlock(gravityVelocityCost, nullptr, delta_g, delta_v[i], delta_v[i+1]);
        problem.AddResidualBlock(gravityVelocityCost, new ceres::HuberLoss(1.0), delta_g, delta_v[i], delta_v[i+1]);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    // options.check_gradients = Config::get<bool>("check_gradients");
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;
    
    _pMap->setInitialGravity(Eigen::Vector3d(delta_g[0], delta_g[1], delta_g[2]));
    _pMap->updateInitialVelocity(0, delta_v);
}

void Optimizer::initialAlignment() {
    /*  cfsd imu coordinate system
              / x
             /
            ------ y
            |
            | z
    
        kitti imu coordinate system
              z |  / x
                | /
          y -----

        euroc imu coordinate system
              x |  / z
                | /
                ------ y
    */
    
    // Find rotation of gravity from the initial body frame to world frame (inertial frame), and refine the gravity magnitude.

    // For euroc: only consider rotation around y and z axis.
    double delta_r[2] = {0,0};
    Eigen::Vector3d unit_gravity(-1,0,0);
    
    ceres::Problem problem; // initial rotation

    ceres::CostFunction* alignmentCost = new AlignmentCostFunction(_pMap->_init_gravity, unit_gravity);
    problem.AddResidualBlock(alignmentCost, nullptr, delta_r);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    // options.check_gradients = Config::get<bool>("check_gradients");
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;

    _pMap->updateInitialRotation(0, Eigen::Vector3d(0, delta_r[0], delta_r[1]));
}

void Optimizer::initialAccBias() {
    double delta_dba[3] = {0,0,0};

    ceres::Problem problem; // acc initial bias

    for (int i = 0; i < WINDOWSIZE-1; i++) {
        ceres::CostFunction* accCost = new AccCostFunction(_pImuPreintegrator->_ic, _pMap->_R[i], _pMap->_v[i], _pMap->_v[i+1], _pMap->_p[i], _pMap->_p[i+1], _pMap->_gravity);
        // problem.AddResidualBlock(accCost, nullptr, delta_dba);
        problem.AddResidualBlock(accCost, new ceres::HuberLoss(1.0), delta_dba);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    // options.check_gradients = Config::get<bool>("check_gradients");
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;

    Eigen::Vector3d delta_ba(delta_dba[0], delta_dba[1], delta_dba[2]);
    _pImuPreintegrator->setInitialAccBias(delta_ba);
    _pMap->repropagate(0, Eigen::Vector3d::Zero(), delta_ba);
}

} // namespace cfsd