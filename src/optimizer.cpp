#include "cfsd/optimizer.hpp"

namespace cfsd {

Optimizer::Optimizer(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<FeatureTracker>& pFeatureTracker, const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose)
    : _pMap(pMap), _pFeatureTracker(pFeatureTracker), _pImuPreintegrator(pImuPreintegrator), _pCameraModel(pCameraModel), _verbose(verbose), _invStdT() {

    _fx = _pCameraModel->_K_L.at<double>(0,0);
    _fy = _pCameraModel->_K_L.at<double>(1,1);
    _cx = _pCameraModel->_K_L.at<double>(0,2);
    _cy = _pCameraModel->_K_L.at<double>(1,2);
    _invStdT << 1/_pCameraModel->_stdX, 0, 0, 1/_pCameraModel->_stdY;
    _useMarginalize = Config::get<bool>("useMarginalize");
    _marginalizeWeight = Config::get<double>("marginalizeWeight");
    _priorWeight = Config::get<double>("priorWeight");
    _delay = Config::get<int>("delay");

    _minimizerProgressToStdout = Config::get<bool>("minimizer_progress_to_stdout");
    _maxNumIterations = Config::get<int>("max_num_iterations");
    _maxSolverTimeInSeconds = Config::get<double>("max_solver_time_in_seconds");
    _numThreads = Config::get<int>("num_threads");
    _checkGradients = Config::get<bool>("check_gradients");
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

    int actualSize = (_pMap->_pKeyframes.size() > WINDOWSIZE) ? WINDOWSIZE : _pMap->_pKeyframes.size();
    
    std::vector<double*> delta_pose_img;
    
    int n = _pMap->_pKeyframes.size() - actualSize;
    
    // Build the problem.
    ceres::Problem problem;

    // Loss function.
    ceres::LossFunction* lossFunction = new ceres::HuberLoss(1.0);
    // ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);

    // Set up prior.
    // ceres::CostFunction* priorCost = new PriorCostFunction(_pMap, n-1, _priorWeight);
    // problem.AddResidualBlock(priorCost, NULL, delta_pose[0], delta_v_dbga[0]);
    
    // Set up imu cost function.
    for (int i = 0; i < actualSize-1; i++) {
        ceres::CostFunction* preintegrationCost = new ImuCostFunction(_pMap, n+i);
        // problem.AddResidualBlock(preintegrationCost, lossFunction, delta_pose[i], delta_v_dbga[i], delta_pose[i+1], delta_v_dbga[i+1]);
        problem.AddResidualBlock(preintegrationCost, NULL, delta_pose[i], delta_v_dbga[i], delta_pose[i+1], delta_v_dbga[i+1]);
    }

    // landmark : (std::vector of keyframeIdx and pointsIdx)
    std::unordered_map< size_t, std::vector< std::pair<int,int> > > landmarks;
    size_t landmarkID;
    for (int i = (_useMarginalize && _pMap->_pKeyframes.size() > WINDOWSIZE) ? -1 : 0; i < actualSize; i++) {
        // If number of keyframes is bigger than windowsize, it means there is frame needed marginalization, i starts from -1.
        const std::vector<cfsd::Ptr<MapPoint>>& points = _pMap->_pKeyframes[n+i]->points;
        for (int j = 0; j < points.size(); j++) {
            landmarkID =points[j]->id;

            // The purpose of this step is to repeated elements, it's due to one orb keypoint can be matched more than one time during matching.
            if (landmarks.find(landmarkID) == landmarks.end())
                landmarks[landmarkID] = std::vector< std::pair<int,int> >();
            
            landmarks[landmarkID].push_back(std::make_pair(n+i,j)); // n+i is the index for _pKeyframes, j is the index for _pKeyframes[n+i]->points
        }
    }

    // Set up image cost.
    for (const auto& l : landmarks) {
        // l.first is landmarkID
        // l.second is std::vector of keyframeIdx and pointsIdx
        int errorTerms = 0;
        delta_pose_img.clear();
        std::unordered_map<int, int> poseMask;
        bool needMarginalize = false;
        for (int i = 0; i < l.second.size(); i++) {
            const auto& pair = l.second[i];
            // pair.first is index for _pKeyframes
            // pair.second is the index for _pKeyframes[pair.first]->points
            if (poseMask.find(pair.first) == poseMask.end()) {
                if (pair.first < n) // the index of the frame to be marginalized is n-1
                    needMarginalize = true;
                else
                    delta_pose_img.push_back(delta_pose[pair.first - n]);
                poseMask[pair.first] = 1;
                errorTerms++;
            }
        }
        if (errorTerms < 2) continue;
        // if (needMarginalize && errorTerms < 3) continue;

        // Each error term is a 2x1 vector.
        Eigen::VectorXd error(2*errorTerms);
        // Each F_ij block (2x6 matrix) is partial derivative of error w.r.t pose_i.
        Eigen::MatrixXd F(2*errorTerms, 6*errorTerms);
        F.setZero();
        // // Each E_ij block (2x3 matrix) is partial detivative of error w.r.t landmark_j.
        // Eigen::MatrixXd E(2*errorTerms, 3);

        for (int i = 0, j = 0; i < l.second.size(); i++) {
            const auto& pair = l.second[i];
            
            if (poseMask[pair.first] == 0) continue;
            poseMask[pair.first] = 0;

            const cfsd::Ptr<Keyframe>& windowFrame = _pMap->_pKeyframes[pair.first];
            const cfsd::Ptr<MapPoint>& mp = windowFrame->points[pair.second];

            Eigen::Vector3d temp = windowFrame->R.inverse() * (_pMap->_frameAndPoints[mp->frameID][mp->positionIdx] - windowFrame->p);
            Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * temp;
            double x = point_wrt_cam.x();
            double y = point_wrt_cam.y();
            double z = point_wrt_cam.z();
            
            Eigen::Matrix<double,2,3> d_e_pcam;
            d_e_pcam(0,0) = _fx / z;
            d_e_pcam(0,1) = 0;
            d_e_pcam(0,2) = -_fx * x / (z * z);
            d_e_pcam(1,0) = 0;
            d_e_pcam(1,1) = _fy / z;
            d_e_pcam(1,2) = -_fy * y / (z * z);
        
            error(2*j) = _invStdT(0,0) * (_fx * x / z + _cx - mp->pixel.x);
            error(2*j+1) = _invStdT(1,1) * (_fy * y / z + _cy - mp->pixel.y);

            // F.block<2,3>(2*j, 6*j) = _invStdT * d_e_pcam * _pCameraModel->_T_CB.so3().matrix() * Sophus::SO3d::hat(temp);
            F.block<2,3>(2*j, 6*j+3) = -_invStdT * d_e_pcam * _pCameraModel->_T_CB.so3().matrix();
            F.block<2,3>(2*j, 6*j) = -F.block<2,3>(2*j, 6*j+3) * Sophus::SO3d::hat(temp);

            // // E.block<2,3>(2*j, 0) = _invStdT * d_e_cam * _pCameraModel->_T_CB.so3().matrix() * windowFrame->R.inverse().matrix();
            // E.block<2,3>(2*j, 0) = -F.block<2,3>(2*j, 6*j+3) * windowFrame->R.inverse().matrix();

            j++;
        }

        // Marginalize prior pose.
        if (needMarginalize) {
            Eigen::MatrixXd F0 = F.leftCols<6>();
            Eigen::MatrixXd F1 = F.rightCols(F.cols()-6);

            Eigen::MatrixXd D = F1.transpose() * (Eigen::MatrixXd::Identity(F0.rows(), F0.rows()) - F0 * (F0.transpose() * F0).inverse() * F0.transpose());
            Eigen::VectorXd tempError = D * error;

            if (std::isnan(tempError(0))) {
                F = F1;
                errorTerms--;
            }
            else {
                double norm = D.norm() / _marginalizeWeight;
                error = tempError / norm;
                F = D * F1 / norm;
                errorTerms = D.rows() / 6;
            }
        }

        // // Structure-less formlization. (failed)
        // // Use svd to calculate the unitary basis of the null space of E.
        // Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU);
        // // E.rows() - svd.singularValues().rows()
        // // E.rows() is 2*errorTerms; the rank of E is 3, i.e. svd.singularValues().rows() is 3
        // Eigen::MatrixXd E_b_nullspace = svd.matrixU().rightCols(2*errorTerms-3);

        // Set up reprojection cost function for a specific landmark (a.k.a. residuals).
        // ceres::CostFunction* reprojectCost = new ImageCostFunction(errorTerms, D * error, D * F); // marginalized factor
        // ceres::CostFunction* reprojectCost = new ImageCostFunction(errorTerms, error, F, E_b_nullspace); // structure-less factor
        ceres::CostFunction* reprojectCost = new ImageCostFunction(errorTerms, error, F); // normal factor, assume landmark fixed
        problem.AddResidualBlock(reprojectCost, lossFunction, delta_pose_img);
        // problem.AddResidualBlock(reprojectCost, NULL, delta_pose_img);
    }

    #ifdef SHOW_IMG
    // Show pixels and reprojected pixels before optimization.
    int yOffset = _pFeatureTracker->_cropOffset;
    const cfsd::Ptr<Keyframe>& latestFrame = _pMap->_pKeyframes.back();
    const std::vector<cfsd::Ptr<MapPoint>>& points = latestFrame->points;
    for (int i = 0; i < points.size(); i++) {
        cfsd::Ptr<MapPoint> mp = points[i];
        Eigen::Vector3d pixel_homo = _pCameraModel->_P_L.block<3,3>(0,0) * (_pCameraModel->_T_CB * (latestFrame->R.inverse() * (_pMap->_frameAndPoints[mp->frameID][mp->positionIdx] - latestFrame->p)));
        #ifdef CFSD
        cv::rectangle(img, cv::Point(mp->pixel.x-4, mp->pixel.y-4 + yOffset), cv::Point(mp->pixel.x+4, mp->pixel.y+4 + yOffset), cv::Scalar(0));
        cv::circle(img, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2) + yOffset), 3, cv::Scalar(0));
        #else
        cv::rectangle(img, cv::Point(mp->pixel.x-4, mp->pixel.y-4), cv::Point(mp->pixel.x+4, mp->pixel.y+4), cv::Scalar(0));
        cv::circle(img, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(0));
        #endif
    }
    #endif

    // Set the solver.
    ceres::Solver::Options options;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR; // bundle adjustment problems have a special sparsity structure that can be solved much more efficiently using Schur-based solvers.
    options.minimizer_progress_to_stdout = _minimizerProgressToStdout;
    options.max_num_iterations = _maxNumIterations;
    options.max_solver_time_in_seconds = _maxSolverTimeInSeconds;
    options.num_threads = _numThreads;
    options.check_gradients = _checkGradients; // default: false
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

    #ifdef SHOW_IMG
    // Show pixels and reprojected pixels after optimization.
    for (int i = 0; i < points.size(); i++) {
        cfsd::Ptr<MapPoint> mp = points[i];
        Eigen::Vector3d pixel_homo = _pCameraModel->_P_L.block<3,3>(0,0) * (_pCameraModel->_T_CB * (latestFrame->R.inverse() * (_pMap->_frameAndPoints[mp->frameID][mp->positionIdx] - latestFrame->p)));
        
        #ifdef CFSD
        cv::circle(img, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2) + yOffset), 3, cv::Scalar(255));
        #else
        cv::circle(img, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(255));
        #endif
    }
    cv::imshow("before vs. after optimization", img);
    cv::waitKey(_delay);
    #endif
}

void Optimizer::globalOptimize() {
    int numKeyframes = _pMap->_pKeyframes.size() - 1;
    double** delta_pose = new double*[numKeyframes];
    double** delta_v_dbga = new double*[numKeyframes];
    for (int i = 0; i < numKeyframes; i++) {
        delta_pose[i] = new double[6];
        for (int j = 0; j < 6; j++)
            delta_pose[i][j] = 0;

        delta_v_dbga[i] = new double[9];
        for (int j = 0; j < 9; j++)
            delta_v_dbga[i][j] = 0;
    }

    // Build the problem.
    ceres::Problem problem;

    // Loss function.
    ceres::LossFunction* lossFunction = new ceres::HuberLoss(1.0);
    // ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0);

    // Set up prior.
    ceres::CostFunction* priorCost = new PriorCostFunction(_pMap, 0, _priorWeight);
    problem.AddResidualBlock(priorCost, NULL, delta_pose[0], delta_v_dbga[0]);

    // Set up imu cost function.
    for (int i = 0; i < numKeyframes-1; i++) {
        ceres::CostFunction* preintegrationCost = new ImuCostFunction(_pMap, i);
        // problem.AddResidualBlock(preintegrationCost, lossFunction, delta_pose[i], delta_v_dbga[i], delta_pose[i+1], delta_v_dbga[i+1]);
        problem.AddResidualBlock(preintegrationCost, NULL, delta_pose[i], delta_v_dbga[i], delta_pose[i+1], delta_v_dbga[i+1]);
    }

    // (landmark : frame)
    std::unordered_map< size_t, std::vector< std::pair<int,int> > > landmarks;
    size_t landmarkID;
    for (int i = 0; i < numKeyframes; i++) {
        const std::vector<cfsd::Ptr<MapPoint>>& points = _pMap->_pKeyframes[i]->points;
        for (int j = 0; j < points.size(); j++) {
            landmarkID =points[j]->id;
            if (landmarks.find(landmarkID) == landmarks.end())
                landmarks[landmarkID] = std::vector< std::pair<int,int> >();
            landmarks[landmarkID].push_back(std::make_pair(i,j));
        }
    }

    double** delta_x = new double*[landmarks.size()];
    for (int i = 0; i < landmarks.size(); i++) {
        delta_x[i] = new double[3];
        for (int j = 0; j < 3; j++)
            delta_x[i][j] = 0;
    }

    // Set up image cost.
    std::vector<double*> delta;
    int k = 0;
    for (const auto& l : landmarks) {
        // Don't consider landmarks that are not seen by all frames in the sliding window.
        // if (l.second.size() != actualSize) continue;

        int errorTerms = 0;
        delta.clear();
        std::unordered_map<int, int> delta_pose_mask;
        for (int i = 0; i < l.second.size(); i++) {
            const auto& pair = l.second[i];
            if (delta_pose_mask.find(pair.first) == delta_pose_mask.end()) {
                delta.push_back(delta_pose[pair.first]);
                delta_pose_mask[pair.first] = 1;
                errorTerms++;
            }
        }
        if (errorTerms < 2) continue;
        delta.push_back(delta_x[k++]);

        // Each error term is a 2x1 vector.
        Eigen::VectorXd error(2*errorTerms);
        // Each F_ij block (2x6 matrix) is partial derivative of error w.r.t pose_i.
        Eigen::MatrixXd F(2*errorTerms, 6*errorTerms);
        F.setZero();
        // Each E_ij block (2x3 matrix) is partial detivative of error w.r.t landmark_j.
        Eigen::MatrixXd E(2*errorTerms, 3);

        for (int i = 0, j = 0; i < l.second.size(); i++) {
            const auto& pair = l.second[i];
            
            if (delta_pose_mask[pair.first] == 0) continue;
            delta_pose_mask[pair.first] = 0;

            const cfsd::Ptr<Keyframe>& windowFrame = _pMap->_pKeyframes[pair.first];
            const cfsd::Ptr<MapPoint>& mp = windowFrame->points[pair.second];

            Eigen::Vector3d temp = windowFrame->R.inverse() * (_pMap->_frameAndPoints[mp->frameID][mp->positionIdx] - windowFrame->p);
            Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * temp;
            double x = point_wrt_cam.x();
            double y = point_wrt_cam.y();
            double z = point_wrt_cam.z();
            
            Eigen::Matrix<double,2,3> d_e_pcam;
            d_e_pcam(0,0) = _fx / z;
            d_e_pcam(0,1) = 0;
            d_e_pcam(0,2) = -_fx * x / (z * z);
            d_e_pcam(1,0) = 0;
            d_e_pcam(1,1) = _fy / z;
            d_e_pcam(1,2) = -_fy * y / (z * z);
        
            error(2*j) = _invStdT(0,0) * (_fx * x / z + _cx - mp->pixel.x);
            error(2*j+1) = _invStdT(1,1) * (_fy * y / z + _cy - mp->pixel.y);

            // F.block<2,3>(2*j, 6*j) = _invStdT * d_e_pcam * _pCameraModel->_T_CB.so3().matrix() * Sophus::SO3d::hat(temp);
            F.block<2,3>(2*j, 6*j+3) = -_invStdT * d_e_pcam * _pCameraModel->_T_CB.so3().matrix();
            F.block<2,3>(2*j, 6*j) = -F.block<2,3>(2*j, 6*j+3) * Sophus::SO3d::hat(temp);

            // E.block<2,3>(2*j, 0) = _invStdT * d_e_cam * _pCameraModel->_T_CB.so3().matrix() * windowFrame->R.inverse().matrix();
            E.block<2,3>(2*j, 0) = -F.block<2,3>(2*j, 6*j+3) * windowFrame->R.inverse().matrix();

            j++;
        }

        // Set up reprojection cost function for a specific landmark (a.k.a. residuals).
        ceres::CostFunction* fullBACost = new FullBAFunction(errorTerms+1, error, F, E);
        problem.AddResidualBlock(fullBACost, lossFunction, delta);
    }

    // Set the solver.
    ceres::Solver::Options options;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR; // bundle adjustment problems have a special sparsity structure that can be solved much more efficiently using Schur-based solvers.
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = _maxNumIterations-10;
    // options.max_solver_time_in_seconds = _maxSolverTimeInSeconds;
    options.num_threads = _numThreads;
    // options.check_gradients = _checkGradients; // default: false
    // options.gradient_check_relative_precision = 1e-6; // default: 1e-8
    ceres::Solver::Summary summary;

    // Run the solver.
    ceres::Solve(options, &problem, &summary);

    // if (_verbose) {
        // Show the report.
        // std::cout << summary.BriefReport() << std::endl;
        std::cout << summary.FullReport() << std::endl;
    // }

    _pMap->updateAllStates(delta_pose, delta_v_dbga);

    for (int i = 0; i < numKeyframes; i++) {
        delete[] delta_pose[i];
        delete[] delta_v_dbga[i];
    }
    delete[] delta_pose;
    delete[] delta_v_dbga;

    for (int i = 0; i < landmarks.size(); i++) 
        delete[] delta_x[i];
    delete[] delta_x;
}


void Optimizer::initialGyrBias() {
    double delta_dbg[3] = {0,0,0};
    
    ceres::Problem problem;

    for (int i = 0; i < WINDOWSIZE-1; i++) {
        cfsd::Ptr<Keyframe>& sfmFrame1 = _pMap->_pKeyframes[i];
        cfsd::Ptr<Keyframe>& sfmFrame2 = _pMap->_pKeyframes[i+1];
        ceres::CostFunction* gyrCost = new BiasGyrCostFunction(sfmFrame2->pImuConstraint, sfmFrame1->R, sfmFrame2->R);
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
        cfsd::Ptr<Keyframe>& sfmFrame1 = _pMap->_pKeyframes[i];
        cfsd::Ptr<Keyframe>& sfmFrame2 = _pMap->_pKeyframes[i+1];
        ceres::CostFunction* gravityVelocityCost = new GravityVelocityCostFunction(sfmFrame2->pImuConstraint, sfmFrame1->R, sfmFrame1->p, sfmFrame2->p);
        // problem.AddResidualBlock(gravityVelocityCost, nullptr, delta_g, delta_v[i], delta_v[i+1]);
        problem.AddResidualBlock(gravityVelocityCost, new ceres::HuberLoss(1.0), delta_g, delta_v[i], delta_v[i+1]);
    }

    // Assume initial velocity is zero?
    // problem.SetParameterBlockConstant(delta_v[0]);

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

        euroc imu coordinate system
              x |  / z
                | /
                ------ y
    
        kitti imu coordinate system
              z |  / x
                | /
          y -----
    */
    
    // Find rotation of gravity from the initial body frame to world frame (inertial frame), and refine the gravity magnitude.

    // Only consider rotation around non-gravitational axis.
    double delta_r[2] = {0,0};
    Eigen::Vector3d unit_gravity;
    
    #ifdef CFSD
    unit_gravity << 0.0, 0.0, 1.0;
    #endif

    #ifdef EUROC
    unit_gravity << -1.0, 0.0, 0.0;
    #endif
    
    #ifdef KITTI
    unit_gravity << 0.0, 0.0, -1.0;
    #endif
    
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

    #ifdef CFSD
    _pMap->updateInitialRotation(0, Eigen::Vector3d(delta_r[0], delta_r[1], 0.0));
    #endif

    #ifdef EUROC
    _pMap->updateInitialRotation(0, Eigen::Vector3d(0.0, delta_r[0], delta_r[1]));
    #endif
    
    #ifdef KITTI
    _pMap->updateInitialRotation(0, Eigen::Vector3d(delta_r[0], delta_r[1], 0.0));
    #endif
}

void Optimizer::initialAccBias() {
    double delta_dba[3] = {0,0,0};

    ceres::Problem problem; // acc initial bias

    for (int i = 0; i < WINDOWSIZE-1; i++) {
        cfsd::Ptr<Keyframe>& sfmFrame1 = _pMap->_pKeyframes[i];
        cfsd::Ptr<Keyframe>& sfmFrame2 = _pMap->_pKeyframes[i+1];
        ceres::CostFunction* accCost = new AccCostFunction(sfmFrame2->pImuConstraint, sfmFrame1->R, sfmFrame1->v, sfmFrame2->v, sfmFrame1->p, sfmFrame2->p, _pMap->_gravity);
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