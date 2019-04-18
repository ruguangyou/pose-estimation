#include "cfsd/optimizer.hpp"

namespace cfsd {

struct ImageCostFunction : public ceres::SizedCostFunction<2, /* residuals */
                                                           6  /* pose (r, p) at time j */> {
                                                           // 3  /* landmark position */> {
    ImageCostFunction(const cfsd::Ptr<CameraModel>& pCameraModel, const Eigen::Vector3d& point, const cv::Point2d& pixel) : _pCameraModel(pCameraModel), _point(point), _pixel(pixel) {}

    virtual ~ImageCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: pose_j
        // pose: [rx,ry,rz, px,py,pz]
        Eigen::Vector3d r_j(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d p_j(parameters[0][3], parameters[0][4], parameters[0][5]);
        // T_WB = (r_j, p_j), is transformation from body to world frame.
        // T_BW = T_WB.inverse()
        Sophus::SO3d R_BW = Sophus::SO3d::exp(r_j);
        Sophus::SE3d T_BW = Sophus::SE3d(R_BW, p_j).inverse();

        // 3D landmark homogeneous coordinates w.r.t camera frame.
        Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * T_BW * _point;
        double x1 = point_wrt_cam(0);
        double y1 = point_wrt_cam(1);
        double z1 = point_wrt_cam(2);
        Eigen::Matrix<double,4,1> point_homo;
        point_homo << x1, y1, z1, 1;

        Eigen::Vector3d pixel_homo;
        pixel_homo = _pCameraModel->_P_L * point_homo;
        residuals[0] = pixel_homo(0)/pixel_homo(2) - _pixel.x;
        residuals[1] = pixel_homo(1)/pixel_homo(2) - _pixel.y;
        
        // Compute jacobians which are crutial for optimization algorithms like Guass-Newton.
        if (!jacobians) return true;

        double fx = _pCameraModel->_P_L(0,0);
        double fy = _pCameraModel->_P_L(1,1);
        Eigen::Matrix<double,2,3> d1;
        d1 << fx / z1, 0, -fx * x1 / (z1*z1), 0, fy / z1, -fy * y1 / (z1*z1);
        
        // Jacobian(2x6) of residual(2x1) w.r.t. ParameterBlock[0](6x1), i.e. pose_j
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6>> jacobian_pose(jacobians[0]);

            Eigen::Matrix<double,3,6> d2;
            d2.block<3,3>(0,0) = -Sophus::SO3d::hat(R_BW * (_point - p_j));
            d2.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

            jacobian_pose = d1 * _pCameraModel->_T_BC.so3().matrix() * d2;
        }

        // // Jacobian(2x3) of residual(2x1) w.r.t. ParameterBlock[1](3x1), i.e. landmark
        // if (jacobians[1]) {
        //     Eigen::Map<Eigen::Matrix<double, 2, 3>> jacobian_landmark(jacobians[1]);

        //     jacobian_landmark = d1 * _pCameraModel->_T_BC.so3().matrix() * R_BW.matrix();
        // }

        return true;
    }

  private:
    cfsd::Ptr<CameraModel> _pCameraModel;
    Eigen::Vector3d _point; // Landmark w.r.t world frame.
    cv::Point2d _pixel; // Pixel coordinates.
};

struct ImuCostFunction : public ceres::SizedCostFunction<15, /* residuals */
                                                          6, /* pose (r, p) at time i */
                                                          9, /* velocity, bias of gyr and acc at time i */
                                                          6, /* pose (r, p) at time j */
                                                          9  /* velocity, bias of gyr and acc at time j */> {
    ImuCostFunction(const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<ImuConstraint>& ic) : _pImuPreintegrator(pImuPreintegrator), _ic(ic) {}

    virtual ~ImuCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: pose_i, vbga_i, pose_j, vbga_j
        // pose: [rx,ry,rz, px,py,pz]

        // Quaternion: w + xi + yj + zk
        // When creating an Eigen quaternion through the constructor the elements are accepted in w, x, y, z order;
        // whereas Eigen stores the elements in memory as [x, y, z, w] where the real part is last.
        // The commonly used memory layout for quaternion is [w, x, y, z], so is it in Ceres.
        // Eigen::Quaterniond q_i(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        // Eigen::Quaterniond q_i(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        // Update: instead of using quaternion, use rotation vector which can be mapped to SO3 space using Exp map.
        // Rotation vector.
        Eigen::Vector3d r_i(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d r_j(parameters[2][0], parameters[2][1], parameters[2][2]);

        // Position.
        Eigen::Vector3d p_i(parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Vector3d p_j(parameters[2][3], parameters[2][4], parameters[2][5]);

        // Velocity.
        Eigen::Vector3d v_i(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d v_j(parameters[3][0], parameters[3][1], parameters[3][2]);

        // Gyroscope bias.
        Eigen::Vector3d bg_i(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d bg_j(parameters[3][3], parameters[3][4], parameters[3][5]);

        // Accelerometer bias.
        Eigen::Vector3d ba_i(parameters[1][6], parameters[1][7], parameters[1][8]);
        Eigen::Vector3d ba_j(parameters[3][6], parameters[3][7], parameters[3][8]);

        // Call Map::evaluate() to compute residuals and jacobians.
        return _pImuPreintegrator->evaluate(_ic, r_i, v_i, p_i, bg_i, ba_i, r_j, v_j, p_j, bg_j, ba_j, residuals, jacobians);
    }

  private:
    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;
    cfsd::Ptr<ImuConstraint> _ic;
};

struct PoseParameterization : public ceres::LocalParameterization {
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // input: pose [r,p]
        //        or rotation and bias [r, b]
        Eigen::Map<const Eigen::Vector3d> r(x);
        Eigen::Map<const Eigen::Vector3d> p(x+3);

        Eigen::Map<const Eigen::Vector3d> dr(delta);
        Eigen::Map<const Eigen::Vector3d> dp(delta+3);

        Eigen::Map<Eigen::Vector3d> rr(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> pp(x_plus_delta+3);

        rr = Sophus::SO3d::exp(r+dr).log();
        pp = p + dp;

        return true;
    }

    virtual bool ComputeJacobian (const double* x, double* jacobian) const {
        Eigen::Map<Eigen::Matrix<double,6,6>> J(jacobian);
        J.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 6; }

    virtual int LocalSize() const { return 6; }
};

//////////////////////////////////////////////////
struct ReprojectCostFunction {
    ReprojectCostFunction(bool isLeft, const Eigen::Vector3d& point, const cv::Point2d& pixel, const cfsd::Ptr<CameraModel>& pCameraModel) 
        : isLeft(isLeft), point(point), pCameraModel(pCameraModel) { u = pixel.x; v = pixel.y; }

    template<typename T>
    bool operator() (const T* const pose_j, T* residuals) const {
        Eigen::Matrix<T,3,1> r_j(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Matrix<T,3,1> p_j(pose_j[3], pose_j[4], pose_j[5]);
        // T_WB = (r_j, p_j), is transformation from body to world frame.
        // T_BW = T_WB.inverse()
        Sophus::SE3<T> T_BW = Sophus::SE3<T>(Sophus::SO3<T>::exp(r_j), p_j).inverse();

        // 3D landmark homogeneous coordinates w.r.t camera frame.
        Eigen::Matrix<T,3,1> point_wrt_cam = pCameraModel->_T_CB * T_BW * point;

        Eigen::Matrix<T,4,1> point_homo;
        point_homo << point_wrt_cam(0), point_wrt_cam(1), point_wrt_cam(2), T(1);
        Eigen::Matrix<T,3,1> pixel_homo;
        if (isLeft)
            pixel_homo = pCameraModel->_P_L * point_homo;
        else
            pixel_homo = pCameraModel->_P_R * point_homo;

        residuals[0] = pixel_homo(0)/pixel_homo(2) - T(u);
        residuals[1] = pixel_homo(1)/pixel_homo(2) - T(v);
        
        return true;
    }

  private:
    // Reproject to left image or right image.
    bool isLeft;

    // 3D landmark point is w.r.t world frame.
    Eigen::Vector3d point;
    
    // Pixel coordinates.
    double u, v;
    
    cfsd::Ptr<CameraModel> pCameraModel;
};
//////////////////////////////////////////////////

struct AccCostFunctor {
    template<typename T>
    bool operator() (const T* const rvec, const T* const ba, T* residuals) const {
        Eigen::Matrix<T,3,1> _rvec(rvec[0], rvec[1], rvec[2]);
        Eigen::Matrix<T,3,1> _ba(ba[0], ba[1], ba[2]);
        Sophus::SO3<T> R = Sophus::SO3<T>::exp(_rvec);
        Eigen::Matrix<T,3,1> correctedAcc = R * (_acc - _ba);

        Eigen::Map<Eigen::Matrix<T,3,1>> residual(residuals);
        residual = (R * (_acc - _ba) + _g);

        return true;
    }

    AccCostFunctor(const Eigen::Vector3d& acc, const Eigen::Vector3d& gravity) : _acc(acc), _g(gravity) {}

    Eigen::Vector3d _acc;
    Eigen::Vector3d _g;
};

struct GyrCostFunctor {
    template<typename T>
    bool operator() (const T* const bg, T* residuals) const {
        Eigen::Matrix<T,3,1> _bg(bg[0], bg[1], bg[2]);
        Eigen::Map<Eigen::Matrix<T,3,1>> residual(residuals);
        residual = _gyr - _bg;
        return true;
    }

    GyrCostFunctor(const Eigen::Vector3d& gyr) : _gyr(gyr) {}

    Eigen::Vector3d _gyr;
};

struct RotationParameterization : public ceres::LocalParameterization {
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // input: rvec
        Eigen::Map<const Eigen::Vector3d> rvec(x);
        Eigen::Map<const Eigen::Vector3d> drvec(delta);
        Eigen::Map<Eigen::Vector3d> rr(x_plus_delta);
        rr = Sophus::SO3d::exp(rvec+drvec).log();

        return true;
    }

    virtual bool ComputeJacobian (const double* x, double* jacobian) const {
        Eigen::Map<Eigen::Matrix3d> J(jacobian);
        J.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 3; }

    virtual int LocalSize() const { return 3; }
};

// ############################################################

Optimizer::Optimizer(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<FeatureTracker>& pFeatureTracker, const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose)
    : _pMap(pMap), _pFeatureTracker(pFeatureTracker), _pImuPreintegrator(pImuPreintegrator), _pCameraModel(pCameraModel), _verbose(verbose) {}

void Optimizer::motionOnlyBA(const cv::Mat& img) {
    // For the first few frames, the local window cannot fit, i.e. actualSize < WINDOWSIZE
    int actualSize = _pMap->getStates(_pose, _v_bga);

    // Build the problem.
    ceres::Problem problem;

    // Add parameter block with parameterization.
    ceres::LocalParameterization* poseParameterization = new PoseParameterization();
    for (int i = 0; i < actualSize; i++) {
        problem.AddParameterBlock(_pose[i], 6, poseParameterization);
        problem.AddParameterBlock(_v_bga[i], 9);
    }

    // Current frame ID.
    int curFrameID =  _pFeatureTracker->_features[_pFeatureTracker->_matchedFeatureIDs[0]]->seenByFrames.back(); 

    cv::Mat img0 = img.clone();
    // Set up reprojection cost function (a.k.a. residuals).
    for (int i = 0; i < _pFeatureTracker->_matchedFeatureIDs.size(); i++) {
        cfsd::Ptr<Feature> f = _pFeatureTracker->_features[_pFeatureTracker->_matchedFeatureIDs[i]];
        for (int j = 0; j < f->seenByFrames.size(); j++) {
            // Check if the frame is within the local window.
            int idx = f->seenByFrames[j] - curFrameID + actualSize - 1;
            if (idx >= 0) {
                // Reproject to left image.
                // double landmark[3] = {f->position(0), f->position(1), f->position(2)};
                // problem.AddParameterBlock(landmark, 3);
                // Keep landmark constant, perform motion-only optimization.
                // problem.SetParameterBlockConstant(landmark);
                ceres::CostFunction* reprojectCost = new ImageCostFunction(_pCameraModel, f->position, f->pixelsL[j]);
                problem.AddResidualBlock(reprojectCost, nullptr, _pose[idx]);

                // // Reproject to left image.
                // ceres::CostFunction* reprojectCost = new ceres::AutoDiffCostFunction<ReprojectCostFunction, 2, 6>(new ReprojectCostFunction(1, f->position, f->pixelsL[j], _pCameraModel));
                // problem.AddResidualBlock(reprojectCost, nullptr, _pose[idx]);
                // // Reproject to right image.
                // ceres::CostFunction* reprojectCost = new ceres::AutoDiffCostFunction<ReprojectCostFunction, 2, 6>(new ReprojectCostFunction(0, f->position, f->pixelsR[j], _pCameraModel));
                // problem.AddResidualBlock(reprojectCost, nullptr, _pose[idx]);
            }

            // Show landmarks in image.
            if (f->seenByFrames[j] == curFrameID) {
                Eigen::Vector3d r_j(_pose[actualSize-1][0], _pose[actualSize-1][1], _pose[actualSize-1][2]);
                Eigen::Vector3d p_j(_pose[actualSize-1][3], _pose[actualSize-1][4], _pose[actualSize-1][5]);
                // T_WB = (r_j, p_j), is transformation from body to world frame.
                // T_BW = T_WB.inverse()
                Sophus::SE3d T_BW = Sophus::SE3d(Sophus::SO3d::exp(r_j), p_j).inverse();
                Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * T_BW * f->position;
                Eigen::Matrix<double,4,1> point_homo;
                point_homo << point_wrt_cam(0), point_wrt_cam(1), point_wrt_cam(2), 1;
                Eigen::Vector3d pixel_homo;
                pixel_homo = _pCameraModel->_P_L * point_homo;

                cv::circle(img0, f->pixelsL[j], 3, cv::Scalar(255,0,0));
                cv::circle(img0, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(0,0,255));
            }
        }
    }
    cv::imshow("before optimization", img0);

    // Set up imu cost function.
    int numImuConstraint = _pMap->_imuConstraint.size();
    for (int j = 1; j < actualSize; j++) {
        int i = j - 1;
        ceres::CostFunction* preintegrationCost = new ImuCostFunction(_pImuPreintegrator, _pMap->_imuConstraint[numImuConstraint - actualSize + j]);
        problem.AddResidualBlock(preintegrationCost, nullptr, _pose[i], _v_bga[i], _pose[j], _v_bga[j]);
    }

    // Set the solver.
    ceres::Solver::Options options;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.max_num_iterations = 8;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR; // bundle adjustment problems have a special sparsity structure that can be solved much more efficiently using Schur-based solvers.
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    std::cout << "Before optimizing:\npose: " << _pose[actualSize-1][3] << ", " << _pose[actualSize-1][4] << ", "<< _pose[actualSize-1][5] << std::endl;
    // Run the solver.
    ceres::Solve(options, &problem, &summary);
    std::cout << "After optimizing:\npose: " << _pose[actualSize-1][3] << ", " << _pose[actualSize-1][4] << ", "<< _pose[actualSize-1][5] << std::endl;

    for (int i = 0; i < _pFeatureTracker->_matchedFeatureIDs.size(); i++) {
        cfsd::Ptr<Feature> f = _pFeatureTracker->_features[_pFeatureTracker->_matchedFeatureIDs[i]];
        for (int j = 0; j < f->seenByFrames.size(); j++) {
            // Show landmarks in image.
            if (f->seenByFrames[j] == curFrameID) {
                Eigen::Vector3d r_j(_pose[actualSize-1][0], _pose[actualSize-1][1], _pose[actualSize-1][2]);
                Eigen::Vector3d p_j(_pose[actualSize-1][3], _pose[actualSize-1][4], _pose[actualSize-1][5]);
                // T_WB = (r_j, p_j), is transformation from body to world frame.
                // T_BW = T_WB.inverse()
                Sophus::SE3d T_BW = Sophus::SE3d(Sophus::SO3d::exp(r_j), p_j).inverse();
                Eigen::Vector3d point_wrt_cam = _pCameraModel->_T_CB * T_BW * f->position;
                Eigen::Matrix<double,4,1> point_homo;
                point_homo << point_wrt_cam(0), point_wrt_cam(1), point_wrt_cam(2), 1;
                Eigen::Vector3d pixel_homo;
                pixel_homo = _pCameraModel->_P_L * point_homo;

                cv::circle(img, f->pixelsL[j], 3, cv::Scalar(255,0,0));
                cv::circle(img, cv::Point(pixel_homo(0)/pixel_homo(2), pixel_homo(1)/pixel_homo(2)), 3, cv::Scalar(0,0,255));
            }
        }
    }
    cv::imshow("after optimization", img);
    cv::waitKey(0);

    if (_verbose) {
        // Show the report.
        std::cout << summary.FullReport() << std::endl;
    }

    _pMap->updateStates(_pose, _v_bga);
    _pMap->checkKeyframe();
    _pImuPreintegrator->updateBias();
}

void Optimizer::initialAlignment() {
    double rvec[3] = {0,0,0};
    double ba[3] = {0,0,0};
    double bg[3] = {0,0,0};

    ceres::Problem problem1; // acc initial rotation and bias
    ceres::Problem problem2; // gyr initial bias

    ceres::LocalParameterization* parameterization = new RotationParameterization();
    problem1.AddParameterBlock(rvec, 3, parameterization);
    problem1.AddParameterBlock(ba, 3);
    problem1.SetParameterBlockConstant(ba); // It's hard to optimize both rvec and ba at the same time.
    problem2.AddParameterBlock(bg, 3);

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
    */
    Eigen::Vector3d gravity(-Config::get<double>("gravity"), 0, 0);

    for (int i = 0; i < _accs.size(); i++) {
        // Only consider non-gravitational directions' residual.
        ceres::CostFunction* accCostFunction = new ceres::AutoDiffCostFunction<AccCostFunctor, 3, 3, 3>(new AccCostFunctor(_accs[i], gravity));
        problem1.AddResidualBlock(accCostFunction, NULL, rvec, ba);
        
        ceres::CostFunction* gyrCostFunction = new ceres::AutoDiffCostFunction<GyrCostFunctor, 3, 3>(new GyrCostFunctor(_gyrs[i]));
        problem2.AddResidualBlock(gyrCostFunction, NULL, bg);
    }
        
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    
    ceres::Solver::Summary summary;
    
    ceres::Solve(options, &problem1, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Initial rvec: " << rvec[0] << " " << rvec[1] << " " << rvec[2] << std::endl;
    std::cout << "Initial ba: " << ba[0] << " " << ba[1] << " " << ba[2] << std::endl;

    ceres::Solve(options, &problem2, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Initial bg: " << bg[0] << " " << bg[1] << " " << bg[2] << std::endl;

    // Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Vector3d(rvec[0], rvec[1], rvec[2]));
    // for (int i = 0; i < _accs.size(); i++) {
    //     std::cout << "raw acc:\n" << _accs[i] << std::endl;
    //     std::cout << "rotated acc:\n" << R * _accs[i] << std::endl;
    // }

    _pImuPreintegrator->setInitialStates(bg, rvec);

    _accs.clear();
    _gyrs.clear();
}

} // namespace cfsd