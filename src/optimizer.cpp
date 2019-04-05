#include "cfsd/optimizer.hpp"

namespace cfsd {

struct ImuCostFunction : public ceres::SizedCostFunction<15, /* residuals */
                                                          6, /* pose (r, p) at time i */
                                                          9, /* velocity, bias of gyr and acc at time i */
                                                          6, /* pose (r, p) at time j */
                                                          9  /* velocity, bias of gyr and acc at time j */> {
    ImuCostFunction(const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const ImuConstraint& ic) : _pImuPreintegrator(pImuPreintegrator), _ic(ic) {}

    virtual ~ImuCostFunction() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // parameters: rvp_i, bga_i, rvp_j, bga_j
        // rvp: [rx,ry,rz, vx,vy,vz, px,py,pz]

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
    ImuConstraint _ic;
};

struct PoseParameterization : public ceres::LocalParameterization {
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // input: pose [r,p]
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
        residuals[1] = pixel_homo(1)/pixel_homo(2) - T(u);
        
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

// ############################################################

Optimizer::Optimizer(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<ImuPreintegrator>& pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose)
    : _pMap(pMap), _pImuPreintegrator(pImuPreintegrator), _pCameraModel(pCameraModel), _verbose(verbose) {}

Optimizer::~Optimizer() { 
    delete _pose[WINDOWSIZE];
    delete _v_bga[WINDOWSIZE];
}

void Optimizer::motionOnlyBA(std::unordered_map<size_t,Feature>& features, const std::vector<size_t>& matchedFeatureIDs) {
    // For the first few frames, the local window cannot fit, i.e. actualSize < WINDOWSIZE
    int actualSize = _pMap->getStates((double**)_pose, (double**)_v_bga);

    // Build the problem.
    ceres::Problem problem;

    // Add parameter block with parameterization.
    ceres::LocalParameterization* poseParameterization = new PoseParameterization();
    for (int i = 0; i < actualSize; i++) {
        problem.AddParameterBlock(_pose[i], 6, poseParameterization);
        problem.AddParameterBlock(_v_bga[i], 9);
    }

    // Current frame ID.
    int curFrameID = features[matchedFeatureIDs[0]].seenByFrames.back(); 

    // Set up reprojection cost function (a.k.a. residuals).
    for (int i = 0; i < matchedFeatureIDs.size(); i++) {
        Feature& f = features[matchedFeatureIDs[i]];
        for (int j = 0; j < f.seenByFrames.size(); j++) {
            // Check if the frame is within the local window.
            int idx = f.seenByFrames[j] - curFrameID + actualSize - 1;
            if (idx >= 0) {
                // Reproject to left image.
                ceres::CostFunction* reprojectCost = new ceres::AutoDiffCostFunction<ReprojectCostFunction, 2, 6>(new ReprojectCostFunction(1, f.position, f.pixelsL[j], _pCameraModel));
                problem.AddResidualBlock(reprojectCost, nullptr, _pose[idx]);

                // // Reproject to right image.
                // ceres::CostFunction* reprojectCost = new ceres::AutoDiffCostFunction<ReprojectCostFunction, 2, 6>(new ReprojectCostFunction(0, f.position, f.pixelsR[j], _pCameraModel));
                // problem.AddResidualBlock(reprojectCost, nullptr, _pose[idx]);
            }
        }
    }

    // Set up imu cost function.
    int numImuConstraint = _pMap->_imuConstraint.size();
    for (int j = 1; j < actualSize; j++) {
        int i = j - 1;
        ceres::CostFunction* costFunction = new ImuCostFunction(_pImuPreintegrator, _pMap->_imuConstraint[numImuConstraint - actualSize + j]);
        problem.AddResidualBlock(costFunction, nullptr, _pose[i], _v_bga[i], _pose[j], _v_bga[j]);
    }

    // Set the solver.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; // bundle adjustment problems have a special sparsity structure that can be solved much more efficiently using Schur-based solvers.
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    // Run the solver.
    ceres::Solve(options, &problem, &summary);
    
    // if (_verbose) {
        // Show the report.
        std::cout << summary.BriefReport() << std::endl;
    // }

    _pMap->checkKeyframe();
    _pMap->updateStates((double**)_pose, (double**)_v_bga);
}


// // Assume vehicle state at time i is identity R and zero v, p, bg, ba,
// // and given the relative transformation from i to j,
// // optimize the state at time j.
// void Optimizer::localOptimize() {
//     // Parameters to be optimized: rotation, velocity, position, deltaBiasGyr, deltaBiasAcc
//     double rvp_i[9]; // r, v, p at time i
//     double bga_i[6]; // bg, ba at time i
//     double rvp_j[9]; // r, v, p at time j
//     double bga_j[6] = {0,0,0,0,0,0}; // bg, ba at time j

//     // Read initial values from Map.
//     _pMap->getParameters(rvp_i, bga_i, rvp_j);

//     #ifdef USE_VIEWER
//     _pViewer->setRawParameters(rvp_i, rvp_j);
//     #endif

//     // Build the problem.
//     ceres::Problem problem;

//     // Add parameter block with parameterization.
//     ceres::LocalParameterization* imuParameterization = new ImuParameterization();
//     problem.AddParameterBlock(rvp_i, 9, imuParameterization);
//     problem.AddParameterBlock(bga_i, 6);
//     problem.AddParameterBlock(rvp_j, 9, imuParameterization);
//     problem.AddParameterBlock(bga_j, 6);

//     // Set up cost function (a.k.a. residuals).
//     ceres::CostFunction* costFunction = new ImuCostFunction(_pMap);
//     problem.AddResidualBlock(costFunction, nullptr, rvp_i, bga_i, rvp_j, bga_j);

//     // Set the solver.
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_QR; // todo...
//     options.minimizer_progress_to_stdout = true;
//     ceres::Solver::Summary summary;

//     // Run the solver.
//     ceres::Solve(options, &problem, &summary);
    
//     // if (_verbose) {
//         // Show the report.
//         std::cout << summary.BriefReport() << std::endl;
//         // std::cout << summary.FullReport() << std::endl;
//         #ifdef DEBUG_IMU
//         std::cout << "Optimized velocity i: " << rvp_i[3] << ", " << rvp_i[4] << ", " << rvp_i[5] << std::endl;
//         std::cout << "Optimized position i: " << rvp_i[6] << ", " << rvp_i[7] << ", " << rvp_i[8] << std::endl;
//         std::cout << "Optimized bg i: " << bga_i[0] << ", " << bga_i[1] << ", " << bga_i[2] << std::endl;
//         std::cout << "Optimized ba i: " << bga_i[0] << ", " << bga_i[1] << ", " << bga_i[2] << std::endl;
//         std::cout << "Optimized velocity j: " << rvp_j[3] << ", " << rvp_j[4] << ", " << rvp_j[5] << std::endl;
//         std::cout << "Optimized position j: " << rvp_j[6] << ", " << rvp_j[7] << ", " << rvp_j[8] << std::endl;
//         std::cout << "Optimized bg j: " << bga_j[0] << ", " << bga_j[1] << ", " << bga_j[2] << std::endl;
//         std::cout << "Optimized ba j: " << bga_j[0] << ", " << bga_j[1] << ", " << bga_j[2] << std::endl;
//         #endif
//     // }

//     #ifdef USE_VIEWER
//     _pViewer->setParameters(rvp_i, rvp_j);
//     #endif

//     // Update state values in ImuPreintegrator.
//     _pImuPreintegrator->updateState(rvp_j, bga_j);

//     // Todo: push back the locally optimized states to map for global optimization later.
// }


} // namespace cfsd