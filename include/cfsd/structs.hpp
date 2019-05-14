#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include "cfsd/common.hpp"

namespace cfsd {

struct Feature {
    Feature() : pixelL(), descriptorL(), descriptorR(), position() {}
    Feature(const cv::Point2d& pixelL, const cv::Mat& descriptorL, const cv::Mat& descriptorR, const Eigen::Vector3d& position, const int& age)
      : pixelL(pixelL), descriptorL(descriptorL), descriptorR(descriptorR), position(position), age(age) {}

    // Pixel coordinate of this feature in each frame.
    cv::Point2d pixelL;
    
    // Feature descriptor, comes from the earliest frame.
    cv::Mat descriptorL;
    cv::Mat descriptorR;

    // 3D landmark position w.r.t world frame.
    Eigen::Vector3d position;

    int age{0};
};

struct ImuConstraint {
    ImuConstraint() : invCovPreintegration_ij(), bg_i(), ba_i(), delta_R_ij(), delta_v_ij(), delta_p_ij(), d_R_bg_ij(), d_v_bg_ij(), d_v_ba_ij(), d_p_bg_ij(), d_p_ba_ij() {}
    ImuConstraint(const Eigen::Matrix<double,15,15>& invCovPreintegration_ij, const Eigen::Vector3d& bg_i, const Eigen::Vector3d& ba_i,
                  const Sophus::SO3d& delta_R_ij, const Eigen::Vector3d& delta_v_ij, const Eigen::Vector3d& delta_p_ij,
                  const Eigen::Matrix3d& d_R_bg_ij, const Eigen::Matrix3d& d_v_bg_ij, const Eigen::Matrix3d& d_v_ba_ij, 
                  const Eigen::Matrix3d& d_p_bg_ij, const Eigen::Matrix3d& d_p_ba_ij, const double& dt) : 
        invCovPreintegration_ij(invCovPreintegration_ij), bg_i(bg_i), ba_i(ba_i), delta_R_ij(delta_R_ij), delta_v_ij(delta_v_ij), delta_p_ij(delta_p_ij),
        d_R_bg_ij(d_R_bg_ij), d_v_bg_ij(d_v_bg_ij), d_v_ba_ij(d_v_ba_ij), d_p_bg_ij(d_p_bg_ij), d_p_ba_ij(d_p_ba_ij), dt(dt) { dt2 = dt* dt; }

    // Inverse of covariance matrix of preintegrated noise [delta_rvec, delta_v, delta_p, delta_bg, delta_ba]
    Eigen::Matrix<double,15,15> invCovPreintegration_ij;

    // Bias of gyroscope and accelerometer at time i.
    Eigen::Vector3d bg_i, ba_i;
    
    // Preintegrated delta_R, delta_v, delta_p.
    Sophus::SO3d delta_R_ij;
    Eigen::Vector3d delta_v_ij;
    Eigen::Vector3d delta_p_ij;

    // Partial derivative of R, v, p with respect to bias of gyr and acc (denoated as bg and ba).
    Eigen::Matrix3d d_R_bg_ij;
    Eigen::Matrix3d d_v_bg_ij;
    Eigen::Matrix3d d_v_ba_ij;
    Eigen::Matrix3d d_p_bg_ij;
    Eigen::Matrix3d d_p_ba_ij;

    // The time between two camera frames, dt2 = dt^2.
    double dt{0}, dt2{0};
};

struct MapPoint {
    MapPoint() : pixel(), position() {}
    MapPoint(const size_t id, const cv::Point2d& pixel, const Eigen::Vector3d& position) : id(id), pixel(pixel), position(position) {}

    size_t id{0};
    
    cv::Point2d pixel;
    
    Eigen::Vector3d position;
};

struct Keyframe {
    Keyframe() : R(), p(Eigen::Vector3d::Zero()), v(Eigen::Vector3d::Zero()), dbg(Eigen::Vector3d::Zero()), dba(Eigen::Vector3d::Zero()), pImuConstraint(), points(), descriptors() {}
    // Keyframe(const Sophus::SO3d& R, const Eigen::Vector3d& p, const Eigen::Vector3d& v, const Eigen::Vector3d& dbg, const Eigen::Vector3d& dba, 
    //          const cfsd::Ptr<ImuConstraint>& pImuConstraint, const std::vector<cfsd::Ptr<MapPoint>>&  points, const long& timestamp) :
    //     R(R), p(p), v(v), dbg(dbg), dba(dba), pImuConstraint(pImuConstraint), points(points), timestamp(timestamp) {}

    Sophus::SO3d R;
    Eigen::Vector3d p;
    Eigen::Vector3d v;
    Eigen::Vector3d dbg;
    Eigen::Vector3d dba;
    cfsd::Ptr<ImuConstraint> pImuConstraint;
    std::vector<cfsd::Ptr<MapPoint>> points;
    cv::Mat descriptors;
    long timestamp{0};
};

} // namespace cfsd

#endif // STRUCTS_HPP
