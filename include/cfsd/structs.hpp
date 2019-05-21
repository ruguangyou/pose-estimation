#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include "cfsd/common.hpp"

namespace cfsd {

struct ImuConstraint {
    ImuConstraint() {}
    
    ImuConstraint(const Eigen::Matrix<double,15,15>& invCovPreintegration_ij, const Eigen::Vector3d& bg_i, const Eigen::Vector3d& ba_i,
                  const Sophus::SO3d& delta_R_ij, const Eigen::Vector3d& delta_v_ij, const Eigen::Vector3d& delta_p_ij,
                  const Eigen::Matrix3d& d_R_bg_ij, const Eigen::Matrix3d& d_v_bg_ij, const Eigen::Matrix3d& d_v_ba_ij, 
                  const Eigen::Matrix3d& d_p_bg_ij, const Eigen::Matrix3d& d_p_ba_ij, const double& dt) : 
        invCovPreintegration_ij(invCovPreintegration_ij), bg_i(bg_i), ba_i(ba_i), delta_R_ij(delta_R_ij), delta_v_ij(delta_v_ij), delta_p_ij(delta_p_ij),
        d_R_bg_ij(d_R_bg_ij), d_v_bg_ij(d_v_bg_ij), d_v_ba_ij(d_v_ba_ij), d_p_bg_ij(d_p_bg_ij), d_p_ba_ij(d_p_ba_ij), dt(dt) { dt2 = dt* dt; }

    // Inverse of covariance matrix of preintegrated noise [delta_rvec, delta_v, delta_p, delta_bg, delta_ba]
    Eigen::Matrix<double,15,15> invCovPreintegration_ij{};

    // Bias of gyroscope and accelerometer at time i.
    Eigen::Vector3d bg_i{};
    Eigen::Vector3d ba_i{};
    
    // Preintegrated delta_R, delta_v, delta_p.
    Sophus::SO3d delta_R_ij{};
    Eigen::Vector3d delta_v_ij{};
    Eigen::Vector3d delta_p_ij{};

    // Partial derivative of R, v, p with respect to bias of gyr and acc (denoated as bg and ba).
    Eigen::Matrix3d d_R_bg_ij{};
    Eigen::Matrix3d d_v_bg_ij{};
    Eigen::Matrix3d d_v_ba_ij{};
    Eigen::Matrix3d d_p_bg_ij{};
    Eigen::Matrix3d d_p_ba_ij{};

    // The time between two camera frames, dt2 = dt^2.
    double dt{0}, dt2{0};
};

struct Feature {
    Feature() {}

    Feature(const int& frameID, const cv::Point2d& pixelL, const cv::KeyPoint& keypointL, const cv::KeyPoint& keypointR, const cv::Mat& descriptorL, const cv::Mat& descriptorR, const int& age)
      : frameID(frameID), pixelL(pixelL), keypointL(keypointL), keypointR(keypointR), descriptorL(descriptorL), descriptorR(descriptorR), age(age) {}

    int frameID{0};

    cv::Point2d pixelL{};

    cv::KeyPoint keypointL{};
    cv::KeyPoint keypointR{};
    cv::Mat descriptorL{};
    cv::Mat descriptorR{};

    int age{0};
};

struct MapPoint {
    MapPoint() {}

    MapPoint(const Eigen::Vector3d& position, const int& frameID, const cv::Point2d& pixel) : position(position) {
        addPixel(frameID, pixel);
    }

    void addPixel(const int& frameID, const cv::Point2d& pixel) {
        pixels[frameID] = pixel;
    }

    // 3D position w.r.t world coordinate.
    Eigen::Vector3d position{};

    // frameID and pixel coordinate in that frame.
    std::map<int, cv::Point2d> pixels{};
};

struct Keyframe {
    Keyframe() : R(Eigen::Matrix3d::Identity()), p(Eigen::Vector3d::Zero()), v(Eigen::Vector3d::Zero()), dbg(Eigen::Vector3d::Zero()), dba(Eigen::Vector3d::Zero()) {}

    Sophus::SO3d R;
    Eigen::Vector3d p;
    Eigen::Vector3d v;
    Eigen::Vector3d dbg;
    Eigen::Vector3d dba;
    cfsd::Ptr<ImuConstraint> pImuConstraint{};
    std::vector<size_t> mapPointIDs{};
    cv::Mat descriptors{};
    long timestamp{0};
};

struct LoopInfo {
    LoopInfo() {}
    
    LoopInfo(const int& loopFrameID, const Sophus::SO3d& R, const Eigen::Vector3d& p) : loopFrameID(loopFrameID), R(R), p(p) {}

    int loopFrameID{-1};
    // R and p are transform from loop frame to current frame.
    Sophus::SO3d R{};
    Eigen::Vector3d p{};
};

} // namespace cfsd

#endif // STRUCTS_HPP
