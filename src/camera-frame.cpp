#include "cfsd/camera-frame.hpp"

namespace cfsd {

// default construtor
CameraFrame::CameraFrame() {}

CameraFrame::CameraFrame(cv::Mat img) : {}

CameraFrame::~CameraFrame() {}

CameraFrame::Ptr CameraFrame::create() {

}

void CameraFrame::setIntrinsics() {

}

void CameraFrame::setExtrinsics() {

}

// image pre-processing
void CameraFrame::removeSky() {

}



// coordinate tranformation
Eigen::Vector2d CameraFrame::camLeft2pixel(const Eigen::Vector3d& p_c) {
    return Eigen::Vector2d(p_c(0,0) / p_c(2,0) * _camLeft(0,0) + _camLeft(0,2),
                           p_c(1,0) / p_c(2,0) * _camLeft(1,1) + _camLeft(1,2));
}

Eigen::Vector2d CameraFrame::camRight2pixel(const Eigen::Vector3d& p_c) {
    return Eigen::Vector2d(p_c(0,0) / p_c(2,0) * _camRight(0,0) + _camRight(0,2),
                           p_c(1,0) / p_c(2,0) * _camRight(1,1) + _camRight(1,2));
}

Eigen::Vector3d CameraFrame::pixel2camLeft(const Eigen::Vector2d& p_p, const double depth) {
    return Eigen::Vector3d((p_p(0,0) - _camLeft(0,2)) / _camLeft(0,0) * depth,
                           (p_p(1,0) - _camLeft(1,2)) / _camLeft(1,1) * depth,
                           depth);
}

Eigen::Vector3d CameraFrame::pixel2camRight(const Eigen::Vector2d& p_p, const double depth) {
    return Eigen::Vector3d((p_p(0,0) - _camRight(0,2)) / _camRight(0,0) * depth,
                           (p_p(1,0) - _camRight(1,2)) / _camRight(1,1) * depth,
                           depth);
}

Eigen::Vector3d CameraFrame::imu2world(const Eigen::Vector2d& p_c, const Sophus::SE3d& T_c_w) {
    // T_c_w is the transform from left camera to world
    return T_c_w * p_c;
}

Eigen::Vector2d CameraFrame::world2imu(const Eigen::Vector3d& p_w, const Sophus::SE3d& T_c_w) {
    return T_c_w.inverse() * p_w;
}

Eigen::Vector3d CameraFrame::camRight2camLeft(const Eigen::Vector3d& p_c) {
    return _right2left.inverse() * p_c;
}

Eigen::Vector3d CameraFrame::camRight2camLeft(const Eigen::Vector3d& p_c) {
    return _right2left * p_c;
}

Eigen::Vector3d CameraFrame::camLeft2imu(const Eigen::Vector3d& p_c) {
    return _left2imu * p_c;
}

Eigen::Vector3d CameraFrame::imu2camLeft(const Eigen::Vector3d& p_i) {
    return _left2imu.inverse() * p_i;
}


} // namespace cfsd

