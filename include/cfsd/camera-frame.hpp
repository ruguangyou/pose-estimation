#ifndef CAMERA_FRAME_HPP
#define CAMERA_FRAME_HPP

#include "cfsd/common.hpp"

namespace cfsd {

class CameraFrame {
    // camera intrinsic parameters
    static Eigen::Matrix3d _camLeft, _camRight;
    static Eigen::Vector3d _distLeft, _distRight;
    // camera extrinsic parameters
    static Sophus::SE3d _right2left; // right camera to left
    static Sophus::SE3d _left2imu;   // left camera to imu

  public:
    using Ptr = std::shared_ptr<CameraFrame>;
    // constructor and deconstructor
    CameraFrame();
    CameraFrame(cv::Mat img, );
    ~CameraFrame();

    // factory function
    static CameraFrame::Ptr create();

    // setting camera parameters
    static void setIntrinsics();
    static void setExtrinsics();

    // image pre-processing
    void removeSky();
    void undistort();
    // void rectify();
    

  private:
    unsigned long _id;
    double _timestamp;
    // left and right images from stereo camera
    cv::Mat _imgLeft, _imgRight;
    // left camera pose (rotation, translation)
    Sophus::SO3d _SO3; // rotation, SO3<double>
    Eigen::Vector3d _t; // translation
    Sophus::SE3d _SE3; // pose

  public:
    // getter funtions
    inline const cv::Mat& getImgLeft() const { return _imgLeft; }
    inline const cv::Mat& getImgRight() const { return _imgRight; }
    inline const Eigen::Matrix3d& getCamLeft() const { return _camLeft; }
    inline const Eigen::Vector3d& getDistLeft() const { return _distLeft; }
    inline const Eigen::Matrix3d& getCamRight() const { return _camRight; }
    inline const Eigen::Vector3d& getDistRight() const { return _distRight; }
    inline const Eigen::Matrix<double,3,4>& getRightToLeft() const { return _right2left.matrix3x4(); }

    inline Eigen::Vector3d getCameraCenter() const; // the coordinated is relative to left camera or imu??

    // coordinates transformation
    /* coordinate system convension defined here:
        imu and camera, take imu coordinates as reference
        the stereo camera, take left camera as reference
        (so for right camera point, it is first transformed to left camera
        coordinate using extrinsic parameters, then to imu coordinate) */
    inline Eigen::Vector2d camLeft2pixel (const Eigen::Vector3d& p_c);
    inline Eigen::Vector2d camRight2pixel(const Eigen::Vector3d& p_c);
    inline Eigen::Vector3d pixel2camLeft (const Eigen::Vector2d& p_p, const double depth);
    inline Eigen::Vector3d pixel2camRight(const Eigen::Vector2d& p_p, const double depth);
    inline Eigen::Vector3d camLeft2world (const Eigen::Vector2d& p_c, const Sophus::SE3d& T_c_w, const double depth);
    inline Eigen::Vector3d world2camLeft (const Eigen::Vector3d& p_w, const Sophus::SE3d& T_c_w);
    inline Eigen::Vector3d camLeft2camRight(const Eigen::Vector3d& p_c);
    inline Eigen::Vector3d camRight2camLeft(const Eigen::Vector3d& p_c);
    inline Eigen::Vector3d camLeft2imu(const Eigen::Vector3d& p_c, const Sophus::SE3d& T_c_i);
    inline Eigen::Vector3d imu2camLeft(const Eigen::Vector3d& p_i, const Sophus::SE3d& T_c_i);

};

} // namespace cfsd

#endif // CAMERA_FRAME_HPP
