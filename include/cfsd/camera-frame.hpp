#ifndef CAMERA_FRAME_HPP
#define CAMERA_FRAME_HPP

#include "cfsd/common.hpp"

namespace cfsd {

class CameraFrame {
    // camera intrinsic parameters
    static Eigen::Matrix3d _camLeft, _camRight;
    static Eigen::Vector3d _distLeft, _distRight;
    // camera extrinsic parameters
    static Sophus::SE3d _right2left;
    static Sophus::SE3d _left2right;
    // camera & imu extrinsics
    // static Sophus::SE3d _left2imu; // left camera to imu

  public:
    using Ptr = std::shared_ptr<CameraFrame>;

    // constructor and deconstructor
    CameraFrame();
    ~CameraFrame();

    CameraFrame(long id, double timestamp, cv::Mat img);

    // factory function
    static CameraFrame::Ptr create(double timestamp, cv::Mat img);

    // setting camera parameters
    static void setIntrinsics(cv::Mat& camLeft, cv::Mat& distLeft, cv::Mat& camRight, cv::Mat& distRight);
    static void setExtrinsics(cv::Mat& R, cv::Mat& t);

    // image pre-processing
    // void extractROI();
    void undistort();
    // void rectify();
    

  private:
    unsigned long _id;
    double _timestamp;
    // left and right images from stereo camera
    cv::Mat _imgLeft, _imgRight;

  public:
    // getter funtions
    inline double getTimestamp() const { return _timestamp; }
    inline const cv::Mat& getImgLeft() const { return _imgLeft; }
    inline const cv::Mat& getImgRight() const { return _imgRight; }
    inline const Eigen::Matrix3d& getCamLeft() const { return _camLeft; }
    inline const Eigen::Vector3d& getDistLeft() const { return _distLeft; }
    inline const Eigen::Matrix3d& getCamRight() const { return _camRight; }
    inline const Eigen::Vector3d& getDistRight() const { return _distRight; }
    inline const Sophus::SE3d& getRightToLeft() const { return _right2left; }
    inline const Sophus::SE3d& getLeftToRight() const { return _left2right; }

    // inline Eigen::Vector3d getCameraCenter() const; // the coordinated is relative to left camera or imu??

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
    // inline Eigen::Vector3d camLeft2imu(const Eigen::Vector3d& p_c, const Sophus::SE3d& T_c_i);
    // inline Eigen::Vector3d imu2camLeft(const Eigen::Vector3d& p_i, const Sophus::SE3d& T_c_i);

};

} // namespace cfsd

#endif // CAMERA_FRAME_HPP
