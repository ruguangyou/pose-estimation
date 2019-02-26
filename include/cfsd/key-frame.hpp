#ifndef KEY_FRAME_HPP
#define KEY_FRAME_HPP

#include "cfsd/common.hpp"
#include "cfsd/camera-frame.hpp"
// #include "cfsd/imu-frame.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/feature2d/feature2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cfsd {

// only store key frames, and use them to perform optimization
// key frame selection is based on camera keypoints matching percentage
class KeyFrame {
  public:
    using Ptr = std::shared_ptr<KeyFrame>;

    KeyFrame();
    ~KeyFrame();

    // factoty function
    static KeyFrame::Ptr create();

    // feature matching between left and right images
    // triangulate to get 3D points corresponding to matched keypoints
    void matchAndTriangulate();

  private:
    unsigned long _id;
    double _timestamp;
    bool _verbose;

    // camera and imu frames
    CameraFrame::Ptr _camFrame;
    // ImuFrame::Ptr _imuFrame;

    // keypoints, descriptors and 3D points
    std::vector<cv::KeyPoint> _camKeypoints;  // left camera
    cv::Mat _camDescriptors;
    std::vector<cv::Point3f> _points3D; // 3D points related to keypoints

    // pose
    Sophus::SE3d _SE3CamLeft, _SE3Imu;

  public:
    // getter functions
    inline const std::vector<cv::KeyPoint>& getCamKeypoints() const { return _camKeypoints; }
    inline const cv::Mat& getDescriptors() const { return _camDescriptors; }
    inline const std::vector<cv::Point3f>& getPoints3D() const { return _points3D; }
    inline const Eigen::Matrix<double,3,4>& getCamLeftPose() const { return _SE3CamLeft.matrix3x4(); }

    void setCamPose(Sophus::SE3d camPose);
    
};

} // namespace cfsd

#endif // KEY_FRAME_HPP