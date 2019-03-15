#ifndef KEY_FRAME_HPP
#define KEY_FRAME_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cfsd/camera-frame.hpp"
// #include "cfsd/imu-frame.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cfsd {

// only store key frames, and use them to perform optimization
// key frame selection is based on camera keypoints matching percentage
class KeyFrame {
  public:
    using Ptr = std::shared_ptr<KeyFrame>;

    KeyFrame();
    ~KeyFrame();
    KeyFrame(long id, long timestamp, CameraFrame::Ptr camFrame, bool verbose, bool debug);

    // factoty function
    static KeyFrame::Ptr create(long timestamp, CameraFrame::Ptr camFrame, bool verbose, bool debug);

    // feature matching between left and right images
    // triangulate to get 3D points corresponding to matched keypoints
    void matchAndTriangulate();

  private:
    unsigned long _id;
    long _timestamp;
    bool _verbose, _debug;

    // camera and imu frames
    CameraFrame::Ptr _camFrame;
    // ImuFrame::Ptr _imuFrame;

    // feature detector and matching parameters
    cv::Ptr<cv::ORB> _orb;
    float _matchRatio; // ratio for selecting good matches
    float _minMatchDist; // min match distance, based on experience, e.g. 30.0f

    // keypoints, descriptors and 3D points
    std::vector<cv::KeyPoint> _camKeypoints; // left camera
    cv::Mat _camDescriptors;
    std::vector<cvPoint3Type> _points3D; // 3D points related to keypoints

    // pose
    SophusSE3Type _SE3CamLeft, _SE3CamRight; // _SE3Imu;

  public:
    // getter functions
    inline const std::vector<cv::KeyPoint>& getCamKeypoints() const { return _camKeypoints; }
    inline const cv::Mat& getDescriptors() const { return _camDescriptors; }
    inline const std::vector<cvPoint3Type>& getPoints3D() const { return _points3D; }
    inline const Eigen::Matrix<precisionType,3,4> getCamLeftPose() const { return _SE3CamLeft.matrix3x4(); }
    inline size_t getNumOfPoints() const { return _points3D.size(); }

    void setCamPose(SophusSE3Type camPose);
    
};

} // namespace cfsd

#endif // KEY_FRAME_HPP