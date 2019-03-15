#ifndef FEATURE_TRACKER_HPP
#define FEATURE_TRACKER_HPP

#include "cfsd/common.hpp"
#include "cfsd/camera-model.hpp"
// #include "cfsd/key-frame.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cfsd {

struct Feature {
    Feature() {}

    Feature(const int& frameCount, const cvPoint2Type& pixelL, const cvPoint2Type& pixelR, const cv::Mat& descriptorL, const cv::Mat& descriptorR)
      : _matchedTimes(0), _pixelL(pixelL), _pixelR(pixelR), _descriptorL(descriptorL), _descriptorR(descriptorR) {
        _seenByFrames.push_back(frameCount);    
    }

    int _matchedTimes;

    // ID of frames which can observe this feature (useful to know this when doing reprojection)
    std::vector<int> _seenByFrames;

    cvPoint2Type _pixelL;
    cvPoint2Type _pixelR;
    
    cv::Mat _descriptorL;
    cv::Mat _descriptorR;
};

class FeatureTracker {
  public:
    enum DetectorType {
        ORB = 0,
        BRISK = 1
    };
  
    FeatureTracker(const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose);

    // Feature matching and tracking, including:
    // - internal match (current frame's left and right image)
    // - external track (current features and past features)
    // - refinement? (improve the quality of matching)
    void process(const cv::Mat& imgLeft, const cv::Mat& imgRight);

    void internalMatch(const cv::Mat& imgLeft, const cv::Mat& imgRight, std::vector<cvPoint2Type>& curPixelsL, std::vector<cvPoint2Type>& curPixelsR, cv::Mat& curDescriptorsL, cv::Mat& curDescriptorsR);

    void externalTrack(const std::vector<cvPoint2Type>& curPixelsL, const std::vector<cvPoint2Type>& curPixelsR, const cv::Mat& curDescriptorsL, const cv::Mat& curDescriptorsR);

    void featurePoolUpdate();

    // TODO:
    // RANSAC remove outliers?
    // undistort image (mask needs to be updated)
    // triangulation (don't change the sign of homogeneous coordinates)
    // far and near points (based on the depth)
    
    // [Update] decide not to do so for the sake of computational efficiency, instead using estimation from IMU and performing optimization.
    // Compute camera pose: shoule use RANSAC scheme for outlier rejection, and solve 3D-2D PnP problem (in particular, P3P problem).
    // void computeCamPose(SophusSE3Type& pose);

  private:
    bool _verbose;

    // Camera Model.
    cfsd::Ptr<CameraModel> _pCameraModel;

    // Feature ID.
    size_t _featureCount;
    
    // Frame ID.
    int _frameCount;

    // Detector to be used (ORB, BRISK, ...)
    DetectorType _detectorType;
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::BRISK> _brisk;

    // Features that pass circular matching, i.e. curLeft <=> histLeft <=> histRight <=> curRight <=> curLeft
    // store the id of the matched features, s.t. the _matchedTimes could be easily updated;
    // also store the id of the not matches old features, s.t. they could be removed.
    std::vector<size_t> _matchedFeatureIDs, _notMatchedFeatureIDs;

    
    // Features in the current frame that have no match with history features, will be added into the feature pool.
    std::vector<cvPoint2Type> _newPixelsL, _newPixelsR; 
    cv::Mat _newDescriptorsL, _newDescriptorsR;

    // Only part of the image is considered to be useful (e.g. the upper half of the image containing sky contributes little to useful features)
    cv::Mat _maskL, _maskR;

    // Match distance should be less than max(_matchRatio*minDist, _minMatchDist)
    // Ratio for selecting good matches.
    float _matchRatio;  
    // Min match distance, based on experience, e.g. 30.0f
    float _minMatchDist;

    // Max times a feature coule be matched, i.e., max times a feature could be seen by different frames.
    int _maxMatchedTimes;

    // History features' id and descriptors
    std::vector<size_t> _histFeatureIDs;
    cv::Mat _histDescriptorsL, _histDescriptorsR;

    // Available features from history frames.
    // - new features would be added
    // - matched features' _matchedTimes will be updated
    // - old features that are not useful anymore would be removed
    // so std::map container is choosed due to the efficient access, insert and erase operation.
    std::map<size_t, Feature> _features;

    // If the image is cropped, the pixel coordinate of keypoints would be different with the uncropped ones,
    // it would cause dismatching between 3D points and 2D pixels when doing projection.
    // int _cropOffset;

};

} // namespace cfsd

#endif // FEATURE_TRACKER_HPP