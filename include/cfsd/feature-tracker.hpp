#ifndef FEATURE_TRACKER_HPP
#define FEATURE_TRACKER_HPP

#include "cfsd/common.hpp"
#include "cfsd/camera-model.hpp"
#include "cfsd/structs.hpp"
#include "cfsd/optimizer.hpp"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cfsd {

class FeatureTracker {
  public:
    enum DetectorType {
        ORB = 0,
        BRISK = 1
    };
  
    FeatureTracker(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<Optimizer>& pOptimizer, const cfsd::Ptr<ImuPreintegrator> pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose);

    // Feature matching and tracking, including:
    // - internal match (current frame's left and right image)
    // - external track (current features and past features)
    // - refinement? (improve the quality of matching)
    void process(const cv::Mat& imgLeft, const cv::Mat& imgRight);

    void internalMatch(const cv::Mat& imgLeft, const cv::Mat& imgRight, std::vector<cv::Point2d>& curPixelsL, std::vector<cv::Point2d>& curPixelsR, cv::Mat& curDescriptorsL, cv::Mat& curDescriptorsR, const bool useRANSAC = true);

    void externalTrack(const std::vector<cv::Point2d>& curPixelsL, const std::vector<cv::Point2d>& curPixelsR, const cv::Mat& curDescriptorsL, const cv::Mat& curDescriptorsR, std::vector<bool>& curFeatureMask, const bool useRANSAC = true);

    void featurePoolUpdate(const std::vector<cv::Point2d>& curPixelsL, const std::vector<cv::Point2d>& curPixelsR, const cv::Mat& curDescriptorsL, const cv::Mat& curDescriptorsR, const std::vector<bool>& curFeatureMask);

    // TODO:
    // RANSAC remove outliers?
    // undistort image (mask needs to be updated)
    // triangulation (don't change the sign of homogeneous coordinates)
    // far and near points (based on the depth)
    
    // [Update] decide not to do so for the sake of computational efficiency, instead using estimation from IMU and performing optimization.
    // Compute camera pose: shoule use RANSAC scheme for outlier rejection, and solve 3D-2D PnP problem (in particular, P3P problem).
    // void computeCamPose(Sophus::SE3d& pose);

  private:
    bool _verbose;

    // Pinhole camera Model.
    cfsd::Ptr<CameraModel> _pCameraModel;

    cfsd::Ptr<Optimizer> _pOptimizer;

    cfsd::Ptr<ImuPreintegrator> _pImuPreintegrator;

    cfsd::Ptr<Map> _pMap;

    // Feature ID.
    size_t _featureCount;
    
    // Frame ID.
    int _frameCount;

    // Detector to be used (ORB, BRISK, ...)
    DetectorType _detectorType;
    cv::Ptr<cv::ORB> _orb;
    cv::Ptr<cv::BRISK> _brisk;

    // Features that pass circular matching, i.e. curLeft <=> histLeft <=> histRight <=> curRight <=> curLeft
    // store the id of the circularly matched features, s.t. the _age grows normally, i.e., increase 1.
    // also store the id of the matches (either left or right side) but not circularly matched features, s.t. the _age will grow more than 1 as penalty.
    // for those not matched features, the _age will grow much more as penalty.
    std::vector<size_t> _matchedFeatureIDs;

    // Current features matched with history features.
    // std::vector<cv::Point2d> _matchedHistPixelsL, _matchedHistPixelsR;
    // std::vector<cv::Point2d> _matchedCurPixelsL, _matchedCurPixelsR;

    // Only part of the image is considered to be useful (e.g. the upper half of the image containing sky contributes little to useful features)
    cv::Mat _maskL, _maskR;

    // Match distance should be less than max(_matchRatio*minDist, _minMatchDist)
    // Ratio for selecting good matches.
    float _matchRatio;  
    // Min match distance, based on experience, e.g. 30.0f
    float _minMatchDist;

    int _maxFeatureAge;

    // History features' id and descriptors
    std::vector<size_t> _histFeatureIDs;
    cv::Mat _histDescriptorsL, _histDescriptorsR;

    // Available features from history frames.
    // - new features would be added
    // - matched features' _age will be updated
    // - old features that are not useful anymore would be removed
    // so std::map container is choosed due to the efficient access, insert and erase operation.
    std::unordered_map<size_t, Feature> _features;

    // If the image is cropped, the pixel coordinate of keypoints would be different with the uncropped ones,
    // it would cause dismatching between 3D points and 2D pixels when doing projection.
    // int _cropOffset;

};

} // namespace cfsd

#endif // FEATURE_TRACKER_HPP