#include "cfsd/feature-tracker.hpp"

namespace cfsd {

// static data member initialization
size_t FeatureTracker::_featureCount = 0;

FeatureTracker::FeatureTracker(const bool verbose) : _verbose(verbose) {
    std::string type = Config::get<std::string>("detectorType");
    if (type == "ORB") {
        _detectorType = ORB;
    }
    else if (type == "BRISK") {
        _detectorType = BRISK;
    }
    else {
        std::cout << "Unexpected type, will use ORB as default detector" << std::endl;
        _detectorType = ORB; 
    }

    _matchRatio = Config::get<float>("matchRatio");

    _minMatchDist = Config::get<float>("minMatchDist");

    _maxMatchedTimes = Config::get<int>("maxMatchedTimes");
    
    switch(_detectorType) {
        case ORB:
        { // give a scope, s.t. local variables can be declared
            if (_verbose) { std::cout << "Setup ORB detector" << std::endl; }
            int numberOfFeatures = Config::get<int>("numberOfFeatures");
            float scaleFactor = Config::get<float>("scaleFactor");
            int levelPyramid = Config::get<int>("levelPyramid");
            _orb = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid);
            break;
        }
        case BRISK:
        {
            if (_verbose) { std::cout << "Setup BRISK detector" << std::endl; }
            break;
        }
    }
}

void FeatureTracker::processFrame(const cv::Mat& imgLeft, const cv::Mat& imgRight) {
    // Current camera frame's keypoints' pixel position and descriptors.
    std::vector<cvPoint2Type> curPixelsL, curPixelsR;
    cv::Mat curDescriptorsL, curDescriptorsR;
    
    internalMatch(imgLeft, imgRight, curPixelsL, curPixelsR, curDescriptorsL, curDescriptorsR);
    
    externalTrack(curPixelsL, curPixelsR, curDescriptorsL, curDescriptorsR);
    
    featurePoolUpdate();

}

void FeatureTracker::internalMatch(const cv::Mat& imgLeft, const cv::Mat& imgRight, std::vector<cvPoint2Type>& curPixelsL, std::vector<cvPoint2Type>& curPixelsR, cv::Mat& curDescriptorsL, cv::Mat& curDescriptorsR) {
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    _orb->detectAndCompute(imgLeft,  cv::noArray(), keypointsL, descriptorsL);
    _orb->detectAndCompute(imgRight, cv::noArray(), keypointsR, descriptorsR);
    
    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force Mathcer
    // cv::FlannBasedMatcher matcher; // Fast Approximate Nearest Neighbor Search Library (maybe not suitable for ORB whose descriptor is binary)
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsL, descriptorsR, matches);

    // Calculate the min distance, i.e. the best match.
    float minDist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
        // syntax: 
        // 1) function template performs argument deduction, so we can use a simple form instead of min_element<T>(...)
        // 2) min_element returns an iterator, in this case, cv::DMatch iterator
        // 3) when using iterator to access object's data member, dereference is needed, similar usage as pointer
    
    // Quick calculation of max and min distances between keypoints
    // float maxDist = 0; float minDist = 10000;
    // for (int i = 0; i < descriptorsL.size(); ++i) {
    //     float dist = matches[i].distance;
    //     if (dist < minDist) minDist = dist;
    //     if (dist > maxDist) maxDist = dist;
    // }

    #ifdef DEBUG
    std::vector<cv::DMatch> good_matches;
    #endif
    // Only keep good matches (i.e. whose distance is less than matchRatio * minDist,
    // or a small arbitary value (e.g. 30.0f) in case min_dist is very small.
    for (auto& m : matches) {
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            curPixelsL.push_back(keypointsL[m.queryIdx].pt);
            curPixelsR.push_back(keypointsR[m.trainIdx].pt);
            curDescriptorsL.push_back(descriptorsL.row(m.queryIdx));
            curDescriptorsR.push_back(descriptorsR.row(m.trainIdx));
            #ifdef DEBUG
            good_matches.push_back(m);
            #endif
        }
    }

    // #ifdef DEBUG
    // // Draw only good matches.
    // cv::Mat img_matches;
    // cv::drawMatches(imgLeft, keypointsL, imgRight, keypointsR, good_matches, img_matches);
    // cv::imshow("Left-Right Good Matches", img_matches);
    // cv::waitKey(0);
    // std::cout << "Left-Right matches: " << good_matches.size() << std::endl;
    // #endif
}

void FeatureTracker::externalTrack(const std::vector<cvPoint2Type>& curPixelsL, const std::vector<cvPoint2Type>& curPixelsR, const cv::Mat& curDescriptorsL, const cv::Mat& curDescriptorsR) {
    // At initializing step, there is no available features in the list yet,
    // so all features detected in the first frame will be added to the list.
    if (_features.empty()) {
        _newPixelsL = curPixelsL;
        _newPixelsR = curPixelsR;
        _newDescriptorsL = curDescriptorsL;
        _newDescriptorsR = curDescriptorsR;
        return;
    }

    #ifdef DEBUG
    int r = 0;
    #endif

    // Perform circular matching; only keep matches that are really good.
    std::vector<cv::DMatch> matchesL, matchesR;
    // Brute Force Mathcer: for each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    float minDist;

    // Store the correspondence map <queryIdx, trainIdx> of 'left' matching.
    std::map<int,int> mapCurHist;
    
    #ifdef DEBUG
    std::cout << "# histDescriptorsL: " << _histDescriptorsL.rows << std::endl
              << "#  curDescriptorsL: " << curDescriptorsL.rows << std::endl;
    #endif
    
    matcher.match(curDescriptorsL, _histDescriptorsL, matchesL);
    minDist = std::min_element(matchesL.begin(), matchesL.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    for (auto& m : matchesL) {
        // Only consider those good matches.
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            mapCurHist[m.queryIdx] = m.trainIdx;
        }
    }
    
    // Clear old contents to store new data.
    _newPixelsL.clear(); _newPixelsR.clear();
    _newDescriptorsL = cv::Mat(); _newDescriptorsR = cv::Mat();
    _matchedFeatureIDs.clear();
    _notMatchedFeatureIDs.clear();
    
    // Search the correspondence of 'right' matching with 'left' matching.
    matcher.match(curDescriptorsR, _histDescriptorsR, matchesR);
    minDist = std::min_element(matchesR.begin(), matchesR.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    for (auto& m : matchesR) {
        // Only consider those good matches.
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            #ifdef DEBUG
            r++;
            #endif
            
            auto search = mapCurHist.find(m.queryIdx);
            if (search != mapCurHist.end() && search->second == m.trainIdx) {
                // Satisfy circular matching, i.e. curLeft <=> histLeft <=> histRight <=> curRight <=> curLeft
                _matchedFeatureIDs.push_back(_histFeatureIDs[m.trainIdx]);
            }
            else {
                // Not satisfy circular matching, features in current frame will be inserted, while old features will be erased.
                _notMatchedFeatureIDs.push_back(_histFeatureIDs[m.trainIdx]);
                _newPixelsL.push_back(curPixelsL[m.queryIdx]);
                _newPixelsR.push_back(curPixelsR[m.queryIdx]);
                _newDescriptorsL.push_back(curDescriptorsL.row(m.queryIdx));
                _newDescriptorsR.push_back(curDescriptorsR.row(m.queryIdx));
            }
        }
    }

    #ifdef DEBUG
    std::cout << " Left hist-cur matches: " << mapCurHist.size() << std::endl
              << "Right hist-cur matches: " << r << std::endl
              << "      Circular matches: " << _matchedFeatureIDs.size() << std::endl
              << "           New matches: " << _newPixelsL.size() << std::endl;
    #endif

    // Todo: feed matched features to backend?
}

void FeatureTracker::featurePoolUpdate() {
    // The number of new and old features in the pool should be well balanced.

    #ifdef DEBUG
    std::cout << "Number of features in pool before updating: " << _features.size() << std::endl;
    int e = 0;
    #endif

    // Insert new features.
    for (int i = 0; i < _newPixelsL.size(); ++i)
        _features[_featureCount++] = Feature(_newPixelsL[i], _newPixelsR[i], _newDescriptorsL.row(i), _newDescriptorsR.row(i));

    // Update _matchedTimes.
    for (int i = 0; i < _matchedFeatureIDs.size(); ++i)
        _features[_matchedFeatureIDs[i]]._matchedTimes++;

    // Erase not matched old features.
    for (int i = 0; i < _notMatchedFeatureIDs.size(); ++i) {
        _features.erase(_notMatchedFeatureIDs[i]);
    }

    // Erase too old features, and put the rest into _histDescriptors.
    _histFeatureIDs.clear();
    _histDescriptorsL = cv::Mat(); _histDescriptorsR = cv::Mat();
    for (auto f = _features.begin(); f != _features.end(); ++f) {
        if (f->second._matchedTimes > _maxMatchedTimes) {
            #ifdef DEBUG
            e++;
            #endif

            f = _features.erase(f);
        }
        else {
            _histFeatureIDs.push_back(f->first);
            _histDescriptorsL.push_back(f->second._descriptorL);
            _histDescriptorsR.push_back(f->second._descriptorR);
        }
    }

    #ifdef DEBUG
    std::cout << "Number of features in pool after updaing: " << _features.size() << std::endl
              << "(" << _newPixelsL.size() << " features inserted, " << _notMatchedFeatureIDs.size() << " not matched old features erased, " << e << " too old features erased)" << std::endl;
    #endif
}

// void FeatureTracker::computeCamPose(SophusSE3Type& pose) {
//     // estimate camera pose by solving 3D-2D PnP problem using RANSAC scheme
//     std::vector<cvPoint3Type> points3D;  // 3D points triangulated from reference frame
//     std::vector<cvPoint2Type> points2D;  // 2D points in current frame

//     const std::vector<cvPoint3Type>& points3DRef = _keyRef->getPoints3D();
//     for (cv::DMatch& m : _matches) {
//         // matches is computed from two sets of descriptors: matcher.match(_descriptorsRef, _descriptorsCur, matches)
//         // the queryIdx corresponds to _descriptorsRef
//         // the trainIdx corresponds to _descriptorsCur
//         // the index of descriptors corresponds to the index of keypoints
//         points3D.push_back(points3DRef[m.queryIdx]);
//         points2D.push_back(_keypoints[m.trainIdx].pt);
//         if (_debug) {
//             std::cout << "3D points in world coordinate: " << points3DRef[m.queryIdx] << ", and the pixel in image: " << _keypoints[m.trainIdx].pt << std::endl;
//         }
//     }

//     cv::Mat camMatrix, distCoeffs, rvec, tvec, inliers;
//     cv::eigen2cv(_camCur->getCamLeft(), camMatrix);
//     cv::eigen2cv(_camCur->getDistLeft(), distCoeffs);
//     // if the 3D points is world coordinates, the computed transformation (rvec 
//     //    and tvec) is from world coordinate system to camera coordinate system
//     // if the 3D points is left camera coordinates of keyframe, the transformation
//     //    is from keyframe's left camera to current left camera
//     // here, assume world coordinates are used, then rvec and tvec describe left camera pose relative to world
//     cv::solvePnPRansac(points3D, points2D, camMatrix, cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE); // since didn't undistort before
//     // cv::solvePnPRansac(points3D, points2D, camMatrix, distCoeffs, rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
//     // cv::solvePnP(points3D, points2D, camMatrix, distCoeffs, rvec, tvec);

//     if (_debug) {
//         std::cout << "[RANSAC PnP] number of inliers: " << inliers.rows << std::endl;
//     }

//     cv::Mat cvR;
//     EigenMatrix3Type R;
//     EigenVector3Type t;
//     cv::Rodrigues(rvec, cvR);
//     cv::cv2eigen(cvR, R);
//     cv::cv2eigen(tvec, t);
//     pose = SophusSE3Type(R, t);
// }

} // namespace cfsd