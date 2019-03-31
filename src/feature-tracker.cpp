#include "cfsd/feature-tracker.hpp"

namespace cfsd {

FeatureTracker::FeatureTracker(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose) :
        _pMap(pMap), _pCameraModel(pCameraModel), _verbose(verbose), _featureCount(0), _frameCount(0) {
    
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

    _maxFeatureAge = Config::get<int>("maxFeatureAge");
    
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

    int height = Config::get<int>("imageHeight");
    int width = Config::get<int>("imageWidth") / 2;
    _maskL = cv::Mat::zeros(height, width, CV_8U);
    _maskR = cv::Mat::zeros(height, width, CV_8U);
    // Pixel representation.
    int h1  = Config::get<int>("h1");
    int h2  = Config::get<int>("h2");
    int w1L = Config::get<int>("w1L");
    int w2L = Config::get<int>("w2L");
    int w3L = Config::get<int>("w3L");
    int w4L = Config::get<int>("w4L");
    int w1R = Config::get<int>("w1R");
    int w2R = Config::get<int>("w2R");
    int w3R = Config::get<int>("w3R");
    int w4R = Config::get<int>("w4R");
    for (int i = h1; i < height; ++i) {
        int x1L = (height-h2) / (w1L-w2L) * (i-height) + w1L;
        int x2L = (height-h2) / (w4L-w3L) * (i-height) + w4L;
        int x1R = (height-h2) / (w1R-w2R) * (i-height) + w1R;
        int x2R = (height-h2) / (w4R-w3R) * (i-height) + w4R;
        for (int j = 0; j < width; ++j) {
            if (i < h2) {
                _maskL.at<char>(i,j) = 255;
                _maskR.at<char>(i,j) = 255;
            }
            else {
                if (j < x1L || j > x2L)
                    _maskL.at<char>(i,j) = 255;
                if (j < x1R || j > x2R)
                    _maskR.at<char>(i,j) = 255;
            }
        }
    }
}

void FeatureTracker::process(const cv::Mat& grayLeft, const cv::Mat& grayRight) {
    // Remap to undistorted and rectified image (detection mask needs to be updated)
    // cv::remap (about 3~5 ms) takes less time than cv::undistort (about 20~30 ms) method
    auto start = std::chrono::steady_clock::now();
    cv::Mat imgLeft, imgRight;
    cv::remap(grayLeft, imgLeft, _pCameraModel->_rmap[0][0], _pCameraModel->_rmap[0][1], cv::INTER_LINEAR);
    cv::remap(grayRight, imgRight, _pCameraModel->_rmap[1][0], _pCameraModel->_rmap[1][1], cv::INTER_LINEAR);
    auto end = std::chrono::steady_clock::now();
    std::cout << "remap elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    #ifdef DEBUG_IMG
    cv::Mat rectified;
    cv::hconcat(imgLeft, imgRight, rectified);
    cv::imshow("rectified", rectified);
    cv::waitKey(1);
    #endif
    
    // Current camera frame's keypoints' pixel position and descriptors.
    std::vector<cv::Point2d> curPixelsL, curPixelsR;
    cv::Mat curDescriptorsL, curDescriptorsR;
    
    start = std::chrono::steady_clock::now();
    internalMatch(imgLeft, imgRight, curPixelsL, curPixelsR, curDescriptorsL, curDescriptorsR);
    end = std::chrono::steady_clock::now();
    std::cout << "internal match elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
    
    start = std::chrono::steady_clock::now();
    externalTrack(curPixelsL, curPixelsR, curDescriptorsL, curDescriptorsR);
    end = std::chrono::steady_clock::now();
    std::cout << "external match elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
    
    start = std::chrono::steady_clock::now();
    featurePoolUpdate();
    end = std::chrono::steady_clock::now();
    std::cout << "feature pool update elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    _frameCount++;
}

void FeatureTracker::internalMatch(const cv::Mat& imgLeft, const cv::Mat& imgRight, std::vector<cv::Point2d>& curPixelsL, std::vector<cv::Point2d>& curPixelsR, cv::Mat& curDescriptorsL, cv::Mat& curDescriptorsR) {
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    _orb->detectAndCompute(imgLeft,  _maskL, keypointsL, descriptorsL);
    _orb->detectAndCompute(imgRight, _maskR, keypointsR, descriptorsR);
    
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
    
    // Quick calculation of max and min distances between keypoints.
    // float maxDist = 0; float minDist = 10000;
    // for (int i = 0; i < keypointsL.size(); ++i) {
    //     float dist = matches[i].distance;
    //     if (dist < minDist) minDist = dist;
    //     if (dist > maxDist) maxDist = dist;
    // }

    // Only keep good matches (i.e. whose distance is less than matchRatio * minDist,
    // or a small arbitary value (e.g. 30.0f) in case min_dist is very small.
    std::vector<cv::Point2d> pixelsL, pixelsR;
    std::vector<int> indexL, indexR;
    for (auto& m : matches) {
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            pixelsL.push_back(keypointsL[m.queryIdx].pt);
            pixelsR.push_back(keypointsR[m.trainIdx].pt);
            indexL.push_back(m.queryIdx);
            indexR.push_back(m.trainIdx);
        }
    }

    // However, only removing matches of big descriptor distance is not enough.
    // To root out outliers, use 2D-2D RANSAC.
    auto start = std::chrono::steady_clock::now();
    cv::Mat ransacMask;
    cv::findFundamentalMat(pixelsL, pixelsR, ransacMask);
    for (int i = 0; i < ransacMask.rows; i++) {
        if (ransacMask.at<bool>(i)) {
            // std::cout << "left pixel: " << pixels1[i] << std::endl;
            // std::cout << "right pixel: " << pixels2[i] << std::endl;
            curPixelsL.push_back(pixelsL[i]);
            curPixelsR.push_back(pixelsR[i]);
            curDescriptorsL.push_back(descriptorsL.row(indexL[i]));
            curDescriptorsR.push_back(descriptorsR.row(indexR[i]));
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "[internal] findFundamentalMat with RANSAC elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    #ifdef DEBUG_IMG
    std::cout << "# Left-Right matches after distance-filter: " << pixelsL.size() << std::endl;
    std::cout << "# Left-Right matches after RANSAC: " << curPixelsL.size() << std::endl;
    #endif
}

void FeatureTracker::externalTrack(const std::vector<cv::Point2d>& curPixelsL, const std::vector<cv::Point2d>& curPixelsR, const cv::Mat& curDescriptorsL, const cv::Mat& curDescriptorsR) {
    // At initializing step, there is no available features in the list yet,
    // so all features detected in the first frame will be added to the list.
    if (_features.empty()) {
        // (std::vector copy constructor performs deep copy)
        _newPixelsL = curPixelsL;
        _newPixelsR = curPixelsR;
        _newDescriptorsL = curDescriptorsL;
        _newDescriptorsR = curDescriptorsR;
        return;
    }

    #ifdef DEBUG_IMG
    int rightCount = 0;
    #endif

    // Perform circular matching; only keep matches that are really good.
    std::vector<cv::DMatch> matchesL, matchesR;
    // Brute Force Mathcer: for each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    float minDist;

    #ifdef DEBUG_IMG
    std::cout << "# histDescriptorsL: " << _histDescriptorsL.rows << std::endl
              << "#  curDescriptorsL: " << curDescriptorsL.rows << std::endl;
    #endif
    
    matcher.match(curDescriptorsL, _histDescriptorsL, matchesL);
    minDist = std::min_element(matchesL.begin(), matchesL.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    std::vector<cv::Point2d> pixelsL, pixelsR;
    std::vector<int> indexL, indexR;
    for (auto& m : matchesL) {
        // Only consider those good matches.
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            pixelsL.push_back(keypointsL[m.queryIdx].pt);
            pixelsR.push_back(keypointsR[m.trainIdx].pt);
            indexL.push_back(m.queryIdx);
            indexR.push_back(m.trainIdx);
        }
    }

    // Store the correspondence map <queryIdx, trainIdx> of 'left' matching.
    std::map<int,int> mapCurHist;

    // Use 2D-2D RANSAC to remove outliers.
    auto start = std::chrono::steady_clock::now();
    cv::Mat ransacMask;
    cv::findFundamentalMat(pixelsL, pixelsR, ransacMask);
    for (int i = 0; i < ransacMask.rows; i++) {
        if (ransacMask.at<bool>(i))
            mapCurHist[indexL[i]] = indexR[i];
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "[external] findFundamentalMat with RANSAC elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    // Record which features in current frame will be viewed as new features, if circular matching is satisfied, it will be false; otherwise, true.
    std::vector<bool> curFeaturesMask (curDescriptorsL.rows, true);
    
    // Search the correspondence of 'right' matching with 'left' matching.
    _matchedFeatureIDs.clear();
    matcher.match(curDescriptorsR, _histDescriptorsR, matchesR);
    minDist = std::min_element(matchesR.begin(), matchesR.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    for (auto& m : matchesR) {
        // Only consider those good matches.
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            #ifdef DEBUG_IMG
            rightCount++;
            #endif
            
            auto search = mapCurHist.find(m.queryIdx);
            if (search != mapCurHist.end() && search->second == m.trainIdx) {
                // Satisfy circular matching, i.e. curLeft <=> histLeft <=> histRight <=> curRight <=> curLeft, the age of history features will increase 1.
                _matchedFeatureIDs.push_back(_histFeatureIDs[m.trainIdx]);
                // Will not be added as new features.
                curFeaturesMask[m.queryIdx] = false;
            }
        }
    }

    // The not matched features in current frame will be viewed as new features and be added into feature pool.
    _newPixelsL.clear();
    _newPixelsR.clear();
    _newDescriptorsL = cv::Mat();
    _newDescriptorsR = cv::Mat();
    for (int i = 0; i < curFeaturesMask.size(); i++) {
        if (!curFeaturesMask[i]) continue;
        _newPixelsL.push_back(curPixelsL[i]);
        _newPixelsR.push_back(curPixelsR[i]);
        _newDescriptorsL.push_back(curDescriptorsL.row(i));
        _newDescriptorsR.push_back(curDescriptorsR.row(i));
    }

    #ifdef DEBUG_IMG
    std::cout << "# left hist-cur matches (RANSAC): " << mapCurHist.size() << std::endl
              << "# right hist-cur matches: " << rightCount << std::endl
              << "# circular matches: " << _matchedFeatureIDs.size() << std::endl
              << "# new features: " << _newPixelsL.size() << std::endl;
    #endif
}

void FeatureTracker::featurePoolUpdate() {
    // The number of new and old features in the pool should be well balanced.

    #ifdef DEBUG_IMG
    std::cout << "Number of features in pool before updating: " << _features.size() << std::endl;
    int eraseCount = 0;
    #endif

    // Insert new features, initial _age is -2, will add 2 later.
    for (int i = 0; i < _newPixelsL.size(); ++i) {
        _features[_featureCount++] = Feature(_frameCount, _newPixelsL[i], _newPixelsR[i], _newDescriptorsL.row(i), _newDescriptorsR.row(i), -2);
    }

    // _age minus 1, will add 2 later.
    for (int i = 0; i < _matchedFeatureIDs.size(); ++i) {
        _features[_matchedFeatureIDs[i]]._age -= 1;
        
        // Todo: feed matched features to backend?
        _pMap->
    }

    // _age add 2 for all features, then erase too old features, and put the rest into _histDescriptors.
    _histFeatureIDs.clear();
    _histDescriptorsL = cv::Mat(); _histDescriptorsR = cv::Mat();
    for (auto f = _features.begin(); f != _features.end(); ++f) {
        f->second._age += 2;
        if (f->second._age > _maxFeatureAge) {
            #ifdef DEBUG_IMG
            eraseCount++;
            #endif

            f = _features.erase(f);
        }
        else {
            _histFeatureIDs.push_back(f->first);
            _histDescriptorsL.push_back(f->second._descriptorL);
            _histDescriptorsR.push_back(f->second._descriptorR);
        }
    }

    #ifdef DEBUG_IMG
    std::cout << "Number of features in pool after updaing: " << _features.size() << std::endl
              << "(" << _newPixelsL.size() << " features inserted, " << eraseCount << " too old features erased)" << std::endl;
    #endif
}

// void FeatureTracker::computeCamPose(Sophus::SE3d& pose) {
//     // estimate camera pose by solving 3D-2D PnP problem using RANSAC scheme
//     std::vector<cv::Point3d> points3D;  // 3D points triangulated from reference frame
//     std::vector<cv::Point2d> points2D;  // 2D points in current frame

//     const std::vector<cv::Point3d>& points3DRef = _keyRef->getPoints3D();
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
//     Eigen::Matrix3d R;
//     Eigen::Vector3d t;
//     cv::Rodrigues(rvec, cvR);
//     cv::cv2eigen(cvR, R);
//     cv::cv2eigen(tvec, t);
//     pose = Sophus::SE3d(R, t);
// }

} // namespace cfsd