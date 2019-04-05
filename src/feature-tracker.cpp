#include "cfsd/feature-tracker.hpp"

namespace cfsd {

FeatureTracker::FeatureTracker(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<Optimizer>& pOptimizer, const cfsd::Ptr<ImuPreintegrator> pImuPreintegrator, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose) :
        _pMap(pMap), _pOptimizer(pOptimizer), _pImuPreintegrator(pImuPreintegrator), _pCameraModel(pCameraModel), _verbose(verbose), _featureCount(0), _frameCount(0) {
    
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
    // int w1L = Config::get<int>("w1L");
    // int w2L = Config::get<int>("w2L");
    // int w3L = Config::get<int>("w3L");
    // int w4L = Config::get<int>("w4L");
    // int w1R = Config::get<int>("w1R");
    // int w2R = Config::get<int>("w2R");
    // int w3R = Config::get<int>("w3R");
    // int w4R = Config::get<int>("w4R");
    // for (int i = h1; i < height; ++i) {
    //     int x1L = (height-h2) / (w1L-w2L) * (i-height) + w1L;
    //     int x2L = (height-h2) / (w4L-w3L) * (i-height) + w4L;
    //     int x1R = (height-h2) / (w1R-w2R) * (i-height) + w1R;
    //     int x2R = (height-h2) / (w4R-w3R) * (i-height) + w4R;
    //     for (int j = 0; j < width; ++j) {
    //         if (i < h2) {
    //             _maskL.at<char>(i,j) = 255;
    //             _maskR.at<char>(i,j) = 255;
    //         }
    //         else {
    //             if (j < x1L || j > x2L)
    //                 _maskL.at<char>(i,j) = 255;
    //             if (j < x1R || j > x2R)
    //                 _maskR.at<char>(i,j) = 255;
    //         }
    //     }
    // }
    for (int i = h1; i < h2; ++i) {
        for (int j = 0; j < width; ++j) {
            _maskL.at<char>(i,j) = 255;
            _maskR.at<char>(i,j) = 255;
        }
    }
}

void FeatureTracker::process(const cv::Mat& grayLeft, const cv::Mat& grayRight) {
    // Remap to undistorted and rectified image (detection mask needs to be updated)
    // cv::remap (about 3~5 ms) takes less time than cv::undistort (about 20~30 ms) method
    auto start = std::chrono::steady_clock::now();
    cv::Mat imgLeft, imgRight;
    std::cout << grayLeft.size() << std::endl;
    cv::remap(grayLeft, imgLeft, _pCameraModel->_rmap[0][0], _pCameraModel->_rmap[0][1], cv::INTER_LINEAR);
    cv::remap(grayRight, imgRight, _pCameraModel->_rmap[1][0], _pCameraModel->_rmap[1][1], cv::INTER_LINEAR);
    auto end = std::chrono::steady_clock::now();
    std::cout << "remap elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    #ifdef DEBUG_IMG
    cv::Mat rectified;
    cv::hconcat(imgLeft, imgRight, rectified);
    cv::imshow("rectified", rectified);
    cv::waitKey(0);
    #endif
    
    // Current camera frame's keypoints' pixel position and descriptors.
    std::vector<cv::Point2d> curPixelsL, curPixelsR;
    cv::Mat curDescriptorsL, curDescriptorsR;
    
    start = std::chrono::steady_clock::now();
    internalMatch(imgLeft, imgRight, curPixelsL, curPixelsR, curDescriptorsL, curDescriptorsR);
    end = std::chrono::steady_clock::now();
    std::cout << "internal match elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
    
    // Record which features in current frame will possibly be viewed as new features, if circular matching is satisfied, it will be false; otherwise, true.
    std::vector<bool> curFeatureMask (curDescriptorsL.rows, true);

    start = std::chrono::steady_clock::now();
    externalTrack(curPixelsL, curPixelsR, curDescriptorsL, curDescriptorsR, curFeatureMask, true);
    end = std::chrono::steady_clock::now();
    std::cout << "external match elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
    
    // TODO..........................
    // This step should be after the motion-only BA, s.t. we can know if current frame is keyframe and also the current camera pose.
    start = std::chrono::steady_clock::now();
    featurePoolUpdate(curPixelsL, curPixelsR, curDescriptorsL, curDescriptorsR, curFeatureMask);
    end = std::chrono::steady_clock::now();
    std::cout << "feature pool update elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    _frameCount++;
}

void FeatureTracker::internalMatch(const cv::Mat& imgLeft, const cv::Mat& imgRight, std::vector<cv::Point2d>& curPixelsL, std::vector<cv::Point2d>& curPixelsR, cv::Mat& curDescriptorsL, cv::Mat& curDescriptorsR, const bool useRANSAC) {
    auto start = std::chrono::steady_clock::now();
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    _orb->detectAndCompute(imgLeft,  _maskL, keypointsL, descriptorsL);
    _orb->detectAndCompute(imgRight, _maskR, keypointsR, descriptorsR);
    auto end = std::chrono::steady_clock::now();
    std::cout << "orb detection elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
    
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

    // Only keep good matches, i.e. distance is less than matchRatio * minDist, or a small arbitary value (e.g. 30.0f) in case min_dist is very small.
    if (useRANSAC) {
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

        // However, only removing matches of big descriptor distance is not enough. To root out outliers, use 2D-2D RANSAC.
        start = std::chrono::steady_clock::now();
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
        end = std::chrono::steady_clock::now();
        std::cout << "[internal] findFundamentalMat with RANSAC elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
        std::cout << "# Left-Right matches after distance-filter: " << pixelsL.size() << std::endl;
        std::cout << "# Left-Right matches after RANSAC: " << curPixelsL.size() << std::endl;
    }
    else {
        for (auto& m : matches) {
            if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
                curPixelsL.push_back(keypointsL[m.queryIdx].pt);
                curPixelsR.push_back(keypointsR[m.trainIdx].pt);
                curDescriptorsL.push_back(descriptorsL.row(m.queryIdx));
                curDescriptorsR.push_back(descriptorsR.row(m.trainIdx));
            }
        }
    }
}

void FeatureTracker::externalTrack(const std::vector<cv::Point2d>& curPixelsL, const std::vector<cv::Point2d>& curPixelsR, const cv::Mat& curDescriptorsL, const cv::Mat& curDescriptorsR, std::vector<bool>& curFeatureMask, const bool useRANSAC) {
    // At initializing step, there is no available features in the list yet, so all features detected in the first frame will be added to the list.
    // The first frame will be set as keyframe as well.
    if (_features.empty()) {
        // curFeatureMask all true.
        return;
    }

    int rightCount = 0;

    std::vector<cv::DMatch> matchesL, matchesR;
    // Brute Force Mathcer: for each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    float minDist;

    std::cout << "# histDescriptorsL: " << _histDescriptorsL.rows << std::endl
              << "#  curDescriptorsL: " << curDescriptorsL.rows << std::endl;

    // Store the correspondence map <queryIdx, trainIdx> of 'left' matching.
    std::map<int,int> mapCurHist;

    std::vector<cv::Point2d> pixelsCur, pixelsHist;
    std::vector<int> indexCur, indexHist;
    matcher.match(curDescriptorsL, _histDescriptorsL, matchesL);
    minDist = std::min_element(matchesL.begin(), matchesL.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    if (useRANSAC) {
        for (auto& m : matchesL) {
            // Only consider those good matches.
            if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
                pixelsCur.push_back(curPixelsL[m.queryIdx]);
                pixelsHist.push_back(_features[_histFeatureIDs[m.trainIdx]].pixelsL[0]);
                indexCur.push_back(m.queryIdx);
                indexHist.push_back(m.trainIdx);
            }
        }

        // To root out outliers, use 2D-2D RANSAC.
        start = std::chrono::steady_clock::now();
        cv::Mat ransacMask;
        cv::findFundamentalMat(pixelsCur, pixelsHist, ransacMask);
        for (int i = 0; i < ransacMask.rows; i++) {
            if (ransacMask.at<bool>(i)) {
                mapCurHist[indexCur[i]] = indexHist[i];
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "[external] findFundamentalMat with RANSAC elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
    }
    else {
        for (auto& m : matchesL)
            // Only consider those good matches.
            if (m.distance < std::max(_matchRatio * minDist, _minMatchDist))
                mapCurHist[m.queryIdx] = m.trainIdx;
    }
    
    // Search the correspondence of 'right' matching with 'left' matching.
    _matchedFeatureIDs.clear();
    matcher.match(curDescriptorsR, _histDescriptorsR, matchesR);
    minDist = std::min_element(matchesR.begin(), matchesR.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    for (auto& m : matchesR) {
        // Only consider those good matches.
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            rightCount++;
            
            // Perform circular matching; only keep matches that are really good.
            auto search = mapCurHist.find(m.queryIdx);
            if (search != mapCurHist.end() && search->second == m.trainIdx) {
                // Satisfy circular matching, i.e. curLeft <=> histLeft <=> histRight <=> curRight <=> curLeft, the age of history features will increase 1.
                size_t featureID = _histFeatureIDs[m.trainIdx];
                _features[featureID].seenByFrames.push_back(_frameCount);
                _features[featureID].pixelsL.push_back(curPixelsL[m.queryIdx]);
                _features[featureID].pixelsR.push_back(curPixelsR[m.queryIdx]);
                _matchedFeatureIDs.push_back(featureID);
                // Will not be added as new features.
                curFeatureMask[m.queryIdx] = false;
            }
        }
    }

    std::cout << " # left hist-cur matches (RANSAC): " << mapCurHist.size() << std::endl
              << " # right hist-cur matches: " << rightCount << std::endl
              << " # circular matches: " << _matchedFeatureIDs.size() << std::endl;

    // Perform motion-only BA.
    start = std::chrono::steady_clock::now();
    _pOptimizer->motionOnlyBA(_features, _matchedFeatureIDs);
    _pImuPreintegrator->updateBias();
    end = std::chrono::steady_clock::now();
    std::cout << "motion-only BA elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
}

void FeatureTracker::featurePoolUpdate(const std::vector<cv::Point2d>& curPixelsL, const std::vector<cv::Point2d>& curPixelsR, const cv::Mat& curDescriptorsL, const cv::Mat& curDescriptorsR, const std::vector<bool>& curFeatureMask) {
    // The number of new and old features in the pool should be well balanced.

    std::cout << "# features in pool before updating: " << _features.size() << std::endl;
    int eraseCount = 0;
    int insertCount = 0;

    // _age minus 1, will add 2 later.
    for (int i = 0; i < _matchedFeatureIDs.size(); i++) {
        _features[_matchedFeatureIDs[i]].age -= 1;
    }

    // age add 2 for all features, then erase too old features, and put the rest into _histDescriptors.
    _histFeatureIDs.clear();
    _histDescriptorsL = cv::Mat();
    _histDescriptorsR = cv::Mat();
    for (auto f = _features.begin(); f != _features.end(); ++f) {
        f->second.age += 2;
        if (f->second.age > _maxFeatureAge) {
            // Save the erased map points.
            _pMap->_mapPoints.push_back(MapPoint(f->second));

            _features.erase(f);
            eraseCount++;
        }
        else {
            _histFeatureIDs.push_back(f->first);
            _histDescriptorsL.push_back(f->second.descriptorL);
            _histDescriptorsR.push_back(f->second.descriptorR);
        }
    }

    // If current frame is keyframe, the not matched features in current frame will be viewed as new features and be added into feature pool.
    if (_features.empty() || _pMap->_isKeyframe) {
        // Triangulate current frame's keypoints.
        cv::Mat points4D;
        cv::triangulatePoints(_pCameraModel->_P1, _pCameraModel->_P2, curPixelsL, curPixelsR, points4D);
        
        for (int i = 0; i < curFeatureMask.size(); i++) {
            // Points4D is in homogeneous coordinates.
            double depth = points4D.at<double>(2,i) / points4D.at<double>(3,i);

            // If the feature is matched before or the corresponding 3D point is too far away (i.e. less accuracy) w.r.t current camera, it will not be added.
            if (!curFeatureMask[i] || depth > 20) continue;

            // Add these new features' descriptors to _hist for the convenience of next external matching.
            _histFeatureIDs.push_back(_featureCount);
            _histDescriptorsL.push_back(curDescriptorsL.row(i));
            _histDescriptorsR.push_back(curDescriptorsR.row(i));

            // The triangulated points coordinates are w.r.t current camera frame, should be converted to world frame.
            Eigen::Vector3d point_wrt_cam = Eigen::Vector3d(points4D.at<double>(0,i) / points4D.at<double>(3,i),
                                                            points4D.at<double>(1,i) / points4D.at<double>(3,i),
                                                            points4D.at<double>(2,i) / points4D.at<double>(3,i));
            // _pMap->getBodyPose() gives the transformation from current body frame to world frame, T_WB.
            // _pCameraModel->_T_BC gives the pre-calibrated transformation from camera to body/imu frame.
            Eigen::Vector3d position = _pMap->getBodyPose() * _pCameraModel->_T_BC * point_wrt_cam;                                                           

            // Insert new features.
            _features[_featureCount++] = Feature(_frameCount, curPixelsL[i], curPixelsR[i], curDescriptorsL.row(i), curDescriptorsR.row(i), position, 0);
            insertCount++;      
        }
    }

    std::cout << "# features in pool after updaing: " << _features.size() << " (" << insertCount << " features inserted, " << eraseCount << " too old features erased)" << std::endl;
}

} // namespace cfsd