#include "cfsd/feature-tracker.hpp"

namespace cfsd {

FeatureTracker::FeatureTracker(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose) :
        _pMap(pMap), _pCameraModel(pCameraModel), _verbose(verbose) {

    _matchRatio = Config::get<float>("matchRatio");

    _minMatchDist = Config::get<float>("minMatchDist");

    _maxVerticalPixelDist = Config::get<float>("maxVerticalPixelDist");

    _maxFeatureAge = Config::get<int>("maxFeatureAge");

    _maxDepth = Config::get<double>("maxDepth");
    
    if (_verbose) { std::cout << "Setup ORB detector" << std::endl; }
    int numberOfFeatures = Config::get<int>("numberOfFeatures");
    float scaleFactor = Config::get<float>("scaleFactor");
    int levelPyramid = Config::get<int>("levelPyramid");
    int edgeThreshold = Config::get<int>("edgeThreshold");
    int patchSize = Config::get<int>("patchSize");
    int fastThreshold = Config::get<int>("fastThreshold");
    if (Config::get<int>("scoreType") == 0) {
        _orb = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
        // _orbLeft = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
        // _orbRight = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
    }
    else {
        _orb = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::FAST_SCORE, patchSize, fastThreshold);
        // _orbLeft = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
        // _orbRight = cv::ORB::create(numberOfFeatures, scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
    }

    _minRotation = Config::get<double>("sfmRotation");
    _minTranslation = Config::get<double>("sfmTranslation");

    // int height = Config::get<int>("processHeight");
    // int width = Config::get<int>("processWidth");
    // _mask = cv::Mat::zeros(height, width, CV_8U);
    // // Pixel representation.
    // int h1  = Config::get<int>("h1");
    // int h2  = Config::get<int>("h2");
    // for (int i = h1; i < h2; ++i)
    //     for (int j = 0; j < width; ++j)
    //         _mask.at<char>(i,j) = 255;

    // // Initialize orb detection (it is found that the first orb detection is very slow).
    // auto start = std::chrono::steady_clock::now();
    // std::vector<cv::KeyPoint> keypoints;
    // cv::Mat descriptors;
    // _orb->detectAndCompute(_mask, _mask, keypoints, descriptors);
    // auto end = std::chrono::steady_clock::now();
    // std::cout << "initial orb detection elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
}

bool FeatureTracker::processImage(const cv::Mat& grayLeft, const cv::Mat& grayRight) {
    // Remap to undistorted and rectified image (detection mask needs to be updated)
    // cv::remap (about 3~5 ms) takes less time than cv::undistort (about 20~30 ms) method
    auto start = std::chrono::steady_clock::now();
    cv::Mat imgLeft, imgRight;
    cv::remap(grayLeft, imgLeft, _pCameraModel->_rmap[0][0], _pCameraModel->_rmap[0][1], cv::INTER_LINEAR);
    cv::remap(grayRight, imgRight, _pCameraModel->_rmap[1][0], _pCameraModel->_rmap[1][1], cv::INTER_LINEAR);
    auto end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "remap elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    // std::cout << _mask.size() << std::endl;
    // std::cout << imgLeft.size() << ", " << imgRight.size() << std::endl;
    // #ifdef DEBUG_IMG
    // cv::Mat rectified;
    // cv::line(imgLeft, cv::Point(0,210), cv::Point(672,210), cv::Scalar(0,0,255), 2);
    // cv::line(imgLeft, cv::Point(0,300), cv::Point(672,300), cv::Scalar(0,0,255), 2);
    // cv::hconcat(imgLeft, imgRight, rectified);
    // cv::imshow("rectified", rectified);
    // cv::waitKey(0);
    // #endif
    
    _curPixelsL.clear();
    _curPixelsR.clear();
    _curDescriptorsL = cv::Mat();
    _curDescriptorsR = cv::Mat();
    
    start = std::chrono::steady_clock::now();
    internalMatch(imgLeft, imgRight);
    end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "internal match elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
    
    // Record which features in current frame will possibly be viewed as new features, if circular matching is satisfied, it will be false; otherwise, true.
    _curFeatureMask.clear();
    _curFeatureMask.resize(_curDescriptorsL.rows, true);

    start = std::chrono::steady_clock::now();
    externalTrack(true);
    end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "external match elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;

    return _matchedFeatureIDs.empty();
}

void FeatureTracker::internalMatch(const cv::Mat& imgLeft, const cv::Mat& imgRight, const bool useRANSAC) {
    auto start = std::chrono::steady_clock::now();
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    // _orb->detectAndCompute(imgLeft,  _mask, keypointsL, descriptorsL);
    // _orb->detectAndCompute(imgRight, _mask, keypointsR, descriptorsR);
    _orb->detectAndCompute(imgLeft,  cv::noArray(), keypointsL, descriptorsL);
    _orb->detectAndCompute(imgRight, cv::noArray(), keypointsR, descriptorsR);
    auto end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "orb detection elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;

    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force Mathcer
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsL, descriptorsR, matches);
    float minDist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;

    // #ifdef DEBUG_IMG
    // cv::Mat img_matches;
    // cv::drawMatches(imgLeft, keypointsL, imgRight, keypointsR, matches, img_matches);
    // cv::imshow("matches", img_matches);
    // cv::waitKey(0);
    // #endif

    // Only keep good matches, for pixel (ul, vl) and (ur, vr), |vl-vr| should be small enough since the image has been rectified.
    if (useRANSAC) {
        std::vector<cv::Point2d> pixelsL, pixelsR;
        std::vector<int> indexL, indexR;
        for (auto& m : matches) {
            if (m.distance < std::max(_matchRatio * minDist, _minMatchDist) && std::abs(keypointsL[m.queryIdx].pt.y - keypointsR[m.trainIdx].pt.y) < _maxVerticalPixelDist) {
                pixelsL.push_back(keypointsL[m.queryIdx].pt);
                pixelsR.push_back(keypointsR[m.trainIdx].pt);
                indexL.push_back(m.queryIdx);
                indexR.push_back(m.trainIdx);
            }
        }

        // However, only removing matches of big descriptor distance maybe not enough. To root out outliers further, use 2D-2D RANSAC.
        start = std::chrono::steady_clock::now();
        cv::Mat ransacMask;
        cv::findFundamentalMat(pixelsL, pixelsR, ransacMask);
        for (int i = 0; i < ransacMask.rows; i++) {
            if (ransacMask.at<bool>(i)) {
                // std::cout << "left pixel: " << pixels1[i] << std::endl;
                // std::cout << "right pixel: " << pixels2[i] << std::endl;
                _curPixelsL.push_back(pixelsL[i]);
                _curPixelsR.push_back(pixelsR[i]);
                _curDescriptorsL.push_back(descriptorsL.row(indexL[i]));
                _curDescriptorsR.push_back(descriptorsR.row(indexR[i]));
            }
        }
        end = std::chrono::steady_clock::now();
        if (_verbose) {
            std::cout << "internal findFundamentalMat with RANSAC elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
            std::cout << "# cur left-right matches: " << matches.size() << std::endl;
            std::cout << "# cur left-right matches after vertical coordinate matching: " << pixelsL.size() << std::endl;
            std::cout << "# cur left-right matches after RANSAC: " << _curPixelsL.size() << std::endl;
        }
    }
    else {
        // std::vector<cv::DMatch> good_matches;
        for (auto& m : matches) {
            if (m.distance < std::max(_matchRatio * minDist, _minMatchDist) && std::abs(keypointsL[m.queryIdx].pt.y - keypointsR[m.trainIdx].pt.y) < _maxVerticalPixelDist) {
                _curPixelsL.push_back(keypointsL[m.queryIdx].pt);
                _curPixelsR.push_back(keypointsR[m.trainIdx].pt);
                _curDescriptorsL.push_back(descriptorsL.row(m.queryIdx));
                _curDescriptorsR.push_back(descriptorsR.row(m.trainIdx));
                // good_matches.push_back(m);
            }
        }
        // cv::Mat img_good_matches;
        // cv::drawMatches(imgLeft, keypointsL, imgRight, keypointsR, good_matches, img_good_matches);
        // cv::imshow("matches", img_good_matches);
        // cv::waitKey(0);

        if (_verbose) {
            std::cout << "# cur left-right matches: " << matches.size() << std::endl;
            std::cout << "# cur left-right matches after vertical coordinate matching: " << _curPixelsL.size() << std::endl;
        }
    }
}

void FeatureTracker::externalTrack(const bool useRANSAC) {
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    
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
    // cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(5,10,2)); // Fast Approximate Nearest Neighbor Search Library
    float minDist;

    if (_verbose) {
        std::cout << "# histDescriptorsL: " << _histDescriptorsL.rows << std::endl
                  << "#  curDescriptorsL: " << _curDescriptorsL.rows << std::endl;
    }

    // Store the correspondence map <queryIdx, trainIdx> of 'left' matching.
    std::map<int,int> mapCurHist;

    matcher.match(_curDescriptorsL, _histDescriptorsL, matchesL);
    minDist = std::min_element(matchesL.begin(), matchesL.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    if (useRANSAC) {
        std::vector<cv::Point2d> pixelsCur, pixelsHist;
        std::vector<int> indexCur, indexHist;
        
        for (auto& m : matchesL) {
            // Only consider those good matches.
            if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
                pixelsCur.push_back(_curPixelsL[m.queryIdx]);
                pixelsHist.push_back(_features[_histFeatureIDs[m.trainIdx]]->pixelL);
                indexCur.push_back(m.queryIdx);
                indexHist.push_back(m.trainIdx);
            }
        }

        //TODO............what if there is zero match?

        // To root out outliers, use 2D-2D RANSAC.
        start = std::chrono::steady_clock::now();
        cv::Mat ransacMask;
        cv::findFundamentalMat(pixelsCur, pixelsHist, ransacMask);
        for (int i = 0; i < ransacMask.rows; i++) {
            if (ransacMask.at<bool>(i))
                mapCurHist[indexCur[i]] = indexHist[i];
        }
        end = std::chrono::steady_clock::now();
        if (_verbose) {
            std::cout << "external findFundamentalMat with RANSAC elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
            std::cout << "# left cur-hist matches: " << matchesL.size() << std::endl;
            std::cout << "# left cur-hist matches after hamming distance selection: " << pixelsCur.size() << std::endl;
            std::cout << "# left cur-hist matches after RANSAC: " << mapCurHist.size() << std::endl;
        }
    }
    else {
        for (auto& m : matchesL)
            // Only consider those good matches.
            if (m.distance < std::max(_matchRatio * minDist, _minMatchDist))
                mapCurHist[m.queryIdx] = m.trainIdx;
        if (_verbose) std::cout << "# left cur-hist matches after hamming distance selection: " << mapCurHist.size() << std::endl;
    }
    
    // Search the correspondence of 'right' matching with 'left' matching.
    _matchedFeatureIDs.clear();
    _pMap->_frames.back().clear();
    matcher.match(_curDescriptorsR, _histDescriptorsR, matchesR);
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
                _pMap->_frames.back().push_back(std::make_pair(_curPixelsL[m.queryIdx], _features[featureID]->position));
                _matchedFeatureIDs.push_back(featureID);
                // Will not be added as new features.
                _curFeatureMask[m.queryIdx] = false;
            }
        }
    }

    if (_verbose) {
        std::cout << "# right cur-hist matches: " << matchesR.size() << std::endl;
        std::cout << "# right cur-hist matches after hamming distance selection: " << rightCount << std::endl;
        std::cout << "# circular matches: " << _matchedFeatureIDs.size() << std::endl;
    }
}

void FeatureTracker::featurePoolUpdate() {
    // The number of new and old features in the pool should be well balanced.
    int eraseCount = 0;
    int insertCount = 0;
    if (_verbose) std::cout << "# features in pool before updating: " << _features.size() << std::endl;

    // If current frame is keyframe, the not matched features in current frame will be viewed as new features and be added into feature pool.
    if (_features.empty() || _pMap->_isKeyframe) {
        // _age minus 1, will add 2 later.
        for (int i = 0; i < _matchedFeatureIDs.size(); i++) {
            _features[_matchedFeatureIDs[i]]->age -= 1;
        }

        // age add 2 for all features, then erase too old features, and put the rest into _histDescriptors.
        _histFeatureIDs.clear();
        _histDescriptorsL = cv::Mat();
        _histDescriptorsR = cv::Mat();
        auto f = _features.begin();
        while (f != _features.end()) {
            f->second->age += 2;
            if (f->second->age > _maxFeatureAge) {
                f = _features.erase(f);
                eraseCount++;
            }
            else {
                _histFeatureIDs.push_back(f->first);
                _histDescriptorsL.push_back(f->second->descriptorL);
                _histDescriptorsR.push_back(f->second->descriptorR);
                // DON'T increment after erasing.
                f++;
            }
        }

        // Triangulate current frame's keypoints.
        cv::Mat points4D;
        cv::triangulatePoints(_pCameraModel->_P1, _pCameraModel->_P2, _curPixelsL, _curPixelsR, points4D);
        
        for (int i = 0; i < _curFeatureMask.size(); i++) {
            // Points4D is in homogeneous coordinates.
            double depth = points4D.at<double>(2,i) / points4D.at<double>(3,i);

            // If the feature is matched before or the corresponding 3D point is too far away (i.e. less accuracy) w.r.t current camera, it will not be added.
            if (!_curFeatureMask[i] || depth > _maxDepth) continue;

            // The triangulated points coordinates are w.r.t current camera frame, should be converted to world frame.
            Eigen::Vector3d point_wrt_cam = Eigen::Vector3d(points4D.at<double>(0,i) / points4D.at<double>(3,i),
                                                            points4D.at<double>(1,i) / points4D.at<double>(3,i),
                                                            depth);

            // _pMap->getBodyPose() gives the transformation from current body frame to world frame, T_WB.
            // _pCameraModel->_T_BC gives the pre-calibrated transformation from camera to body/imu frame.
            Eigen::Vector3d position = _pMap->getBodyPose() * _pCameraModel->_T_BC * point_wrt_cam;

            #ifdef USE_VIEWER
            _pMap->_pViewer->pushLandmark(position(0), position(1), position(2));
            #endif

            // Insert new features.
            _features[_featureID] = std::make_shared<Feature>(_curPixelsL[i], _curDescriptorsL.row(i), _curDescriptorsR.row(i), position, 0);
            
            // Add these new features' descriptors to _hist for the convenience of next external matching.
            _histFeatureIDs.push_back(_featureID);
            _histDescriptorsL.push_back(_curDescriptorsL.row(i));
            _histDescriptorsR.push_back(_curDescriptorsR.row(i));

            // Feature seen by this frame.
            _pMap->_frames.back().push_back(std::make_pair(_curPixelsL[i], position));

            _featureID++;
            
            insertCount++;
        }
        if (_verbose) std::cout << "# 3D points within 20 meters: " << insertCount++ << std::endl;

        _pMap->_frames.resize(_pMap->_frames.size() + 1);
    }

    if (_verbose) std::cout << "# features in pool after updaing: " << _features.size() << " (" << insertCount << " features inserted, " << eraseCount << " too old features erased)" << std::endl;

    _frameID++;
}

// void FeatureTracker::extractOrb(int flag, const cv::Mat& img) {
//     if (flag == 0) // left
//         _orbLeft->detectAndCompute(img, cv::noArray(), _keypointsL, _descriptorsL);
//     else // right
//         _orbRight->detectAndCompute(img, cv::noArray(), _keypointsR, _descriptorsR);
// }

bool FeatureTracker::structFromMotion(const cv::Mat& grayLeft, const cv::Mat& grayRight, Eigen::Vector3d& r, Eigen::Vector3d& p, const bool atBeginning) {
    cv::Mat imgLeft, imgRight;
    cv::remap(grayLeft, imgLeft, _pCameraModel->_rmap[0][0], _pCameraModel->_rmap[0][1], cv::INTER_LINEAR);
    cv::remap(grayRight, imgRight, _pCameraModel->_rmap[1][0], _pCameraModel->_rmap[1][1], cv::INTER_LINEAR);

    if (atBeginning) { 
        _orb->detectAndCompute(imgLeft,  cv::noArray(), _refKeypointsL, _refDescriptorsL);
        return false;
    }
    
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    // _orb->detectAndCompute(imgLeft,  _mask, keypointsL, descriptorsL);
    // _orb->detectAndCompute(imgRight, _mask, keypointsR, descriptorsR);
    _orb->detectAndCompute(imgLeft,  cv::noArray(), keypointsL, descriptorsL);
    _orb->detectAndCompute(imgRight, cv::noArray(), keypointsR, descriptorsR);
    // std::thread orbLeft(&FeatureTracker::extractOrb, this, 0, imgLeft);
    // std::thread orbRight(&FeatureTracker::extractOrb, this, 1, imgRight);
    // orbLeft.join();
    // orbRight.join();

    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force Mathcer
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsL, descriptorsR, matches);
    float minDist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;

    // cv::Mat out;
    // cv::drawMatches(imgLeft, _keypointsL, imgRight, _keypointsR, matches, out);
    // std::cout << _keypointsL.size() << std::endl;
    // cv::imshow("out", out);
    // cv::waitKey(0);

    // Only keep good matches, for pixel (ul, vl) and (ur, vr), |vl-vr| should be small enough since the image has been rectified.
    std::vector<cv::Point2d> pixelsL, pixelsR;
    std::vector<int> indexL;
    for (auto& m : matches) {
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist) && std::abs(keypointsL[m.queryIdx].pt.y - keypointsR[m.trainIdx].pt.y) < _maxVerticalPixelDist) {
            pixelsL.push_back(keypointsL[m.queryIdx].pt);
            pixelsR.push_back(keypointsR[m.trainIdx].pt);
            indexL.push_back(m.queryIdx);
        }
    }

    // Removing matches of big descriptor distance maybe not enough. To root out outliers further, use 2D-2D RANSAC.
    cv::Mat ransacMask;
    cv::findFundamentalMat(pixelsL, pixelsR, ransacMask);
    std::vector<cv::Point2d> goodPixelsL, goodPixelsR;
    cv::Mat goodDescriptorsL;
    for (int i = 0; i < ransacMask.rows; i++) {
        if (ransacMask.at<bool>(i)) {
            goodPixelsL.push_back(pixelsL[i]);
            goodPixelsR.push_back(pixelsR[i]);
            goodDescriptorsL.push_back(descriptorsL.row(indexL[i]));
        }
    }

    // Triangulate current frame's keypoints.
    cv::Mat points4D;
    cv::triangulatePoints(_pCameraModel->_P1, _pCameraModel->_P2, goodPixelsL, goodPixelsR, points4D);
    std::vector<cv::Point3d> objectPoints;

    // Match with reference keyframe.
    matches.clear();
    matcher.match(goodDescriptorsL, _refDescriptorsL, matches);
    minDist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    // Only keep good matches.
    std::vector<cv::Point2d> imagePoints;
    for (auto& m : matches) {
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
            // Points4D (4x1) is in homogeneous coordinates.
            int i = m.queryIdx;
            double depth = points4D.at<double>(2,i) / points4D.at<double>(3,i);
            // Discard points that are too far away (i.e. less accurate) w.r.t current camera.
            if (depth > _maxDepth) continue;
            
            imagePoints.push_back(_refKeypointsL[m.trainIdx].pt);
            objectPoints.push_back(cv::Point3d(points4D.at<double>(0,i) / points4D.at<double>(3,i), points4D.at<double>(1,i) / points4D.at<double>(3,i), depth));
        }
    }

    auto start = std::chrono::steady_clock::now();
    cv::Mat rvec, tvec;
    cv::solvePnPRansac(objectPoints, imagePoints, _pCameraModel->_K_L, cv::noArray(), rvec, tvec);
    cv::cv2eigen(rvec, r);
    cv::cv2eigen(tvec, p);
    auto end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "solvePnPRansac elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;

    // std::cout << r.norm() << std::endl << p.norm() << std::endl;
    // if either rotation or translation is significant, keep this frame.
    if (r.norm() > _minRotation || p.norm() > _minTranslation) {
        _refKeypointsL = keypointsL;
        _refDescriptorsL = descriptorsL;
        return true;
    }
    return false;
}

} // namespace cfsd