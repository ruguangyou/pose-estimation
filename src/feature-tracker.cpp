#include "cfsd/feature-tracker.hpp"

namespace cfsd {

FeatureTracker::FeatureTracker(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose) :
        _pMap(pMap), _pCameraModel(pCameraModel), _verbose(verbose), 
        _orbLeft(), _orbRight(), _ORBextractorLeft(), _ORBextractorRight(), _roi(), _imgSize(), 
        _curPixelsL(), _curPixelsR(), _curKeypointsL(), _curKeypointsR(), _curDescriptorsL(), _curDescriptorsR(), _curFeatureMask(), 
        _histFeatureIDs(), _histKeypointsL(), _histKeypointsR(), _histDescriptorsL(), _histDescriptorsR(), 
        _refKeypointsL(), _refDescriptorsL(), _matchedFeatureIDs(), _pFeatures() {
    
    _matchRatio = Config::get<float>("matchRatio");

    _minMatchDist = Config::get<float>("minMatchDist");

    _maxVerticalPixelDist = Config::get<float>("maxVerticalPixelDist");

    _maxFeatureAge = Config::get<int>("maxFeatureAge");

    _maxDepth = Config::get<double>("maxDepth");
    
    if (_verbose) { std::cout << "Setup ORB detector" << std::endl; }
    _cvORB = Config::get<bool>("cvORB");
    int numberOfFeatures = Config::get<int>("numberOfFeatures");
    float scaleFactor = Config::get<float>("scaleFactor");
    int levelPyramid = Config::get<int>("levelPyramid");
    
    if (_cvORB) { // use ORB of OpenCV
        int edgeThreshold = Config::get<int>("edgeThreshold");
        int patchSize = Config::get<int>("patchSize");
        int fastThreshold = Config::get<int>("fastThreshold");
        int gridRow = Config::get<int>("gridRow");
        int gridCol = Config::get<int>("gridCol");
        if (Config::get<int>("scoreType") == 0) {
            _orbLeft = cv::ORB::create(numberOfFeatures/(gridRow*gridCol), scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
            _orbRight = cv::ORB::create(numberOfFeatures/(gridRow*gridCol), scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
        }
        else {
            _orbLeft = cv::ORB::create(numberOfFeatures/(gridRow*gridCol), scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
            _orbRight = cv::ORB::create(numberOfFeatures/(gridRow*gridCol), scaleFactor, levelPyramid, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
        }
    }
    else { // use ORB of ORB_SLAM2
        int iniThFAST = Config::get<int>("iniThFAST");
        int minThFAST = Config::get<int>("minThFAST");
        _ORBextractorLeft = new ORB_SLAM2::ORBextractor(numberOfFeatures, scaleFactor, levelPyramid, iniThFAST, minThFAST);
        _ORBextractorRight = new ORB_SLAM2::ORBextractor(numberOfFeatures, scaleFactor, levelPyramid, iniThFAST, minThFAST);
    }

    _minRotation = Config::get<double>("sfmRotation");
    _minTranslation = Config::get<double>("sfmTranslation");

    _solvePnP = Config::get<int>("solvePnP");

    #ifdef CFSD
    _cropOffset = Config::get<int>("h1");
    _roi.x = 0;
    _roi.y = _cropOffset;
    _roi.width = Config::get<int>("imageWidth");
    _roi.height = Config::get<int>("h2") - _cropOffset;
    #endif
}

void FeatureTracker::orbDetectWithGrid(int flag, const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    cv::Ptr<cv::ORB> orb = (flag == 0) ? _orbLeft : _orbRight;
    int gridRow = Config::get<int>("gridRow");
    int gridCol = Config::get<int>("gridCol");
    for (int r = 0; r < gridRow; r++) {
        for (int c = 0; c < gridCol; c++) {
            cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
            for (int i = 0; i < img.rows / gridRow; i++)
                for (int j = 0; j < img.cols / gridCol; j++)
                    mask.at<char>(img.rows/gridRow * r + i, img.cols/gridCol * c + j) = (char)255;
            std::vector<cv::KeyPoint> ks;
            cv::Mat ds;
            orb->detectAndCompute(img, mask, ks, ds);
            keypoints.insert(keypoints.end(), ks.begin(), ks.end());
            if (descriptors.empty()) descriptors = ds;
            else cv::vconcat(descriptors, ds, descriptors);
        }
    }
}

void FeatureTracker::extractORB(int flag, const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    if (flag == 0)
        (*_ORBextractorLeft)(img, cv::Mat(), keypoints, descriptors);
    else
        (*_ORBextractorRight)(img, cv::Mat(), keypoints, descriptors);
}

bool FeatureTracker::processImage(const cv::Mat& grayLeft, const cv::Mat& grayRight, cv::Mat& descriptorsMat) {
    cv::Mat imgLeft, imgRight;
    #ifdef CFSD
    imgLeft = grayLeft(_roi);
    imgRight = grayRight(_roi);
    #else
    imgLeft = grayLeft;
    imgRight = grayRight;
    #endif
    _imgSize = imgLeft.size();

    // cv::Mat rectified;
    // cv::hconcat(imgLeft, imgRight, rectified);
    // cv::imshow("rectified", rectified);
    // cv::waitKey(0);
    
    _curPixelsL.clear();
    _curPixelsR.clear();
    _curKeypointsL.clear();
    _curKeypointsR.clear();
    _curDescriptorsL = cv::Mat();
    _curDescriptorsR = cv::Mat();
    
    auto start = std::chrono::steady_clock::now();
    internalMatch(imgLeft, imgRight, descriptorsMat);
    auto end = std::chrono::steady_clock::now();
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

void FeatureTracker::internalMatch(const cv::Mat& imgLeft, const cv::Mat& imgRight, cv::Mat& descriptorsMat, const bool useRANSAC) {
    auto start = std::chrono::steady_clock::now();
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    if (_cvORB) {
        std::thread orbLeft(&FeatureTracker::orbDetectWithGrid, this, 0, imgLeft, std::ref(keypointsL), std::ref(descriptorsL));
        std::thread orbRight(&FeatureTracker::orbDetectWithGrid, this, 1, imgRight, std::ref(keypointsR), std::ref(descriptorsR));
        orbLeft.join();
        orbRight.join();
    }
    else {
        std::thread orbLeft(&FeatureTracker::extractORB, this, 0, imgLeft, std::ref(keypointsL), std::ref(descriptorsL));
        std::thread orbRight(&FeatureTracker::extractORB, this, 1, imgRight, std::ref(keypointsR), std::ref(descriptorsR));
        orbLeft.join();
        orbRight.join();
    }
    auto end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "orb detection elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;

    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force Mathcer
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsL, descriptorsR, matches);

    std::vector<bool> vbInliers;
    // Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence.
	GmsMatcher gms(keypointsL, _imgSize, keypointsR, _imgSize, matches);
	int num_inliers = gms.GetInlierMask(vbInliers, false, false);
	std::cout << "(internal) GMS matching inliers: " << num_inliers << ", all matches: " << matches.size() << std::endl;

	// Collect inlier matches.
	for (size_t i = 0; i < vbInliers.size(); ++i) {
		if (vbInliers[i] == true) {
            auto& m = matches[i];
            _curPixelsL.push_back(keypointsL[m.queryIdx].pt);
            _curPixelsR.push_back(keypointsR[m.trainIdx].pt);
            _curKeypointsL.push_back(keypointsL[m.queryIdx]);
            _curKeypointsR.push_back(keypointsR[m.trainIdx]);
            _curDescriptorsL.push_back(descriptorsL.row(m.queryIdx));
            _curDescriptorsR.push_back(descriptorsR.row(m.trainIdx));
		}
	}

    // Use all descriptors in bag-of-words model.
    descriptorsMat = descriptorsL;
    // Use well matched descriptors in bag-of-words model.
    // descriptorsMat = _curDescriptorsL;
}

void FeatureTracker::externalTrack(const bool useRANSAC) {
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    
    // At initializing step, there is no available features in the list yet, so all features detected in the first frame will be added to the list.
    // The first frame will be set as keyframe as well.
    if (_pFeatures.empty()) {
        // curFeatureMask all true.
        return;
    }

    int rightCount = 0;

    if (_verbose) {
        std::cout << "# histDescriptorsL: " << _histDescriptorsL.rows << std::endl
                  << "#  curDescriptorsL: " << _curDescriptorsL.rows << std::endl;
    }

    // Store the correspondence map <queryIdx, trainIdx> of 'left' matching.
    std::map<int,int> mapCurHist;

    // Brute Force Mathcer: for each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one.
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matchesL;
    matcher.match(_curDescriptorsL, _histDescriptorsL, matchesL);
    std::vector<bool> vbInliersL;
    GmsMatcher gmsL(_curKeypointsL, _imgSize, _histKeypointsL, _imgSize, matchesL);
	int num_inliers = gmsL.GetInlierMask(vbInliersL, false, false);
	std::cout << "(external left) GMS matching inliers: " << num_inliers << ", all matches: " << matchesL.size() << std::endl;

    // Collect inlier matches.
	for (size_t i = 0; i < vbInliersL.size(); ++i)
		if (vbInliersL[i] == true)
            mapCurHist[matchesL[i].queryIdx] = matchesL[i].trainIdx;

    // float minDist;
    // minDist = std::min_element(matchesL.begin(), matchesL.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    // if (useRANSAC) {
    //     std::vector<cv::Point2d> pixelsCur, pixelsHist;
    //     std::vector<int> indexCur, indexHist;
        
    //     for (auto& m : matchesL) {
    //         // Only consider those good matches.
    //         if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
    //             pixelsCur.push_back(_curPixelsL[m.queryIdx]);
    //             pixelsHist.push_back(_pFeatures[_histFeatureIDs[m.trainIdx]]->pixelL);
    //             indexCur.push_back(m.queryIdx);
    //             indexHist.push_back(m.trainIdx);
    //         }
    //     }

    //     //TODO............what if there is zero match?

    //     // To root out outliers, use 2D-2D RANSAC.
    //     start = std::chrono::steady_clock::now();
    //     cv::Mat ransacMask;
    //     cv::findFundamentalMat(pixelsCur, pixelsHist, ransacMask);
    //     for (int i = 0; i < ransacMask.rows; i++) {
    //         if (ransacMask.at<bool>(i))
    //             mapCurHist[indexCur[i]] = indexHist[i];
    //     }
    //     end = std::chrono::steady_clock::now();
    //     if (_verbose) {
    //         std::cout << "external findFundamentalMat with RANSAC elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
    //         std::cout << "# left cur-hist matches: " << matchesL.size() << std::endl;
    //         std::cout << "# left cur-hist matches after hamming distance selection: " << pixelsCur.size() << std::endl;
    //         std::cout << "# left cur-hist matches after RANSAC: " << mapCurHist.size() << std::endl;
    //     }
    // }
    // else {
    //     for (auto& m : matchesL)
    //         // Only consider those good matches.
    //         if (m.distance < std::max(_matchRatio * minDist, _minMatchDist))
    //             mapCurHist[m.queryIdx] = m.trainIdx;
    //     if (_verbose) std::cout << "# left cur-hist matches after hamming distance selection: " << mapCurHist.size() << std::endl;
    // }
    
    // Search the correspondence of 'right' matching with 'left' matching.
    _matchedFeatureIDs.clear();
    std::vector<size_t>& mapPointIDs = _pMap->_pKeyframes.back()->mapPointIDs;
    cv::Mat& descriptors = _pMap->_pKeyframes.back()->descriptors;

    std::vector<cv::DMatch> matchesR;
    matcher.match(_curDescriptorsR, _histDescriptorsR, matchesR);
    std::vector<bool> vbInliersR;
    GmsMatcher gmsR(_curKeypointsR, _imgSize, _histKeypointsR, _imgSize, matchesR);
	num_inliers = gmsR.GetInlierMask(vbInliersR, false, false);
	std::cout << "(external right) GMS matching inliers: " << num_inliers << ", all matches: " << matchesR.size() << std::endl;

    // Collect inlier matches.
    std::unordered_map<size_t, bool> uniqueFeature;
	for (size_t i = 0; i < vbInliersR.size(); ++i) {
		if (vbInliersR[i] == true) {
            rightCount++;

            auto& m = matchesR[i];
            // Perform circular matching; only keep matches that are really good.
            auto search = mapCurHist.find(m.queryIdx);
            if (search != mapCurHist.end() && search->second == m.trainIdx) {
                // Satisfy circular matching, i.e. curLeft <=> histLeft <=> histRight <=> curRight <=> curLeft, the age of history features will increase 1.
                size_t featureID = _histFeatureIDs[m.trainIdx];
                
                // Avoid adding repeated elements, which is due to one orb keypoint can be matched more than one time during matching.
                if (uniqueFeature.find(featureID) != uniqueFeature.end()) continue;
                uniqueFeature[featureID] = true;
                
                // Some unnecessary map points might have been erased.
                if (_pMap->_pMapPoints.find(featureID) == _pMap->_pMapPoints.end()) continue;
                
                _pMap->_pMapPoints[featureID]->addPixel(_frameID, _curPixelsL[m.queryIdx]);
                mapPointIDs.push_back(featureID);
                descriptors.push_back(_curDescriptorsL.row(m.queryIdx));
                _matchedFeatureIDs.push_back(featureID);
                // Will not be added as new features.
                _curFeatureMask[m.queryIdx] = false;
            }
        }
    }
    
    // minDist = std::min_element(matchesR.begin(), matchesR.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    // std::unordered_map<size_t, bool> uniqueFeature;
    // for (auto& m : matchesR) {
    //     // Only consider those good matches.
    //     if (m.distance < std::max(_matchRatio * minDist, _minMatchDist)) {
    //         rightCount++;
            
    //         // Perform circular matching; only keep matches that are really good.
    //         auto search = mapCurHist.find(m.queryIdx);
    //         if (search != mapCurHist.end() && search->second == m.trainIdx) {
    //             // Satisfy circular matching, i.e. curLeft <=> histLeft <=> histRight <=> curRight <=> curLeft, the age of history features will increase 1.
    //             size_t featureID = _histFeatureIDs[m.trainIdx];
                
    //             // Avoid adding repeated elements, which is due to one orb keypoint can be matched more than one time during matching.
    //             if (uniqueFeature.find(featureID) != uniqueFeature.end()) continue;
    //             uniqueFeature[featureID] = true;
                
    //             // Some unnecessary map points might have been erased.
    //             if (_pMap->_pMapPoints.find(featureID) == _pMap->_pMapPoints.end()) continue;
                
    //             _pMap->_pMapPoints[featureID]->addPixel(_frameID, _curPixelsL[m.queryIdx]);
    //             mapPointIDs.push_back(featureID);
    //             descriptors.push_back(_curDescriptorsL.row(m.queryIdx));
    //             _matchedFeatureIDs.push_back(featureID);
    //             // Will not be added as new features.
    //             _curFeatureMask[m.queryIdx] = false;
    //         }
    //     }
    // }

    std::cout << "# circular matches: " << _matchedFeatureIDs.size() << std::endl;
}

void FeatureTracker::featurePoolUpdate(const long& imgTimestamp) {
    _pMap->_pKeyframes.back()->timestamp = imgTimestamp;

    // If current frame is keyframe, the not matched features in current frame will be viewed as new features and be added into feature pool.
    // The number of new and old features in the pool should be well balanced.
    int eraseCount = 0;
    int insertCount = 0;
    if (_verbose) std::cout << "# features in pool before updating: " << _pFeatures.size() << std::endl;

    // _age minus 1, will add 2 later.
    for (int i = 0; i < _matchedFeatureIDs.size(); i++) {
        _pFeatures[_matchedFeatureIDs[i]]->age -= 1;
    }

    // age add 2 for all features, then erase too old features, and put the rest into _histDescriptors.
    _histFeatureIDs.clear();
    _histKeypointsL.clear();
    _histKeypointsR.clear();
    _histDescriptorsL = cv::Mat();
    _histDescriptorsR = cv::Mat();
    auto f = _pFeatures.begin();
    while (f != _pFeatures.end()) {
        f->second->age += 2;
        if (f->second->age > _maxFeatureAge) {
            f = _pFeatures.erase(f);
            eraseCount++;
        }
        else {
            _histFeatureIDs.push_back(f->first);
            _histKeypointsL.push_back(f->second->keypointL);
            _histKeypointsR.push_back(f->second->keypointR);
            _histDescriptorsL.push_back(f->second->descriptorL);
            _histDescriptorsR.push_back(f->second->descriptorR);
            f++; // DON'T increment after erasing
        }
    }

    // Triangulate current frame's keypoints.
    cv::Mat points4D;
    cv::triangulatePoints(_pCameraModel->_P1, _pCameraModel->_P2, _curPixelsL, _curPixelsR, points4D);
    
    std::vector<size_t>& mapPointIDs = _pMap->_pKeyframes.back()->mapPointIDs;
    cv::Mat& descriptors = _pMap->_pKeyframes.back()->descriptors;

    // _pMap->getBodyPose() gives the transformation from current body frame to world frame, T_WB.
    Sophus::SE3d T_WB = _pMap->getBodyPose();
    
    for (int i = 0; i < _curFeatureMask.size(); i++) {
        // Points4D is in homogeneous coordinates.
        double depth = points4D.at<double>(2,i) / points4D.at<double>(3,i);

        // If the feature is matched before or the corresponding 3D point is too far away (i.e. less accuracy) w.r.t current camera, it will not be added.
        if (!_curFeatureMask[i] || depth < 0.1 || depth > _maxDepth) continue;

        // The triangulated points coordinates are w.r.t current camera frame, should be converted to world frame.
        Eigen::Vector3d point_wrt_cam = Eigen::Vector3d(points4D.at<double>(0,i) / points4D.at<double>(3,i),
                                                        points4D.at<double>(1,i) / points4D.at<double>(3,i),
                                                        depth);

        // _pCameraModel->_T_BC gives the pre-calibrated transformation from camera to body/imu frame.
        Eigen::Vector3d position = T_WB * _pCameraModel->_T_BC * point_wrt_cam;
        
        // The map point is triangulated from this frame.
        _pMap->_pMapPoints[_featureID] = std::make_shared<MapPoint>(position, _frameID, _curPixelsL[i]);
        mapPointIDs.push_back(_featureID);
        descriptors.push_back(_curDescriptorsL.row(i));

        // Insert new features.
        _pFeatures[_featureID] = std::make_shared<Feature>(_frameID, _curPixelsL[i], _curKeypointsL[i], _curKeypointsR[i], _curDescriptorsL.row(i), _curDescriptorsR.row(i), 0);
        
        // Add these new features' descriptors to _hist for the convenience of next external matching.
        _histFeatureIDs.push_back(_featureID);
        _histKeypointsL.push_back(_curKeypointsL[i]);
        _histKeypointsR.push_back(_curKeypointsR[i]);
        _histDescriptorsL.push_back(_curDescriptorsL.row(i));
        _histDescriptorsR.push_back(_curDescriptorsR.row(i));

        #ifdef USE_VIEWER
        _pMap->_pViewer->pushLandmark(_frameID, position);
        #endif

        _featureID++;
        insertCount++;
    }
    _frameID++;

    if (_verbose) {
        std::cout << "# 3D points within " << _maxDepth << " meters: " << insertCount << std::endl;
        std::cout << "# features in pool after updaing: " << _pFeatures.size() << " (" << insertCount << " features inserted, " << eraseCount << " too old features erased)" << std::endl;
    }
}

bool FeatureTracker::structFromMotion(const cv::Mat& grayLeft, const cv::Mat& grayRight, Eigen::Vector3d& r, Eigen::Vector3d& p, const bool atBeginning) {
    cv::Mat imgLeft, imgRight;
    #ifdef CFSD
    imgLeft = grayLeft(_roi);
    imgRight = grayRight(_roi);
    #else
    imgLeft = grayLeft;
    imgRight = grayRight;
    #endif

    if (atBeginning) { 
        // _orb->detectAndCompute(imgLeft, cv::noArray(), _refKeypointsL, _refDescriptorsL);
        if (_cvORB)
            orbDetectWithGrid(0, imgLeft, _refKeypointsL, _refDescriptorsL);
        else
            (*_ORBextractorLeft)(imgLeft, cv::Mat(), _refKeypointsL, _refDescriptorsL);
        return false;
    }
    
    std::vector<cv::KeyPoint> keypointsL, keypointsR;
    cv::Mat descriptorsL, descriptorsR;
    if (_cvORB) {
        std::thread orbLeft(&FeatureTracker::orbDetectWithGrid, this, 0, imgLeft, std::ref(keypointsL), std::ref(descriptorsL));
        std::thread orbRight(&FeatureTracker::orbDetectWithGrid, this, 1, imgRight, std::ref(keypointsR), std::ref(descriptorsR));
        orbLeft.join();
        orbRight.join();
    }
    else {
        std::thread orbLeft(&FeatureTracker::extractORB, this, 0, imgLeft, std::ref(keypointsL), std::ref(descriptorsL));
        std::thread orbRight(&FeatureTracker::extractORB, this, 1, imgRight, std::ref(keypointsR), std::ref(descriptorsR));
        orbLeft.join();
        orbRight.join();
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force Mathcer
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsL, descriptorsR, matches);
    float minDist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;

    // cv::Mat out;
    // cv::drawMatches(imgLeft, keypointsL, imgRight, keypointsR, matches, out);
    // std::cout << matches.size() << std::endl;
    // cv::imshow("matches", out);
    // cv::waitKey(0);

    // Only keep good matches, for pixel (ul, vl) and (ur, vr), |vl-vr| should be small enough since the image has been rectified.
    std::vector<cv::Point2d> pixelsL, pixelsR;
    std::vector<int> indexL;
    // std::vector<cv::DMatch> good_matches;
    for (auto& m : matches) {
        if (m.distance < std::max(_matchRatio * minDist, _minMatchDist) && std::abs(keypointsL[m.queryIdx].pt.y - keypointsR[m.trainIdx].pt.y) < _maxVerticalPixelDist) {
            pixelsL.push_back(keypointsL[m.queryIdx].pt);
            pixelsR.push_back(keypointsR[m.trainIdx].pt);
            indexL.push_back(m.queryIdx);
            // good_matches.push_back(m);
        }
    }

    // cv::drawMatches(imgLeft, keypointsL, imgRight, keypointsR, good_matches, out);
    // std::cout << good_matches.size() << std::endl;
    // cv::imshow("good matches", out);
    // cv::waitKey(0);

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
            if (depth < 0.1 || depth > _maxDepth) continue;
            
            imagePoints.push_back(_refKeypointsL[m.trainIdx].pt);
            objectPoints.push_back(cv::Point3d(points4D.at<double>(0,i) / points4D.at<double>(3,i), points4D.at<double>(1,i) / points4D.at<double>(3,i), depth));
        }
    }

    auto start = std::chrono::steady_clock::now();
    cv::Mat rvec, tvec;
    cv::solvePnPRansac(objectPoints, imagePoints, _pCameraModel->_K_L, cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, cv::noArray(), _solvePnP);
    cv::cv2eigen(rvec, r);
    cv::cv2eigen(tvec, p);
    auto end = std::chrono::steady_clock::now();
    if (_verbose) std::cout << "solvePnPRansac elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;

    // std::cout << r.norm() << std::endl << p.norm() << std::endl;
    // If either rotation or translation is significant, keep this frame.
    if (r.norm() > _minRotation || p.norm() > _minTranslation) {
        _refKeypointsL = keypointsL;
        _refDescriptorsL = descriptorsL;
        return true;
    }
    return false;
}

} // namespace cfsd