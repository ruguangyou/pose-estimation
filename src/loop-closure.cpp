#include "cfsd/loop-closure.hpp"

namespace cfsd {

LoopClosure::LoopClosure(const cfsd::Ptr<Map>& pMap, const cfsd::Ptr<Optimizer>& pOptimizer, const cfsd::Ptr<CameraModel>& pCameraModel, const bool verbose) : 
    _db(), _pMap(pMap), _pOptimizer(pOptimizer), _pCameraModel(pCameraModel), _verbose(verbose) {
    
    std::string vocFile = Config::get<std::string>("vocabulary");

    // Load the vocabulary from binary file.
    std::cout << "Loading orb vocabulary..." << std::endl;
    OrbVocabulary voc;
    auto start = std::chrono::steady_clock::now();
    voc.loadFromBinaryFile(vocFile);
    auto end = std::chrono::steady_clock::now();
    std::cout << "... done! elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl;
    
    // db creates a copy of the vocabulary, we may get rid of "voc" now.
    std::cout << "Creating vocabulary database..." << std::endl;
    _db.setVocabulary(voc, false, 0);
    std::cout << "... done! " << _db << std::endl << std::endl;

    _minFrameInterval = Config::get<int>("minFrameInterval");
    _minScore = Config::get<double>("minScore");

    _solvePnP = Config::get<int>("solvePnP");
}

void LoopClosure::run() {
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    
    while(true) {
        bool toCloseLoop;
        int wait, loopFrameID, curFrameID;
        {
            std::lock_guard<std::mutex> dataLock(_dataMutex);
            toCloseLoop = _toCloseLoop;
            wait = _wait;
            loopFrameID = _loopFrameID;
            curFrameID = _curFrameID;
        }

        if (toCloseLoop && wait == WINDOWSIZE) {
            std::cout << "Try to close loop with frame " << loopFrameID << " and frame " << curFrameID << std::endl << std::endl;
            Eigen::Vector3d r, p;
            start = std::chrono::steady_clock::now();
            if (computeLoopInfo(loopFrameID, curFrameID, r, p)) {
                end = std::chrono::steady_clock::now();
                std::cout << "Compute PnP for loop elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
                
                _pMap->pushLoopInfo(curFrameID, loopFrameID, r, p);

                start = std::chrono::steady_clock::now();
                _pOptimizer->loopCorrection(curFrameID);
                end = std::chrono::steady_clock::now();
                std::cout << "Loop correction elapsed time: " << std::chrono::duration<double, std::milli>(end-start).count() << "ms" << std::endl << std::endl;
                
                #ifdef USE_VIEWER
                _pMap->_pViewer->pushLoopConnection(loopFrameID, curFrameID);
                #endif
            }

            std::lock_guard<std::mutex> dataLock(_dataMutex);
            _toCloseLoop = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void LoopClosure::changeStructure(const cv::Mat& mat, std::vector<cv::Mat>& vec) {
    vec.resize(mat.rows);
    for(int i = 0; i < mat.rows; ++i)
        vec[i] = mat.row(i);
}

void LoopClosure::addImage(const cv::Mat& descriptorsMat) {
    std::vector<cv::Mat> descriptorsVec;
    changeStructure(descriptorsMat, descriptorsVec);
    
    std::lock_guard<std::mutex> dbLock(_dbMutex);
    _db.add(descriptorsVec);
}

int LoopClosure::detectLoop(const cv::Mat& descriptorsMat, const int& frameID) {
    std::vector<cv::Mat> descriptorsVec;
    changeStructure(descriptorsMat, descriptorsVec);

    DBoW2::QueryResults ret;
    {
        std::lock_guard<std::mutex> dbLock(_dbMutex);
        // The last parameter is max_id, means only querying entries with id <= max_id are returned in ret. < 0 means all
        _db.query(descriptorsVec, ret, 4, frameID-_minFrameInterval);
    }

    if (_verbose) std::cout << "Searching for current frame. " << ret << std::endl;

    bool findLoop = false;
    // ret[0] is the most similar image, so should set a higher score limit for it.
    if (frameID > _minFrameInterval && ret.size() > 0 && ret[0].Score > 3*_minScore) {
        // The rest are possible loop candidates.
        for (int i = 1; i < ret.size(); i++) {
            if (ret[i].Score > _minScore) {
                // Loop is found only if ret[0] and at least one ret[i] have scores higher than threshold.
                findLoop = true;
            }
        }
    }

    // Search the earliest keyframe that makes loop if there is any.
    // (earliest makes it more frames in between to optimize)
    int minFrameID = frameID;
    if (findLoop) {
        for (int i = 0; i < ret.size(); i++)
            if (ret[i].Id < minFrameID) 
                minFrameID = ret[i].Id;

        std::cout << "loop closure candidate frame id: " << minFrameID << ", current frame id: " << frameID << std::endl << std::endl;
    }

    return (minFrameID == frameID) ? -1 : minFrameID;
}

void LoopClosure::setToCloseLoop(const int& minLoopFrameID, const int& curFrameID) {
    std::lock_guard<std::mutex> dataLock(_dataMutex);
    _wait = 0;
    _toCloseLoop = true;
    _loopFrameID = minLoopFrameID;
    _curFrameID = curFrameID;
}

void LoopClosure::setToCloseLoop() {
    std::lock_guard<std::mutex> dataLock(_dataMutex);
    _wait++;
}

bool LoopClosure::computeLoopInfo(const int& loopFrameID, const int& curFrameID, Eigen::Vector3d& r, Eigen::Vector3d& p) {
    cfsd::Ptr<Keyframe>& loopFrame = _pMap->_pKeyframes[loopFrameID];
    cfsd::Ptr<Keyframe>& curFrame = _pMap->_pKeyframes[curFrameID];
    
    cv::BFMatcher matcher(cv::NORM_HAMMING); // Brute Force Mathcer
    std::vector<cv::DMatch> matches;
    matcher.match(loopFrame->descriptors, curFrame->descriptors, matches);

    if (matches.size() < 40) {
        std::cout << "Too few matches: " << matches.size() << std::endl;
        return false;
    }

    // Only keep good matches.
    float minDist = std::min_element(matches.begin(), matches.end(), [] (const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; })->distance;
    std::vector<cv::Point2d> imagePoints;
    std::vector<cv::Point3d> objectPoints;
    for (auto& m : matches) {
        if (m.distance < std::max(2.0f * minDist, 30.f)) {
            const size_t& mapPointID1 = loopFrame->mapPointIDs[m.queryIdx];
            const size_t& mapPointID2 = curFrame->mapPointIDs[m.trainIdx];
            
            // Some unnecessary map points might have been erased.
            if (_pMap->_pMapPoints.find(mapPointID1) == _pMap->_pMapPoints.end()) continue;
            
            const Eigen::Vector3d& position = _pMap->_pMapPoints[mapPointID1]->position;
            objectPoints.push_back(cv::Point3d(position(0), position(1), position(2)));

            // Some unnecessary map points might have been erased.
            if (_pMap->_pMapPoints.find(mapPointID2) == _pMap->_pMapPoints.end()) continue;

            imagePoints.push_back(_pMap->_pMapPoints[mapPointID2]->pixels[curFrameID]);
        }
    }

    if (imagePoints.size() < 15) {
        std::cout << "Too few points: " << imagePoints.size() << std::endl;
        return false;
    }

    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(objectPoints, imagePoints, _pCameraModel->_K_L, cv::noArray(), rvec, tvec, false, 100, 8.0, 0.99, inliers, _solvePnP);

    if (inliers.rows < 10) {
        std::cout << "Too few inliers" << std::endl;
        return false;
    }
    std::cout << "Number of inliers: " << inliers.rows << std::endl;

    // r and p represent the rotation and translation from loop frame to current frame.
    cv::cv2eigen(rvec, r);
    cv::cv2eigen(tvec, p);
    
    return true;
}

} // namespace cfsd