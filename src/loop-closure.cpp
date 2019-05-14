#include "cfsd/loop-closure.hpp"

namespace cfsd {

LoopClosure::LoopClosure(const bool verbose) : _db() {
    
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
}

void LoopClosure::changeStructure(const cv::Mat& mat, std::vector<cv::Mat>& vec) {
    vec.resize(mat.rows);
    for(int i = 0; i < mat.rows; ++i)
        vec[i] = mat.row(i);
}

void LoopClosure::addImage(const cv::Mat& descriptorsMat) {
    std::vector<cv::Mat> descriptorsVec;

    changeStructure(descriptorsMat, descriptorsVec);
    
    _db.add(descriptorsVec);
}

int LoopClosure::detectLoop(const cv::Mat& descriptorsMat, const int& frameID) {
    std::vector<cv::Mat> descriptorsVec;

    changeStructure(descriptorsMat, descriptorsVec);

    DBoW2::QueryResults ret;

    // The last parameter is max_id, means only querying entries with id <= max_id are returned in ret. < 0 means all
    _db.query(descriptorsVec, ret, 4, frameID-_minFrameInterval);

    std::cout << "Searching for current frame. " << ret << std::endl;

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

        std::cout << "loop closure candidate, frame id: " << minFrameID << "\n\n\n\n\n\n\n\n\n\n\n\n" << std::endl;
    }

    return (minFrameID == frameID) ? -1 : minFrameID;
}

} // namespace cfsd