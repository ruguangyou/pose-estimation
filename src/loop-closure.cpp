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

void LoopClosure::detectLoop(const cv::Mat& descriptorsMat) {
    std::vector<cv::Mat> descriptorsVec;

    changeStructure(descriptorsMat, descriptorsVec);

    DBoW2::QueryResults ret;

    // ret[0] is always the same image in this case, because we added it to the database. ret[1] is the second best match.
    _db.query(descriptorsVec, ret, 4);

    std::cout << "Searching for current frame. " << ret << std::endl;
}

} // namespace cfsd