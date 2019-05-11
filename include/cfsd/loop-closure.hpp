#ifndef LOOP_CLOSURE_HPP
#define LOOP_CLOSURE_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"

// DBoW2 include path: /usr/local/include/DBoW2
#include <DBoW2.h>

namespace cfsd {

class LoopClosure {
  public:
    LoopClosure(const bool verbose);

    // Change the structure descriptors, put each row of Mat into vector.
    void changeStructure(const cv::Mat& mat, std::vector<cv::Mat>& vec);

    void addImage(const cv::Mat& descriptorsMat);

    void detectLoop(const cv::Mat& descriptorsMat);

  private:
    OrbDatabase _db;
};

} // namespace cfsd

#endif // LOOP_CLOSURE_HPP