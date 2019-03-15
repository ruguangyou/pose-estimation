#ifndef IMAGE_READER_HPP
#define IMAGE_READER_HPP

#include "cfsd/common.hpp"
#include "cfsd/config.hpp"
#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

namespace cfsd {

class ImageReader {
  public:
    ImageReader(cluon::OD4Session& od4, const std::string& sharedMemoryPath, const bool verbose);

    // Execute in an independent thread for continuously reading image data from shared memory.
    void run();

    // Grab data from the queue and pop out the grabbed data.
    void grabData(cv::Mat& pImg, long& timestamp);

    int getQueueSize();

    bool isSharedMemoryValid(int& retCode);

  private:
    bool _verbose;

    // Interface to OD4 session.
    cluon::OD4Session& _od4;

    // Smart point (unique_ptr) to shared memory.
    std::unique_ptr<cluon::SharedMemory> _pSharedMemory;

    // Resolution of the image from camera.
    int _height, _width;

    // Images read from shared memory are stored in a queue.
    std::queue<cv::Mat> _imgQueue;

    // Timestamps of moments when data is read from shared memory.
    std::queue<long> _timestamps;

    // The number of images stayed in the queue.
    int _queueSize;

    // Mutex, prevents data being edited by multi-threads.
    std::mutex _dataMutex;
};

}

#endif // IMAGE_READER_HPP