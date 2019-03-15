#include "cfsd/image-reader.hpp"

namespace cfsd {

ImageReader::ImageReader(cluon::OD4Session& od4, const std::string& sharedMemoryName, const bool verbose)
    : _od4(od4), _verbose(verbose), _queueSize(0) {
        _pSharedMemory = std::make_unique<cluon::SharedMemory>(sharedMemoryName);
        _height = Config::get<int>("imageHeight");
        _width = Config::get<int>("imageWidth");
        if (_verbose) { std::cout << "Image resolution: " << _width << "x" << _height << std::endl; }
}

void ImageReader::run() {
    while(_od4.isRunning()) {
        // cv::Mat img;
        // long timestamp;

        // Wait for a notification of a new frame.
        _pSharedMemory->wait();
        // Lock the shared memory.
        _pSharedMemory->lock();
        {
            // Copy image into cvMat structure. Be aware of that any code between lock/unlock is blocking the camera to 
            // provide the next frame. Thus, any computationally heavy algorithms should be placed outside lock/unlock.
            cv::Mat wrapped(_height, _width, CV_8UC4, _pSharedMemory->data());
            
            // img = wrapped.clone();
            cluon::data::TimeStamp TS = cluon::time::now();
            // timestamp = cluon::time::toMicroseconds(TS);

            // Lock the data member.
            std::lock_guard<std::mutex> lockData(_dataMutex);
            
            _imgQueue.push(wrapped.clone());
            
            _timestamps.push(cluon::time::toMicroseconds(TS));
            
            _queueSize++;
            
            if(_verbose) { std::cout << _queueSize << " images are waiting in the queue!" << std::endl; }

            #ifdef DEBUG_IMG
            if (_queueSize > 99) {
                std::cout << "Queue is full, stop reading!" << std::endl;
                break;
            }
            #endif
        }
        _pSharedMemory->unlock();

        // // Lock the data member.
        // std::lock_guard<std::mutex> lockData(_dataMutex);
        // _imgQueue.push(img);
        // _timestamps.push(timestamp);
        // _queueSize++;
    }
}

void ImageReader::grabData(cv::Mat& img, long& timestamp) {
    std::lock_guard<std::mutex> lockData(_dataMutex);
    img = _imgQueue.front();
    timestamp = _timestamps.front();
    _imgQueue.pop();
    _timestamps.pop();
    _queueSize--;
}

int ImageReader::getQueueSize() {
    std::lock_guard<std::mutex> lockData(_dataMutex);
    return _queueSize;
}

bool ImageReader::isSharedMemoryValid(int& retCode) {
    bool valid = _pSharedMemory && _pSharedMemory->valid();
    if (valid) {
        retCode = 0;
        std::clog << "Attached to shared memory '" << _pSharedMemory->name() << " (" << _pSharedMemory->size() << " bytes)." << std::endl;
    }
    else {
        retCode = 1;
        std::cerr << "Failed to attach to shared memory." << std::endl;
    }
    return valid;
}

} // namespace cfsd