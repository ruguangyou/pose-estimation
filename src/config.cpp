#include "cfsd/config.hpp"

namespace cfsd {

Config::Config() : _file() {}

Config::~Config() {
    // close file when deconstructing
    if (_pConfig->_file.isOpened()) {
        _pConfig->_file.release();
    }
}

// static data member initialization
std::shared_ptr<Config> Config::_pConfig = nullptr;

void Config::setParameterFile(const std::string& filename) {
    if (_pConfig == nullptr) {
        _pConfig = std::shared_ptr<Config> (new Config());
        std::cout << "Open file " << filename << std::endl;
    }
    _pConfig->_file = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);

    if (_pConfig->_file.isOpened() == false) {
        std::cerr << "Parameter file " << filename << " does not exist" << std::endl;
        _pConfig->_file.release();
        return;
    }
}

} // namespace cfsd