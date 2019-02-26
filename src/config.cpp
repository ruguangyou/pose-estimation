#include "cfsd/config.hpp"

namespace cfsd {

Config::Config() {}

Config::~Config() {
    // close file when deconstructing
    if (_config->_file.isOpened()) {
        _config->_file.release();
    }
}

std::shared_ptr<Config> Config::_config = nullptr; // static data member initialization

void Config::setParameterFile(const std::string& filename) {
    if (_config == nullptr) {
        _config = std::shared_ptr<Config> (new Config());
    }
    _config->_file = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);

    if (_config->_file.isOpened() == false) {
        std::cerr << "Parameter file " << filename << " does not exist" << std::endl;
        _config->_file.release();
        return;
    }
}

template <typename T>
T Initializer::get(const std::string& key) {
    T value;
    _config->_file[key] >> value;
    return value;
}

} // namespace cfsd