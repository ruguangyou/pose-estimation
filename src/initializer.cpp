#include "cfsd/initializer.hpp"

namespace cfsd {

Initializer::Initializer() {};
Initializer::~Initializer() {
    // close file when deconstructing
    if (_initializer->m_file.isOpened()) {
        _initializer->m_file.release();
    }
}

std::shared_ptr<Initializer> Initializer::_initializer = nullptr; // static data member initialization

void Initializer::setParameterFile (const std::string& filename) {
    if (_initializer == nullptr) {
        _initializer = std::shared_ptr<Initializer> (new Initializer);
    }
    _initializer->m_file = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);

    if (_initializer->m_file.isOpened() == false) {
        std::cerr << "Parameter file " << filename << " does not exist" << std::endl;
        _initializer->m_file.release();
        return;
    }
}

template <typename T>
T Initializer::get (const std::string& key) {
    T value;
    Initializer::_initializer->m_file[key] >> value;
    return value;
}

} // namespace cfsd