#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include "cfsd/common.hpp"

namespace cfsd {

class Initializer {
  public:
    ~Initializer();
    static void setParameterFile (const std::string& filename);
    template <typename T> static T get (const std::string& key);
  
  private:
    static std::shared_ptr<Initializer> _initializer;
    cv::FileStorage _file;
    Initializer(); // private constructor makes a singleton
};

} // namespace cfsd

#endif // INITIALIZER_HPP
