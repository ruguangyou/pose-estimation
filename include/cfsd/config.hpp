#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "cfsd/common.hpp"

namespace cfsd {

class Config {
  public:
    ~Config();
    
    static void setParameterFile(const std::string& filename);
    
    template <typename T>
    static T get(const std::string& key) {
        T value;
        _pConfig->_file[key] >> value;
        return value;
    }
  
  private:
    static std::shared_ptr<Config> _pConfig;
    cv::FileStorage _file;
    Config(); // private constructor makes a singleton
};

} // namespace cfsd

#endif // CONFIG_HPP