#ifndef MAP_HPP
#define MAP_HPP

#include "cfsd/common.hpp"
#include "cfsd/key-frame.hpp"

namespace cfsd {

class Map {
  public: 
    using Ptr = std::shared_ptr<Map>;

    Map();
    ~Map();
    Map(bool verbose, bool debug);
    
    static Map::Ptr create(bool verbose, bool debug);

  
  private:
    bool _verbose, _debug;

    
};

} // namespace cfsd

#endif // MAP_HPP