#include "cfsd/map.hpp"

namespace cfsd {

Map::Map() {}

Map::~Map() {}

Map::Map(bool verbose, bool debug) : _verbose(verbose), _debug(debug) {}

Map::Ptr Map::create(bool verbose = false, bool debug = false) {
    return Map::Ptr(new Map(verbose, debug));
}


} // namespace cfsd