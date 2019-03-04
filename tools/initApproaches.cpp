#include <iostream>
#include <memory>

template<typename T>
using Ptr = std::shared_ptr<T>;

struct Foo {
    Foo(int x) : _x(x) {}
    int _x;
};

int main() {
    // std::shared_ptr<Foo> a = new Foo(1); // failed, conversion from 'Foo*' to 'shared_ptr<Foo>' is invalid
    std::shared_ptr<Foo> b{new Foo(1)}; // OK
    std::shared_ptr<Foo> c{new Foo{1}}; // OK
    std::shared_ptr<Foo> d{std::make_shared<Foo>(1)}; // OK
    std::shared_ptr<Foo> e = std::make_shared<Foo>(1); // OK
    // std::shared_ptr<Foo> f = std::make_shared<Foo>{1}; // failed

    Ptr<Foo> x{new Foo{2}};
    std::cout << x->_x << std::endl;
    
    return 0;
}