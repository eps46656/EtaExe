#include <io/stl.cpp>

int main() {
    eta::io::stl::STL stl_f("/home/eps/desktop/Jump_Pack.stl");

    std::cout << "stl_f.tris.size(): " << stl_f.tris.size() << "\n";

    stl_f.write("/home/eps/desktop/Jump_Pack2.stl");

    return 0;
}