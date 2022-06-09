#include <rgba.cu>
#include <interp.cu>

int main() {
    //

    unsigned int w(1920);
    unsigned int h(1080);

    eta::rgba::Image img(w, h);

    for (unsigned int i(0); i < w; ++i) {
        for (unsigned int j(0); j < h; ++j) {
            unsigned char c(eta::interp::linear(0, w, 0, 255, i));
            unsigned char d(eta::interp::linear(0, h, 0, 255, j));
            img.pixel(i, j).r = c;
            img.pixel(i, j).g = c;
            img.pixel(i, j).b = d;
            img.pixel(i, j).a = 255;
        }
    }

    // std::ofstream ofs("/home/eps/desktop/k.rgba");

    // rgba::output_rgba(ofs, w, h, pixels);

    img.write("/home/eps/desktop/k.rgba");

    return 0;
}