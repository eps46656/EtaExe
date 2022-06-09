#include <iostream>

#include "math/Matrix.cuh"
#include "math/Vector2.cuh"
#include "math/Vector3.cuh"
#include "math/Vector4.cuh"
#include "render/QuestAManager.cu"
#include "render/QuestBManager.cu"
#include "Printer.cuh"

unsigned int SetRandom(unsigned int seed) {
    ::srand(seed);
    return seed;
}

unsigned int SetRandom() {
    time_t t;
    unsigned int seed((long int)time(&t));
    ::srand(seed);
    return seed;
}

int Random() { return ::rand(); }

int Random(int lower, int upper) { return Random() % (upper - lower) + lower; }

double RandomNum(double lower, double upper) {
    double b(4096);
    return Random(lower * b, upper * b) / b;
}

double RandomNormalNum() { return RandomNum(-256, 256); }

double sq(double x) { return x * x; }

int main() {
    eta::PrintFormat("seed: ", SetRandom(), '\n');

    // test_matrix();

    // testing_func();

    // eta::utility::Vector3 vec;

    // eta::render::QuestAManager quest_a_manager;

    std::string s("abc");

    eta::Print(s, "\n");

    return 0;
}