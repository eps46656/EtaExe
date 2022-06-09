#include <cstdlib>
#include <ctime>

long int SetRandom(long int seed) {
    srand48(seed);
    return seed;
}

long int SetRandom() {
    time_t t;
    long int seed{ (long int)time(&t) };
    SetRandom(seed);
    return seed;
}

unsigned long int Random() { return lrand48(); }

float Random_f() { return static_cast<float>(Random()) / (1 + Random()); }

float Random_f_zero_to_one() {
    unsigned long int a{ Random() };
    unsigned long int b{ 1 + Random() };
    return static_cast<float>(a % b) / b;
}

float Random_f(float lower, float upper) {
    return lower + (upper - lower) * Random_f_zero_to_one();
}

float Random_f_normal() { return Random_f(-100, 100); }