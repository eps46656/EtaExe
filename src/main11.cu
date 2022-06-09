#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <ctime>

#include "random.h"
#include "/media/D/EtaDevelop/Eta/Vector.cuh"
#include "/media/D/EtaDevelop/Eta/Matrix.cuh"
#include "/media/D/EtaDevelop/Eta/utils.cuh"

#include "/media/D/EtaDevelop/Eta/PointLight.cu"

using namespace eta;

template<typename T> struct C {
    void F(int x) { this->G(x); }

    void G(int x) { std::cout << x << " hello\n"; }
};

#define N (3)
#define M (3)

int main() {
    std::cout << "seed: " << SetRandom() << '\n';

    Vector<num, 3> v{ -4, 8, -3.5 };
    Vector<num, 3> u{ -5, 8, 6 };

    Matrix<num, N, M> m;

    for (int i{ 0 }; i < N; ++i) {
        for (int j{ 0 }; j < M; ++j) { m.data[i][j] = Random_f_normal(); }
    }

    Matrix<num, N, M> m_inv{ m.inv() };

    Matrix<num, N, M> I{ Matrix<num, N, M>::eye() };

    Matrix<num, N, M> I1{ m * m_inv };
    Matrix<num, N, M> I2{ m_inv * m };

    Print(m);
    Print(m_inv);
    Print(I);
    Print(I1);
    Print(I2);

    Print(I == I1, '\n');
    Print(I == I2, '\n');

    return 0;
}
