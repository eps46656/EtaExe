#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <math_constants.h>

// #include "render/TriangleBatch.cu"
#include "render/RayCasting.cu"

// #include "TriangleBatch.cu"

// #include "Camera.cu"

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

eta::num_t RandomNum(eta::num_t lower, eta::num_t upper) {
    eta::num_t b(4096);
    return Random(lower * b, upper * b) / b;
}

eta::num_t RandomNormalNum() { return RandomNum(-256, 256); }

/*
__global__ void F() {
    eta::num_t nan(-CUDART_NAN);
    eta::num_t inf(CUDART_INF);
    eta::num_t inf2(ETA__inf);
    ::printf("nan: %f, %x\n", nan, nan);
    ::printf("inf: %f, %x\n", inf, inf);
    ::printf("inf2: %f, %x\n", inf2, inf2);

    ::printf("nan < inf: %d\n", nan < inf);
    ::printf("nan < inf2: %d\n", nan < inf2);
    ::printf("nan > inf2: %d\n", nan > inf2);
    ::printf("nan == inf2: %d\n", nan == inf2);
    ::printf("nan == nan: %d\n", nan == nan);
    ::printf("inf < inf2: %d\n", inf < inf2);
    ::printf("inf == inf2: %d\n", inf == inf2);
    ::printf("inf2 < inf: %d\n", inf2 < inf);
    ::printf("inf == inf2: %d\n", inf == inf2);
}
*/

void NormalInitTriangle(eta::render::Triangle* tri) {
    tri->id = 1;
    tri->origin[0] = RandomNormalNum();
    tri->origin[1] = RandomNormalNum();
    tri->origin[2] = RandomNormalNum();
    tri->axis[0] = RandomNormalNum();
    tri->axis[1] = RandomNormalNum();
    tri->axis[2] = RandomNormalNum();
    tri->axis[3] = RandomNormalNum();
    tri->axis[4] = RandomNormalNum();
    tri->axis[5] = RandomNormalNum();
}

void NormalInitRayBatch(eta::render::RayBatch* ray_batch) {
    for (int i(0); i < ETA__ray_num_per_ray_batch; ++i) {
        ray_batch->origin_0[i] = RandomNormalNum();
        ray_batch->origin_1[i] = RandomNormalNum();
        ray_batch->origin_2[i] = RandomNormalNum();
        ray_batch->direct_0[i] = RandomNormalNum();
        ray_batch->direct_1[i] = RandomNormalNum();
        ray_batch->direct_2[i] = RandomNormalNum();
    }
}

void TestCalcTriangleRayCastBlock() {
    eta::render::Triangle tri[ETA__tri_num_per_tri_batch];

    for (unsigned int i(0); i < ETA__tri_num_per_tri_batch; ++i) {
        NormalInitTriangle(tri + i);
    }

    eta::render::TriangleBatch tri_batch;

    for (unsigned int i(0); i < ETA__tri_num_per_tri_batch; ++i) {
        eta::render::CalcTriangleBatch(i, &tri_batch, tri + i);
    }

    //

    eta::num_t r[9];

    for (unsigned int i(0); i < ETA__tri_num_per_tri_batch; ++i) {
        r[0] = tri[i].axis[0] * tri_batch.iaxis_0[i] +
               tri[i].axis[1] * tri_batch.iaxis_3[i] +
               tri[i].axis[2] * tri_batch.iaxis_6[i];
        r[1] = tri[i].axis[0] * tri_batch.iaxis_1[i] +
               tri[i].axis[1] * tri_batch.iaxis_4[i] +
               tri[i].axis[2] * tri_batch.iaxis_7[i];
        r[2] = tri[i].axis[0] * tri_batch.iaxis_2[i] +
               tri[i].axis[1] * tri_batch.iaxis_5[i] +
               tri[i].axis[2] * tri_batch.iaxis_8[i];
        r[3] = tri[i].axis[3] * tri_batch.iaxis_0[i] +
               tri[i].axis[4] * tri_batch.iaxis_3[i] +
               tri[i].axis[5] * tri_batch.iaxis_6[i];
        r[4] = tri[i].axis[3] * tri_batch.iaxis_1[i] +
               tri[i].axis[4] * tri_batch.iaxis_4[i] +
               tri[i].axis[5] * tri_batch.iaxis_7[i];
        r[5] = tri[i].axis[3] * tri_batch.iaxis_2[i] +
               tri[i].axis[4] * tri_batch.iaxis_5[i] +
               tri[i].axis[5] * tri_batch.iaxis_8[i];

        eta::num_t n_sq(tri_batch.iaxis_2[i] * tri_batch.iaxis_2[i] +
                        tri_batch.iaxis_5[i] * tri_batch.iaxis_5[i] +
                        tri_batch.iaxis_8[i] * tri_batch.iaxis_8[i]);

        // ::printf("cheking i = %d\n", i);

        if (ETA__ne_const(r[0], 1)) {
            ::printf("r[0] = " ETA__format_num "\n", r[0]);
        }

        if (ETA__ne_const(r[1], 0)) {
            ::printf("r[1] = " ETA__format_num "\n", r[1]);
        }

        if (ETA__ne_const(r[2], 0)) {
            ::printf("r[2] = " ETA__format_num "\n", r[2]);
        }

        if (ETA__ne_const(r[3], 0)) {
            ::printf("r[3] = " ETA__format_num "\n", r[3]);
        }

        if (ETA__ne_const(r[4], 1)) {
            ::printf("r[4] = " ETA__format_num "\n", r[4]);
        }

        if (ETA__ne_const(r[5], 0)) {
            ::printf("r[5] = " ETA__format_num "\n", r[5]);
        }

        if (ETA__ne_const(n_sq, 1)) {
            ::printf("n_sq = " ETA__format_num "\n", n_sq);
        }

        /*::printf(ETA__format_num ", " ETA__format_num ", " ETA__format_num "\n",
                 r[0], r[1], r[2]);*/
    }
}

#if false

void TestBatch_A_1024() {
    eta::render::Triangle tri_host[ETA__tri_num_per_tri_batch];
    for (unsigned int i(0); i < ETA__tri_num_per_tri_batch; ++i) {
        NormalInitTriangle(tri_host + i);
    }

    eta::render::Triangle* tri_device(
        eta::DeviceMalloc<eta::render::Triangle>(ETA__tri_num_per_tri_batch));
    eta::MemcpyHostToDevice(tri_device, tri_host,
                            sizeof(eta::render::Triangle) * ETA__tri_num_per_tri_batch);

    eta::render::TriangleBatch tri_batch_host;

    for (unsigned int i(0); i < ETA__tri_num_per_tri_batch; ++i) {
        eta::render::CalcTriangleBatch(i, &tri_batch_host, tri_host + i);
    }

    eta::render::TriangleBatch* tri_batch_device(eta::DeviceMalloc<eta::render::TriangleBatch>());
    eta::render::MemcpyHostToDevice(tri_batch_device, &tri_batch_host,
                            sizeof(eta::render::TriangleBatch));

    eta::render::RayBatch ray_batch_host;
    NormalInitRayBatch(&ray_batch_host);

    eta::render::RayBatch* ray_batch_device(eta::DeviceMalloc<eta::render::RayBatch>());
    eta::render::MemcpyHostToDevice(ray_batch_device, &ray_batch_host, sizeof(eta::render::RayBatch));

    eta::render::ResultBatch* result_batch(eta::DeviceMalloc<eta::render::ResultBatch>());

    eta::render::Batch_A_512<<<1, 512>>>(tri_device, result_batch, tri_batch_device,
                               ray_batch_device);

    cudaDeviceSynchronize();
}

#endif

void Test_Batch_A_speed() {
    eta::render::Triangle tri_host[ETA__tri_num_per_tri_batch];
    eta::render::TriangleBatch tri_batch_host;

    int testing_tri_batch_num(512);

    std::vector<eta::render::TriangleBatch*> tri_batch_device(
        testing_tri_batch_num);

    const eta::render::TriangleBatch** tri_batch_vec(
        eta::DeviceMalloc<const eta::render::TriangleBatch*>(
            testing_tri_batch_num));

    for (int tri_batch_i(0); tri_batch_i < testing_tri_batch_num;
         ++tri_batch_i) {
        tri_batch_device[tri_batch_i] =
            eta::DeviceMalloc<eta::render::TriangleBatch>();

        for (int tri_i(0); tri_i < ETA__tri_num_per_tri_batch; ++tri_i) {
            NormalInitTriangle(tri_host + tri_i);
            eta::render::CalcTriangleBatch(tri_i, &tri_batch_host,
                                           tri_host + tri_i);
        }

        eta::MemcpyHostToDevice(tri_batch_device[tri_batch_i], &tri_batch_host,
                                sizeof(eta::render::TriangleBatch));
    }

    eta::MemcpyHostToDevice(tri_batch_vec, tri_batch_device.data(),
                            sizeof(const eta::render::TriangleBatch*) *
                                testing_tri_batch_num);

    //

    int testing_ray_batch_num(1);

    eta::render::RayBatch* ray_batch_host(
        new eta::render::RayBatch[testing_ray_batch_num]);

    eta::render::RayBatch* ray_batch_device(
        eta::DeviceMalloc<eta::render::RayBatch>(testing_ray_batch_num));

    for (int i(0); i < testing_ray_batch_num; ++i) {
        NormalInitRayBatch(ray_batch_host + i);
    }

    eta::MemcpyHostToDevice(ray_batch_device, ray_batch_host,
                            sizeof(eta::render::RayBatch) *
                                testing_ray_batch_num);

    eta::render::ResultBatch* result_batch(
        eta::DeviceMalloc<eta::render::ResultBatch>());

    //

    cudaStream_t s;
    cudaStreamCreate(&s);

    /*eta::render::RayCastingNearest2<<<1, 1024, 0, s>>>(
        result_batch, 1, tri_batch_vec, ray_batch_device);*/

    eta::render::RayCastingNearest<<<1, 1024, 0, s>>>(
        result_batch, tri_batch_device[0], ray_batch_device);

    cudaStreamSynchronize(s);

    cudaEvent_t begin;
    cudaEventCreate(&begin);

    cudaEvent_t end;
    cudaEventCreate(&end);

    cudaEventRecord(begin, s);

    for (int tri_batch_i(0); tri_batch_i < testing_tri_batch_num;
         ++tri_batch_i) {
        eta::render::RayCastingNearest<<<1, 1024, 0, s>>>(
            result_batch, tri_batch_device[tri_batch_i], ray_batch_device);
    }

    /*eta::render::RayCastingNearest2<<<1, 1024, 0, s>>>(
        result_batch, testing_tri_batch_num, tri_batch_vec, ray_batch_device);
        */

    cudaEventRecord(end, s);

    cudaStreamSynchronize(s);

    float duration;
    cudaEventElapsedTime(&duration, begin, end);

    printf("duration: %f ms\n", duration);
}

void SharedMemConfig() {
    cudaSharedMemConfig c;
    ETA__CheckCudaError(cudaDeviceGetSharedMemConfig(&c));
    printf("%d\n", c);

    ETA__CheckCudaError(
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

    ETA__CheckCudaError(cudaDeviceGetSharedMemConfig(&c));
    printf("%d\n", c);
}

void pause() {
    printf("press any key to end");
    char end;
    int x(scanf("%c", &end));
}

int main() {
    printf("seed: %d\n", SetRandom());
    Test_Batch_A_speed();

    // pause();

    return 0;
}