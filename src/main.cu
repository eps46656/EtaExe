#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <ctime>

#include "/home/eps/desktop/Eta/Base/Triangle.cuh"
#include "/home/eps/desktop/Eta/Base/RayCast.cuh"
#include "/home/eps/desktop/Eta/Base/Calculus.cuh"
#include "/home/eps/desktop/Eta/Base/Reflection.cuh"
#include "random.h"

#///////////////////////////////////////////////////////////////////////////////
/*
void correct_calc_mat_cross(eta::num_t* dst, const eta::num_t* mat) {
	dst[0] = mat[ETA__max_dim * 0 + 1] * mat[ETA__max_dim * 1 + 2] -
			 mat[ETA__max_dim * 0 + 2] * mat[ETA__max_dim * 1 + 1];

	dst[1] = mat[ETA__max_dim * 0 + 2] * mat[ETA__max_dim * 1 + 0] -
			 mat[ETA__max_dim * 0 + 0] * mat[ETA__max_dim * 1 + 2];

	dst[2] = mat[ETA__max_dim * 0 + 0] * mat[ETA__max_dim * 1 + 1] -
			 mat[ETA__max_dim * 0 + 1] * mat[ETA__max_dim * 1 + 0];
}

void test_mat_cross() {
	eta::num_t* mat_hst_a(new eta::num_t[eta::sizeof_mat()]);
	auto mat_dev_a(deviceMalloc<eta::num_t>(eta::sizeof_mat()));

	eta::num_t* vec_hst_a(new eta::num_t[eta::sizeof_vec()]);
	eta::num_t* vec_hst_b(new eta::num_t[eta::sizeof_vec()]);

	int size(3);

	for (int i(0); i != 3; ++i) { mat_hst_a[i] = Random_d(); }

	eta::print_mat(size - 1, size, mat_hst_a);

	correct_calc_mat_cross(vec_hst_a, mat_hst_a);
	eta::mat_cross(size, vec_hst_b, mat_hst_a);

	eta::print_vec(size, vec_hst_a);
	eta::print_vec(size, vec_hst_b);

	// eta::print_vec(size, mat_hst_a);
	// eta::print_vec(size, mat_hst_a + ETA__max_dim);

	printf(ETA__format_num "\n", eta::vec_dot_vec(size, vec_hst_b, mat_hst_a));

	printf(ETA__format_num "\n",
		   eta::vec_dot_vec(size, vec_hst_b, mat_hst_a + ETA__max_dim));
}*/

#///////////////////////////////////////////////////////////////////////////////

/*void random_initialize_plane(eta::Plane* p) {
    p->origin[0] = Random_d_normal();
    p->origin[1] = Random_d_normal();
    p->origin[2] = Random_d_normal();

    p->axis[0] = Random_d_normal();
    p->axis[1] = Random_d_normal();
    p->axis[2] = Random_d_normal();

    p->axis[3] = Random_d_normal();
    p->axis[4] = Random_d_normal();
    p->axis[5] = Random_d_normal();
}*/

#///////////////////////////////////////////////////////////////////////////////

#define TRI_NUM (64)

void test_triangle() {
    num_t* tri_host(new num_t[TRI_NUM * 19]);
    num_t* tri_origin_0_host(tri_host + TRI_NUM * 0);
    num_t* tri_origin_1_host(tri_host + TRI_NUM * 1);
    num_t* tri_origin_2_host(tri_host + TRI_NUM * 2);
    num_t* tri_axis_0_host(tri_host + TRI_NUM * 3);
    num_t* tri_axis_1_host(tri_host + TRI_NUM * 4);
    num_t* tri_axis_2_host(tri_host + TRI_NUM * 5);
    num_t* tri_axis_3_host(tri_host + TRI_NUM * 6);
    num_t* tri_axis_4_host(tri_host + TRI_NUM * 7);
    num_t* tri_axis_5_host(tri_host + TRI_NUM * 8);
    num_t* tri_area_host(tri_host + TRI_NUM * 9);
    num_t* tri_iaxis_0_host(tri_host + TRI_NUM * 10);
    num_t* tri_iaxis_1_host(tri_host + TRI_NUM * 11);
    num_t* tri_iaxis_2_host(tri_host + TRI_NUM * 12);
    num_t* tri_iaxis_3_host(tri_host + TRI_NUM * 13);
    num_t* tri_iaxis_4_host(tri_host + TRI_NUM * 14);
    num_t* tri_iaxis_5_host(tri_host + TRI_NUM * 15);
    num_t* tri_iaxis_6_host(tri_host + TRI_NUM * 16);
    num_t* tri_iaxis_7_host(tri_host + TRI_NUM * 17);
    num_t* tri_iaxis_8_host(tri_host + TRI_NUM * 18);

    num_t* tri_device(eta::DeviceMalloc<num_t>(TRI_NUM * 19));
    num_t* tri_origin_0_device(tri_device + TRI_NUM * 0);
    num_t* tri_origin_1_device(tri_device + TRI_NUM * 1);
    num_t* tri_origin_2_device(tri_device + TRI_NUM * 2);
    num_t* tri_axis_0_device(tri_device + TRI_NUM * 3);
    num_t* tri_axis_1_device(tri_device + TRI_NUM * 4);
    num_t* tri_axis_2_device(tri_device + TRI_NUM * 5);
    num_t* tri_axis_3_device(tri_device + TRI_NUM * 6);
    num_t* tri_axis_4_device(tri_device + TRI_NUM * 7);
    num_t* tri_axis_5_device(tri_device + TRI_NUM * 8);
    num_t* tri_area_device(tri_device + TRI_NUM * 9);
    num_t* tri_iaxis_0_device(tri_device + TRI_NUM * 10);
    num_t* tri_iaxis_1_device(tri_device + TRI_NUM * 11);
    num_t* tri_iaxis_2_device(tri_device + TRI_NUM * 12);
    num_t* tri_iaxis_3_device(tri_device + TRI_NUM * 13);
    num_t* tri_iaxis_4_device(tri_device + TRI_NUM * 14);
    num_t* tri_iaxis_5_device(tri_device + TRI_NUM * 15);
    num_t* tri_iaxis_6_device(tri_device + TRI_NUM * 16);
    num_t* tri_iaxis_7_device(tri_device + TRI_NUM * 17);
    num_t* tri_iaxis_8_device(tri_device + TRI_NUM * 18);

    for (int i(0); i != TRI_NUM * 9; ++i) { tri_host[i] = Random_d_normal(); }

    eta::MemcpyHostToDevice(tri_device, tri_host, sizeof(num_t) * TRI_NUM * 9);

    eta::batch_get_extd_attr_512<<<1, TRI_NUM>>>(
        tri_area_device,

        tri_iaxis_0_device, tri_iaxis_1_device, tri_iaxis_2_device,
        tri_iaxis_3_device, tri_iaxis_4_device, tri_iaxis_5_device,
        tri_iaxis_6_device, tri_iaxis_7_device, tri_iaxis_8_device,

        tri_axis_0_device, tri_axis_1_device, tri_axis_2_device,
        tri_axis_3_device, tri_axis_4_device, tri_axis_5_device);

    eta::MemcpyDeviceToHost(tri_area_host, tri_area_device,
                            sizeof(num_t) * TRI_NUM * 10);

    for (int tri_i(0); tri_i != TRI_NUM; ++tri_i) {
        num_t axis_0(tri_axis_0_host[tri_i]);
        num_t axis_1(tri_axis_1_host[tri_i]);
        num_t axis_2(tri_axis_2_host[tri_i]);
        num_t axis_3(tri_axis_3_host[tri_i]);
        num_t axis_4(tri_axis_4_host[tri_i]);
        num_t axis_5(tri_axis_5_host[tri_i]);
        num_t iaxis_0(tri_iaxis_0_host[tri_i]);
        num_t iaxis_1(tri_iaxis_1_host[tri_i]);
        num_t iaxis_2(tri_iaxis_2_host[tri_i]);
        num_t iaxis_3(tri_iaxis_3_host[tri_i]);
        num_t iaxis_4(tri_iaxis_4_host[tri_i]);
        num_t iaxis_5(tri_iaxis_5_host[tri_i]);
        num_t iaxis_6(tri_iaxis_6_host[tri_i]);
        num_t iaxis_7(tri_iaxis_7_host[tri_i]);
        num_t iaxis_8(tri_iaxis_8_host[tri_i]);

        num_t k[] = {
            axis_0 * iaxis_0 + axis_1 * iaxis_3 + axis_2 * iaxis_6,
            axis_0 * iaxis_1 + axis_1 * iaxis_4 + axis_2 * iaxis_7,
            axis_0 * iaxis_2 + axis_1 * iaxis_5 + axis_2 * iaxis_8,
            axis_3 * iaxis_0 + axis_4 * iaxis_3 + axis_5 * iaxis_6,
            axis_3 * iaxis_1 + axis_4 * iaxis_4 + axis_5 * iaxis_7,
            axis_3 * iaxis_2 + axis_4 * iaxis_5 + axis_5 * iaxis_8,
        };

        num_t sqm(iaxis_2 * iaxis_2 + iaxis_5 * iaxis_5 + iaxis_8 * iaxis_8);

        if (ETA__ne_const(k[0], 1) || ETA__ne_const(k[1], 0) ||
            ETA__ne_const(k[2], 0) || ETA__ne_const(k[3], 0) ||
            ETA__ne_const(k[4], 1) || ETA__ne_const(k[5], 0) ||
            ETA__ne_const(sqm, 1)) {
            printf("error\n");
        }

        printf(ETA__format_num "\n", k[0]);
    }

    // eta::num_t t;
    // eta::num_t s0;
    // eta::num_t s1;
    // eta::num_t point[3];

    // eta::ray_cast(&t, &s0, &s1, point, &tr, &ray);
}

#///////////////////////////////////////////////////////////////////////////////

void test_Random_d_zero_to_one() {
    for (int i(0); i != 8; ++i) {
        printf(ETA__format_num "\n", Random_d_zero_to_one());
    }
}

#///////////////////////////////////////////////////////////////////////////////

#if false

void test_batch_ray_cast() {
    #define TR_NUM (64)
    #define RAY_NUM (992)

    eta::Plane* p(new eta::Triangle[TR_NUM]);
    for (int i(0); i != TR_NUM; ++i) {
        random_initialize_plane(p + i);
        eta::set_plane(p + i);
    }

    eta::Ray* ray(new eta::Ray[RAY_NUM]);
    for (int i(0); i != RAY_NUM; ++i) { random_initialize_ray(ray + i); }

    // declare

    eta::num_t* dst_t_host(new eta::num_t[TR_NUM * RAY_NUM]);
    eta::num_t* dst_s_0_host(new eta::num_t[TR_NUM * RAY_NUM]);
    eta::num_t* dst_s_1_host(new eta::num_t[TR_NUM * RAY_NUM]);

    eta::num_t* dst_t_device(deviceMalloc<eta::num_t>(TR_NUM * RAY_NUM));
    eta::num_t* dst_s_0_device(deviceMalloc<eta::num_t>(TR_NUM * RAY_NUM));
    eta::num_t* dst_s_1_device(deviceMalloc<eta::num_t>(TR_NUM * RAY_NUM));

    eta::num_t* tr_origin_0_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_origin_1_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_origin_2_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_axis_6_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_axis_7_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_axis_8_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_inv_0_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_inv_1_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_inv_3_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_inv_4_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_inv_6_host(new eta::num_t[TR_NUM]);
    eta::num_t* tr_inv_7_host(new eta::num_t[TR_NUM]);

    eta::num_t* tr_origin_0_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_origin_1_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_origin_2_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_axis_6_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_axis_7_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_axis_8_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_inv_0_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_inv_1_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_inv_3_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_inv_4_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_inv_6_device(deviceMalloc<eta::num_t>(TR_NUM));
    eta::num_t* tr_inv_7_device(deviceMalloc<eta::num_t>(TR_NUM));

    eta::num_t* ray_origin_0_host(new eta::num_t[RAY_NUM]);
    eta::num_t* ray_origin_1_host(new eta::num_t[RAY_NUM]);
    eta::num_t* ray_origin_2_host(new eta::num_t[RAY_NUM]);
    eta::num_t* ray_direct_0_host(new eta::num_t[RAY_NUM]);
    eta::num_t* ray_direct_1_host(new eta::num_t[RAY_NUM]);
    eta::num_t* ray_direct_2_host(new eta::num_t[RAY_NUM]);

    eta::num_t* ray_origin_0_device(deviceMalloc<eta::num_t>(RAY_NUM));
    eta::num_t* ray_origin_1_device(deviceMalloc<eta::num_t>(RAY_NUM));
    eta::num_t* ray_origin_2_device(deviceMalloc<eta::num_t>(RAY_NUM));
    eta::num_t* ray_direct_0_device(deviceMalloc<eta::num_t>(RAY_NUM));
    eta::num_t* ray_direct_1_device(deviceMalloc<eta::num_t>(RAY_NUM));
    eta::num_t* ray_direct_2_device(deviceMalloc<eta::num_t>(RAY_NUM));

    // tr data

    for (int i(0); i != TR_NUM; ++i) {
        tr_origin_0_host[i] = tr[i].vertex[0];
        tr_origin_1_host[i] = tr[i].vertex[1];
        tr_origin_2_host[i] = tr[i].vertex[2];
        tr_axis_6_host[i] = tr[i].axis[6];
        tr_axis_7_host[i] = tr[i].axis[7];
        tr_axis_8_host[i] = tr[i].axis[8];
        tr_inv_0_host[i] = tr[i].inv[0];
        tr_inv_1_host[i] = tr[i].inv[1];
        tr_inv_3_host[i] = tr[i].inv[3];
        tr_inv_4_host[i] = tr[i].inv[4];
        tr_inv_6_host[i] = tr[i].inv[6];
        tr_inv_7_host[i] = tr[i].inv[7];
    }

    cudaMemcpy(tr_origin_0_device, tr_origin_0_host,
               sizeof(eta::num_t) * TR_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(tr_origin_1_device, tr_origin_1_host,
               sizeof(eta::num_t) * TR_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(tr_origin_2_device, tr_origin_2_host,
               sizeof(eta::num_t) * TR_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(tr_axis_6_device, tr_axis_6_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_axis_7_device, tr_axis_7_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_axis_8_device, tr_axis_8_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_inv_0_device, tr_inv_0_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_inv_1_device, tr_inv_1_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_inv_3_device, tr_inv_3_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_inv_4_device, tr_inv_4_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_inv_6_device, tr_inv_6_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(tr_inv_7_device, tr_inv_7_host, sizeof(eta::num_t) * TR_NUM,
               cudaMemcpyHostToDevice);

    // ray data

    for (int i(0); i != RAY_NUM; ++i) {
        ray_origin_0_host[i] = ray[i].origin[0];
        ray_origin_1_host[i] = ray[i].origin[1];
        ray_origin_2_host[i] = ray[i].origin[2];
        ray_direct_0_host[i] = ray[i].direct[0];
        ray_direct_1_host[i] = ray[i].direct[1];
        ray_direct_2_host[i] = ray[i].direct[2];
    }

    cudaMemcpy(ray_origin_0_device, ray_origin_0_host,
               sizeof(eta::num_t) * RAY_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(ray_origin_1_device, ray_origin_1_host,
               sizeof(eta::num_t) * RAY_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(ray_origin_2_device, ray_origin_2_host,
               sizeof(eta::num_t) * RAY_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(ray_direct_0_device, ray_direct_0_host,
               sizeof(eta::num_t) * RAY_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(ray_direct_1_device, ray_direct_1_host,
               sizeof(eta::num_t) * RAY_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(ray_direct_2_device, ray_direct_2_host,
               sizeof(eta::num_t) * RAY_NUM, cudaMemcpyHostToDevice);

    // ray cast

    eta::batch_ray_cast<<<TR_NUM, RAY_NUM>>>(
        dst_t_device, dst_s_0_device, dst_s_1_device, tr_origin_0_device,
        tr_origin_1_device, tr_origin_2_device, tr_axis_6_device,
        tr_axis_7_device, tr_axis_8_device, tr_inv_0_device, tr_inv_1_device,
        tr_inv_3_device, tr_inv_4_device, tr_inv_6_device, tr_inv_7_device,
        ray_origin_0_device, ray_origin_1_device, ray_origin_2_device,
        ray_direct_0_device, ray_direct_1_device, ray_direct_2_device);

    // copy data from device to host

    cudaMemcpy(dst_t_host, dst_t_device, sizeof(eta::num_t) * TR_NUM * RAY_NUM,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_s_0_host, dst_s_0_device,
               sizeof(eta::num_t) * TR_NUM * RAY_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_s_1_host, dst_s_1_device,
               sizeof(eta::num_t) * TR_NUM * RAY_NUM, cudaMemcpyDeviceToHost);

    // evaluate

    for (int tr_i(0); tr_i != TR_NUM; ++tr_i) {
        for (int ray_i(0); ray_i != RAY_NUM; ++ray_i) {
            int i(tr_i * RAY_NUM + ray_i);
            eta::num_t t(dst_t_host[i]);
            eta::num_t s_0(dst_s_0_host[i]);
            eta::num_t s_1(dst_s_1_host[i]);

            eta::num_t k[] = {
                ray[ray_i].origin[0] + ray[ray_i].direct[0] * t,
                ray[ray_i].origin[1] + ray[ray_i].direct[1] * t,
                ray[ray_i].origin[2] + ray[ray_i].direct[2] * t,
            };

            eta::num_t l[] = {
                tr[tr_i].vertex[0] + tr[tr_i].axis[0] * s_0 +
                    tr[tr_i].axis[3] * s_1,
                tr[tr_i].vertex[1] + tr[tr_i].axis[1] * s_0 +
                    tr[tr_i].axis[4] * s_1,
                tr[tr_i].vertex[2] + tr[tr_i].axis[2] * s_0 +
                    tr[tr_i].axis[5] * s_1,
            };

            eta::num_t k_sub_l[] = {
                k[0] - l[0],
                k[1] - l[1],
                k[2] - l[2],
            };

            eta::num_t abs_k_sub_l(eta::abs_vec_3(k_sub_l));

            printf("[%4d, %4d] " ETA__format_num "\n", tr_i, ray_i,
                   abs_k_sub_l);

            /*printf("[%4d, %4d] ", tr_i, ray_i);*/

            /*printf("[ " ETA__format_num ", " ETA__format_num
				   ", " ETA__format_num " ]",
				   point[0], point[1], point[2]);*/

            /*printf(ETA__format_num ", " ETA__format_num " ", abs_point_sub_k,
				   abs_point_sub_l);*/

            /*printf("%d\n", eta::num_cmp_eq_zero(abs_point_sub_k) &&
							   eta::num_cmp_eq_zero(abs_point_sub_l));*/

            /*if (!(eta::num_cmp_eq_zero(abs_point_sub_k) &&
				  eta::num_cmp_eq_zero(abs_point_sub_l))) {
				printf("[%4d, %4d] failure\n", tr_i, ray_i);
			}*/
        }
    }

    // free
}

#endif

#if false

void test_batch_ray_cast() {
    #define TR_NUM (8)
    #define RAY_NUM (8)

    eta::Triangle* tr(new eta::Triangle[TR_NUM]);
    for (int i(0); i != TR_NUM; ++i) {
        random_initialize_triangle(tr + i);
        eta::set_triangle(tr + i);
    }

    eta::Ray* ray(new eta::Ray[RAY_NUM]);
    for (int i(0); i != RAY_NUM; ++i) { random_initialize_ray(ray + i); }

    eta::num_t* dst_t(new eta::num_t[TR_NUM * RAY_NUM]);
    eta::num_t* dst_s_0(new eta::num_t[TR_NUM * RAY_NUM]);
    eta::num_t* dst_s_1(new eta::num_t[TR_NUM * RAY_NUM]);

    eta::batch_ray_cast(TR_NUM, RAY_NUM, dst_t, dst_s_0, dst_s_1, tr, ray);

    for (int tr_i(0); tr_i != TR_NUM; ++tr_i) {
        for (int ray_i(0); ray_i != RAY_NUM; ++ray_i) {
            int i(tr_i * RAY_NUM + ray_i);
            eta::num_t t(dst_t[i]);
            eta::num_t s_0(dst_s_0[i]);
            eta::num_t s_1(dst_s_1[i]);

            eta::num_t k[] = {
                ray[ray_i].origin[0] + ray[ray_i].direct[0] * t,
                ray[ray_i].origin[1] + ray[ray_i].direct[1] * t,
                ray[ray_i].origin[2] + ray[ray_i].direct[2] * t,
            };

            eta::num_t l[] = {
                tr[tr_i].vertex[0] + tr[tr_i].axis[0] * s_0 +
                    tr[tr_i].axis[3] * s_1,
                tr[tr_i].vertex[1] + tr[tr_i].axis[1] * s_0 +
                    tr[tr_i].axis[4] * s_1,
                tr[tr_i].vertex[2] + tr[tr_i].axis[2] * s_0 +
                    tr[tr_i].axis[5] * s_1,
            };

            eta::num_t k_sub_l[] = {
                k[0] - l[0],
                k[1] - l[1],
                k[2] - l[2],
            };

            eta::num_t abs_k_sub_l(eta::abs_vec_3(k_sub_l));

            printf("[%4d, %4d] " ETA__format_num "\n", tr_i, ray_i,
                   abs_k_sub_l);
        }
    }
}

#endif

#define P_SIZE (64)
#define R_SIZE (992)

#if false

void test_batch_ray_cast() {
    num_t* mem_host(new num_t[P_SIZE * R_SIZE * 3 + // dst
                              P_SIZE * 9 + // for plane
                              R_SIZE * 6 // for ray
    ]);

    num_t* dst_host(mem_host);

    num_t* t_dst_host(dst_host + P_SIZE * R_SIZE * 0);
    num_t* s_0_dst_host(dst_host + P_SIZE * R_SIZE * 1);
    num_t* s_1_dst_host(dst_host + P_SIZE * R_SIZE * 2);

    num_t* p_host(mem_host + P_SIZE * R_SIZE * 3);

    num_t* p_origin_0_host(p_host + P_SIZE * 0);
    num_t* p_origin_1_host(p_host + P_SIZE * 1);
    num_t* p_origin_2_host(p_host + P_SIZE * 2);

    num_t* p_axis_0_host(p_host + P_SIZE * 3);
    num_t* p_axis_1_host(p_host + P_SIZE * 4);
    num_t* p_axis_2_host(p_host + P_SIZE * 5);
    num_t* p_axis_3_host(p_host + P_SIZE * 6);
    num_t* p_axis_4_host(p_host + P_SIZE * 7);
    num_t* p_axis_5_host(p_host + P_SIZE * 8);

    num_t* r_host(mem_host + R_SIZE * 9);

    num_t* r_origin_0_host(r_host + R_SIZE * 0);
    num_t* r_origin_1_host(r_host + R_SIZE * 1);
    num_t* r_origin_2_host(r_host + R_SIZE * 2);
    num_t* r_direct_0_host(r_host + R_SIZE * 3);
    num_t* r_direct_1_host(r_host + R_SIZE * 4);
    num_t* r_direct_2_host(r_host + R_SIZE * 5);

    //

    num_t* mem_device;
    cudaError_t err(cudaMalloc(&mem_device,
                               sizeof(num_t) * (P_SIZE * R_SIZE * 3 + // for dst
                                                P_SIZE * 19 + // for plane
                                                R_SIZE * 6 // for ray
                                                )));

    printf("%s\n", cudaGetErrorString(err));

    num_t* dst_device(mem_device);

    num_t* t_dst_device(dst_device + P_SIZE * R_SIZE * 0);
    num_t* s_0_dst_device(dst_device + P_SIZE * R_SIZE * 1);
    num_t* s_1_dst_device(dst_device + P_SIZE * R_SIZE * 2);

    num_t* p_device(mem_device + P_SIZE * R_SIZE * 3);

    num_t* p_origin_0_device(p_device + P_SIZE * 0);
    num_t* p_origin_1_device(p_device + P_SIZE * 1);
    num_t* p_origin_2_device(p_device + P_SIZE * 2);

    num_t* p_axis_0_device(p_device + P_SIZE * 3);
    num_t* p_axis_1_device(p_device + P_SIZE * 4);
    num_t* p_axis_2_device(p_device + P_SIZE * 5);
    num_t* p_axis_3_device(p_device + P_SIZE * 6);
    num_t* p_axis_4_device(p_device + P_SIZE * 7);
    num_t* p_axis_5_device(p_device + P_SIZE * 8);

    num_t* p_det_device(p_device + P_SIZE * 9);

    num_t* p_iaxis_0_device(p_device + P_SIZE * 10);
    num_t* p_iaxis_1_device(p_device + P_SIZE * 11);
    num_t* p_iaxis_2_device(p_device + P_SIZE * 12);
    num_t* p_iaxis_3_device(p_device + P_SIZE * 13);
    num_t* p_iaxis_4_device(p_device + P_SIZE * 14);
    num_t* p_iaxis_5_device(p_device + P_SIZE * 15);
    num_t* p_iaxis_6_device(p_device + P_SIZE * 16);
    num_t* p_iaxis_7_device(p_device + P_SIZE * 17);
    num_t* p_iaxis_8_device(p_device + P_SIZE * 18);

    num_t* r_device(mem_device + P_SIZE * R_SIZE * 3 + P_SIZE * 19);

    num_t* r_origin_0_device(r_device + R_SIZE * 0);
    num_t* r_origin_1_device(r_device + R_SIZE * 1);
    num_t* r_origin_2_device(r_device + R_SIZE * 2);
    num_t* r_direct_0_device(r_device + R_SIZE * 3);
    num_t* r_direct_1_device(r_device + R_SIZE * 4);
    num_t* r_direct_2_device(r_device + R_SIZE * 5);

    //

    for (int i(0); i != P_SIZE * 9 + R_SIZE * 6; ++i) {
        mem_host[P_SIZE * R_SIZE * 3 + i] = Random_d_normal();
    }

    //

    cudaMemcpy(p_device, p_host, sizeof(num_t) * P_SIZE * 9,
               cudaMemcpyHostToDevice);

    cudaMemcpy(r_device, r_host, sizeof(num_t) * R_SIZE * 6,
               cudaMemcpyHostToDevice);

    //

    eta::batch_get_iaxis<<<1, P_SIZE>>>(
        p_det_device, p_iaxis_0_device, p_iaxis_1_device, p_iaxis_2_device,
        p_iaxis_3_device, p_iaxis_4_device, p_iaxis_5_device, p_iaxis_6_device,
        p_iaxis_7_device, p_iaxis_8_device, p_axis_0_device, p_axis_1_device,
        p_axis_2_device, p_axis_3_device, p_axis_4_device, p_axis_5_device);

    //

    eta::batch_ray_cast<<<P_SIZE, R_SIZE>>>(
        t_dst_device, s_0_dst_device, s_1_dst_device, p_origin_0_device,
        p_origin_1_device, p_origin_2_device, p_iaxis_0_device,
        p_iaxis_1_device, p_iaxis_2_device, p_iaxis_3_device, p_iaxis_4_device,
        p_iaxis_5_device, p_iaxis_6_device, p_iaxis_7_device, p_iaxis_8_device,
        r_origin_0_device, r_origin_1_device, r_origin_2_device,
        r_direct_0_device, r_direct_1_device, r_direct_2_device);

    //

    cudaMemcpy(dst_host, dst_device, sizeof(num_t) * P_SIZE * R_SIZE * 3,
               cudaMemcpyDeviceToHost);

    //

    for (int p_i(0); p_i != P_SIZE; ++p_i) {
        for (int r_i(0); r_i != R_SIZE; ++r_i) {
            num_t t(t_dst_host[p_i * R_SIZE + r_i]);
            num_t s_0(s_0_dst_host[p_i * R_SIZE + r_i]);
            num_t s_1(s_1_dst_host[p_i * R_SIZE + r_i]);

            printf("  t: " ETA__format_num "\n", t);
            printf("s_0: " ETA__format_num "\n", s_0);
            printf("s_1: " ETA__format_num "\n", s_1);

            num_t k[] = {
                r_origin_0_host[r_i] + r_direct_0_host[r_i] * t,
                r_origin_1_host[r_i] + r_direct_1_host[r_i] * t,
                r_origin_2_host[r_i] + r_direct_2_host[r_i] * t,
            };

            num_t l[] = {
                p_origin_0_host[p_i] + p_axis_0_host[p_i] * s_0 +
                    p_axis_3_host[p_i] * s_1,
                p_origin_1_host[p_i] + p_axis_1_host[p_i] * s_0 +
                    p_axis_4_host[p_i] * s_1,
                p_origin_2_host[p_i] + p_axis_2_host[p_i] * s_0 +
                    p_axis_5_host[p_i] * s_1,
            };

            num_t k_sub_l[] = {
                k[0] - l[0],
                k[1] - l[1],
                k[2] - l[2],
            };

            num_t abs_k_sub_l(eta::abs_vec_3(k_sub_l));

            // printf("%2d %4d " ETA__format_num "\n", p_i, r_i, abs_k_sub_l);
        }
    }
}

#endif

#if false

void test_batch_ray_cast() {
    num_t* dst_host(new num_t[P_SIZE * R_SIZE * 3]);
    num_t* t_dst_host(dst_host + P_SIZE * R_SIZE * 0);
    num_t* s_0_dst_host(dst_host + P_SIZE * R_SIZE * 1);
    num_t* s_1_dst_host(dst_host + P_SIZE * R_SIZE * 2);

    num_t* p_host(new num_t[P_SIZE * 3]);
    num_t* p_origin_0_host(p_host + P_SIZE * 0);
    num_t* p_origin_1_host(p_host + P_SIZE * 1);
    num_t* p_origin_2_host(p_host + P_SIZE * 2);
    num_t* p_axis_0_host(p_host + P_SIZE * 3);
    num_t* p_axis_1_host(p_host + P_SIZE * 4);
    num_t* p_axis_2_host(p_host + P_SIZE * 5);
    num_t* p_axis_3_host(p_host + P_SIZE * 6);
    num_t* p_axis_4_host(p_host + P_SIZE * 7);
    num_t* p_axis_5_host(p_host + P_SIZE * 8);

    num_t* r_host(new num_t[R_SIZE * 9]);
    num_t* r_origin_0_host(r_host + R_SIZE * 0);
    num_t* r_origin_1_host(r_host + R_SIZE * 1);
    num_t* r_origin_2_host(r_host + R_SIZE * 2);
    num_t* r_direct_0_host(r_host + R_SIZE * 3);
    num_t* r_direct_1_host(r_host + R_SIZE * 4);
    num_t* r_direct_2_host(r_host + R_SIZE * 5);

    //

    num_t* dst_device(eta::DeviceMalloc<num_t>(P_SIZE * R_SIZE * 3));
    num_t* t_dst_device(dst_device + P_SIZE * R_SIZE * 0);
    num_t* s_0_dst_device(dst_device + P_SIZE * R_SIZE * 1);
    num_t* s_1_dst_device(dst_device + P_SIZE * R_SIZE * 2);

    num_t* p_device(eta::DeviceMalloc<num_t>(P_SIZE * 19));
    num_t* p_origin_0_device(p_device + P_SIZE * 0);
    num_t* p_origin_1_device(p_device + P_SIZE * 1);
    num_t* p_origin_2_device(p_device + P_SIZE * 2);
    num_t* p_axis_0_device(p_device + P_SIZE * 3);
    num_t* p_axis_1_device(p_device + P_SIZE * 4);
    num_t* p_axis_2_device(p_device + P_SIZE * 5);
    num_t* p_axis_3_device(p_device + P_SIZE * 6);
    num_t* p_axis_4_device(p_device + P_SIZE * 7);
    num_t* p_axis_5_device(p_device + P_SIZE * 8);
    num_t* p_det_device(p_device + P_SIZE * 9);
    num_t* p_iaxis_0_device(p_device + P_SIZE * 10);
    num_t* p_iaxis_1_device(p_device + P_SIZE * 11);
    num_t* p_iaxis_2_device(p_device + P_SIZE * 12);
    num_t* p_iaxis_3_device(p_device + P_SIZE * 13);
    num_t* p_iaxis_4_device(p_device + P_SIZE * 14);
    num_t* p_iaxis_5_device(p_device + P_SIZE * 15);
    num_t* p_iaxis_6_device(p_device + P_SIZE * 16);
    num_t* p_iaxis_7_device(p_device + P_SIZE * 17);
    num_t* p_iaxis_8_device(p_device + P_SIZE * 18);

    num_t* r_device(eta::DeviceMalloc<num_t>(R_SIZE * 6));
    num_t* r_origin_0_device(r_device + R_SIZE * 0);
    num_t* r_origin_1_device(r_device + R_SIZE * 1);
    num_t* r_origin_2_device(r_device + R_SIZE * 2);
    num_t* r_direct_0_device(r_device + R_SIZE * 3);
    num_t* r_direct_1_device(r_device + R_SIZE * 4);
    num_t* r_direct_2_device(r_device + R_SIZE * 5);

    //

    for (int i(0); i != P_SIZE * 9; ++i) { p_host[i] = Random_d_normal(); }
    for (int i(0); i != R_SIZE * 6; ++i) { r_host[i] = Random_d_normal(); }

    //

    cudaMemcpy(p_device, p_host, sizeof(num_t) * P_SIZE * 9,
               cudaMemcpyHostToDevice);

    cudaMemcpy(r_device, r_host, sizeof(num_t) * R_SIZE * 6,
               cudaMemcpyHostToDevice);

    //

    eta::batch_get_iaxis<<<1, P_SIZE>>>(
        p_det_device, p_iaxis_0_device, p_iaxis_1_device, p_iaxis_2_device,
        p_iaxis_3_device, p_iaxis_4_device, p_iaxis_5_device, p_iaxis_6_device,
        p_iaxis_7_device, p_iaxis_8_device, p_axis_0_device, p_axis_1_device,
        p_axis_2_device, p_axis_3_device, p_axis_4_device, p_axis_5_device);

    //

    eta::batch_ray_cast<<<P_SIZE, R_SIZE>>>(
        t_dst_device, s_0_dst_device, s_1_dst_device, p_origin_0_device,
        p_origin_1_device, p_origin_2_device, p_iaxis_0_device,
        p_iaxis_1_device, p_iaxis_2_device, p_iaxis_3_device, p_iaxis_4_device,
        p_iaxis_5_device, p_iaxis_6_device, p_iaxis_7_device, p_iaxis_8_device,
        r_origin_0_device, r_origin_1_device, r_origin_2_device,
        r_direct_0_device, r_direct_1_device, r_direct_2_device);

    //

    cudaMemcpy(dst_host, dst_device, sizeof(num_t) * P_SIZE * R_SIZE * 3,
               cudaMemcpyDeviceToHost);

    //

    int count(0);

    for (int p_i(0); p_i != P_SIZE; ++p_i) {
        for (int r_i(0); r_i != R_SIZE; ++r_i) {
            num_t t(t_dst_host[p_i * R_SIZE + r_i]);
            num_t s_0(s_0_dst_host[p_i * R_SIZE + r_i]);
            num_t s_1(s_1_dst_host[p_i * R_SIZE + r_i]);

            // printf("  t: " ETA__format_num "\n", t);
            // printf("s_0: " ETA__format_num "\n", s_0);
            // printf("s_1: " ETA__format_num "\n", s_1);

            double k[] = {
                r_origin_0_host[r_i] + r_direct_0_host[r_i] * t,
                r_origin_1_host[r_i] + r_direct_1_host[r_i] * t,
                r_origin_2_host[r_i] + r_direct_2_host[r_i] * t,
            };

            double l[] = {
                p_origin_0_host[p_i] + p_axis_0_host[p_i] * s_0 +
                    p_axis_3_host[p_i] * s_1,
                p_origin_1_host[p_i] + p_axis_1_host[p_i] * s_0 +
                    p_axis_4_host[p_i] * s_1,
                p_origin_2_host[p_i] + p_axis_2_host[p_i] * s_0 +
                    p_axis_5_host[p_i] * s_1,
            };

            double k_l_avg[] = { (k[0] + l[0]) / 2, (k[1] + l[1]) / 2,
                                 (k[2] + l[2]) / 2 };

            double k_sub_l[] = { k[0] - l[0], k[1] - l[1], k[2] - l[2] };

            double abs_k_sub_l(sqrt(k_sub_l[0] * k_sub_l[0] +
                                    k_sub_l[1] * k_sub_l[1] +
                                    k_sub_l[2] * k_sub_l[2]));
            double abs_k_l_avg(sqrt(k_l_avg[0] * k_l_avg[0] +
                                    k_l_avg[1] * k_l_avg[1] +
                                    k_l_avg[2] * k_l_avg[2]));

            double err(abs_k_sub_l / abs_k_l_avg);

            //if (!eta::num_eq_zero(err)) {
            // printf("%2d %4d " ETA__format_num "\n", p_i, r_i, err);
            if (ETA__ne_const(abs_k_sub_l, 0)) {
                printf("%2d %4d " ETA__format_num "\n", p_i, r_i, abs_k_sub_l);
                ++count;
            }
            //}
        }
    }

    printf("%d / %d\n", count, P_SIZE * R_SIZE);
}

#endif

#///////////////////////////////////////////////////////////////////////////////

int main() {
    // eta::initialize_determine_table();

    // g<<<1, 5>>>();
    // cudaDeviceSynchronize();

    long int seed(SetRandom());
    printf("seed: %lu\n", seed);

    test_triangle();

    // test_batch_ray_cast();

    // test_Random_d_zero_to_one();

    // test_mat_cross();

    /*int max_det_i = 2;

	int row_i = 3;
	int det_i = 5;

	std::cout << ((row_i ^ det_i) & 1) << "\n";

	std::cout << (-2 * ((row_i ^ det_i) & 1) + 1) << "\n";

	std::cout << ((-2 * ((row_i ^ det_i) & 1) + 1)) << "\n";*/

    // std::cout << ((~((row_i ^ det_i) & 1)) | 1) << "\n";

    // std::cout << ((((row_i ^ det_i) & 1) << 1) - 1) << "\n";

    // cudaDeviceProp cdp;

    // cudaGetDeviceProperties(&cdp, 0);

    // cudaSharedMemConfig csc;
    // cudaDeviceGetSharedMemConfig(&csc);

    // std::cout << csc << "\n";

    // std::cout <<

    // std::cout << cdp.regsPerBlock << "\n";
    /*std::cout << cdp.maxThreadsDim[0] << ", " << cdp.maxThreadsDim[1] << ", "
			  << cdp.maxThreadsDim[2] << "\n";*/

    return 0;
}