/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Mael Keryell
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */
#include <stdio.h>
#include <stdint.h>
#include <starpu.h>

static inline long long jlstarpu_max(long long a, long long b)
{
	return (a > b) ? a : b;
}

static inline long long jlstarpu_interval_size(long long start, long long step, long long stop)
{
    if (stop >= start){
            return jlstarpu_max(0, (stop - start + 1) / step);
    } else {
            return jlstarpu_max(0, (stop - start - 1) / step);
    }
}

void nbody_acc(void** buffers_NJVQ1U4V, void* cl_arg_NJVQ1U4V)
{
    uint32_t ld_xIZ5HaKV = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_NJVQ1U4V[(1) - (1)]));
    double* ptr_xIZ5HaKV = (double*) (STARPU_MATRIX_GET_PTR(buffers_NJVQ1U4V[(1) - (1)]));
    uint32_t ld_QZvmSRYk = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_NJVQ1U4V[(2) - (1)]));
    double* ptr_QZvmSRYk = (double*) (STARPU_MATRIX_GET_PTR(buffers_NJVQ1U4V[(2) - (1)]));
    double* ptr_U7pwlAjr = (double*) (STARPU_VECTOR_GET_PTR(buffers_NJVQ1U4V[(3) - (1)]));
    double* ptr_AQKcQq1a = (double*) (STARPU_VECTOR_GET_PTR(buffers_NJVQ1U4V[(4) - (1)]));
    int64_t* ptr_qQoIJuDP = (int64_t*) (STARPU_VECTOR_GET_PTR(buffers_NJVQ1U4V[(5) - (1)]));

    int64_t widthp = (int64_t) (STARPU_MATRIX_GET_NY(buffers_NJVQ1U4V[(1) - (1)]));
    int64_t widtha = (int64_t) (STARPU_MATRIX_GET_NY(buffers_NJVQ1U4V[(2) - (1)]));
    
    int64_t start_nj3HDzsW = (int64_t) (1);
    int64_t stop_nj3HDzsW = (int64_t) (widtha);
    int64_t plan;

    for (plan = start_nj3HDzsW ; plan <= stop_nj3HDzsW ; plan += 1)
    {
        double sumaccx = (double) (0);
        double sumaccy = (double) (0);
        
        int64_t start_TzfU6QY7 = (int64_t) (1);
        int64_t stop_TzfU6QY7 = (int64_t) (widthp);
        int64_t oplan;

        for (oplan = start_TzfU6QY7 ; oplan <= stop_TzfU6QY7 ; oplan += 1)
        {
            double eps = (double) (ptr_AQKcQq1a[(3) - (1)]);
            int64_t Id = (int64_t) ((ptr_qQoIJuDP[(1) - (1)]) * (widtha));
            double G = (double) (ptr_AQKcQq1a[(1) - (1)]);
            int64_t b = (int64_t) ((((plan) + (Id)) >= (oplan)) + (((plan) + (Id)) <= (oplan)));
            
            if ((b) < (2))
            {
                double dx = (double) ((ptr_xIZ5HaKV[((1) + (((oplan) - (1)) * (ld_xIZ5HaKV))) - (1)]) - (ptr_xIZ5HaKV[((1) + ((((plan) + (Id)) - (1)) * (ld_xIZ5HaKV))) - (1)]));
                double dy = (double) ((ptr_xIZ5HaKV[((2) + (((oplan) - (1)) * (ld_xIZ5HaKV))) - (1)]) - (ptr_xIZ5HaKV[((2) + ((((plan) + (Id)) - (1)) * (ld_xIZ5HaKV))) - (1)]));
                double modul = (double) (sqrt(((dx) * (dx)) + ((dy) * (dy))));
                sumaccx = (sumaccx) + (((G) * (ptr_U7pwlAjr[(oplan) - (1)]) * (dx)) / (((modul) + (eps)) * ((modul) + (eps)) * ((modul) + (eps))));
                sumaccy = (sumaccy) + (((G) * (ptr_U7pwlAjr[(oplan) - (1)]) * (dy)) / (((modul) + (eps)) * ((modul) + (eps)) * ((modul) + (eps))));
            };
	    
        }
        ;

	
        ptr_QZvmSRYk[((1) + (((plan) - (1)) * (ld_QZvmSRYk))) - (1)] = sumaccx;
        ptr_QZvmSRYk[((2) + (((plan) - (1)) * (ld_QZvmSRYk))) - (1)] = sumaccy;

        /* ptr_QZvmSRYk[((1) + (((plan) - (1)) * (ld_QZvmSRYk))) - (1)] = ptr_qQoIJuDP[(1) - (1)]; */

        /* ptr_QZvmSRYk[((2) + (((plan) - (1)) * (ld_QZvmSRYk))) - (1)] = ptr_qQoIJuDP[(1) - (1)]; */
	
	

    }
    ;
}


void nbody_updt(void** buffers_kCJlJluA, void* cl_arg_kCJlJluA)
{
    uint32_t ld_tlJ0FAub = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_kCJlJluA[(1) - (1)]));
    double* ptr_tlJ0FAub = (double*) (STARPU_MATRIX_GET_PTR(buffers_kCJlJluA[(1) - (1)]));
    uint32_t ld_CwAKodfw = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_kCJlJluA[(2) - (1)]));
    double* ptr_CwAKodfw = (double*) (STARPU_MATRIX_GET_PTR(buffers_kCJlJluA[(2) - (1)]));
    uint32_t ld_9CU1xW4b = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_kCJlJluA[(3) - (1)]));
    double* ptr_9CU1xW4b = (double*) (STARPU_MATRIX_GET_PTR(buffers_kCJlJluA[(3) - (1)]));
    double* ptr_D81NeTCr = (double*) (STARPU_VECTOR_GET_PTR(buffers_kCJlJluA[(4) - (1)]));
    int64_t widthp = (int64_t) (STARPU_MATRIX_GET_NY(buffers_kCJlJluA[(1) - (1)]));
    
    int64_t start_FQShiJ9y = (int64_t) (1);
    int64_t stop_FQShiJ9y = (int64_t) (widthp);
    int64_t i;

    for (i = start_FQShiJ9y ; i <= stop_FQShiJ9y ; i += 1)
    {
        ptr_CwAKodfw[((1) + (((i) - (1)) * (ld_CwAKodfw))) - (1)] = (ptr_CwAKodfw[((1) + (((i) - (1)) * (ld_CwAKodfw))) - (1)]) + ((ptr_9CU1xW4b[((1) + (((i) - (1)) * (ld_9CU1xW4b))) - (1)]) * (ptr_D81NeTCr[(2) - (1)]));
        ptr_CwAKodfw[((2) + (((i) - (1)) * (ld_CwAKodfw))) - (1)] = (ptr_CwAKodfw[((2) + (((i) - (1)) * (ld_CwAKodfw))) - (1)]) + ((ptr_9CU1xW4b[((2) + (((i) - (1)) * (ld_9CU1xW4b))) - (1)]) * (ptr_D81NeTCr[(2) - (1)]));
        ptr_tlJ0FAub[((1) + (((i) - (1)) * (ld_tlJ0FAub))) - (1)] = (ptr_tlJ0FAub[((1) + (((i) - (1)) * (ld_tlJ0FAub))) - (1)]) + ((ptr_CwAKodfw[((1) + (((i) - (1)) * (ld_CwAKodfw))) - (1)]) * (ptr_D81NeTCr[(2) - (1)]));
        ptr_tlJ0FAub[((2) + (((i) - (1)) * (ld_tlJ0FAub))) - (1)] = (ptr_tlJ0FAub[((2) + (((i) - (1)) * (ld_tlJ0FAub))) - (1)]) + ((ptr_CwAKodfw[((2) + (((i) - (1)) * (ld_CwAKodfw))) - (1)]) * (ptr_D81NeTCr[(2) - (1)]));
    }
    ;
}


