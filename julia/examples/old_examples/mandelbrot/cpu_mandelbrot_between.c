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

struct Params
{
	float cr;
	float ci;
	unsigned taskx;
	unsigned tasky;
	unsigned width;
	unsigned height;
};

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

void mandelbrot(void** buffers_86BwRM71, void* cl_arg_86BwRM71)
{
    uint32_t ld_o2BQqRir = (uint32_t) (STARPU_MATRIX_GET_LD(buffers_86BwRM71[(1) - (1)]));
    int64_t* ptr_o2BQqRir = (int64_t*) (STARPU_MATRIX_GET_PTR(buffers_86BwRM71[(1) - (1)]));
    
//ARRAY PAR
    double* ptr_Ul4Ys0Mt = (double*) (STARPU_VECTOR_GET_PTR(buffers_86BwRM71[(2) - (1)]));
    int64_t* ptr_cE3zj60d = (int64_t*) (STARPU_VECTOR_GET_PTR(buffers_86BwRM71[(3) - (1)]));
//

    int64_t local_width = (int64_t) (STARPU_MATRIX_GET_NY(buffers_86BwRM71[(1) - (1)]));
    int64_t local_height = (int64_t) (STARPU_MATRIX_GET_NX(buffers_86BwRM71[(1) - (1)]));
    double conv_limit = (double) (2.0);

//STRUCT PAR
    
    /* struct Params *params = cl_arg_86BwRM71; */
  

    /* double centerr = params->cr; */
    /* double centeri = params->ci; */

    /* unsigned Idx = params->taskx; */
    /* unsigned Idy = params->tasky; */

    /* unsigned width = params->width; */
    /* unsigned height = params->height; */

    /* /\* printf("cr / ci: %f / %f\n", centerr, centeri); *\/ */

    /* int64_t zoom = width * 0.25296875; */
    /* int64_t max_iterations = (width/2) * 0.049715909 * log10(zoom); */

//


    int64_t start_qxJwMzwA = (int64_t) (1);
    int64_t stop_qxJwMzwA = (int64_t) (local_width);
    int64_t x;

    for (x = start_qxJwMzwA ; x <= stop_qxJwMzwA ; x += 1)
    {
        
        int64_t start_ekV9GHK1 = (int64_t) (1);
        int64_t stop_ekV9GHK1 = (int64_t) (local_height);
        int64_t y;

        for (y = start_ekV9GHK1 ; y <= stop_ekV9GHK1 ; y += 1)
        {
	    //ARRAY PAR
	    double max_iterations = (double) (ptr_Ul4Ys0Mt[(5) - (1)]);
            double zoom = (double) ((ptr_Ul4Ys0Mt[(3) - (1)]) * (0.25296875));
	    //

	    //STRUCT PAR
	    /* double X = x + Idy*local_width; */
	    /* double Y = y + Idx*local_height; */

	    /* double cr = centerr + (X - (width / 2))/zoom; */
	    /* double ci = centeri + (Y - (height / 2))/zoom; */
	    //

	    //ARRAY PAR
            int64_t X = (int64_t) ((x) + ((local_width) * ((ptr_cE3zj60d[(2) - (1)]) - (1))));
            int64_t Y = (int64_t) ((y) + ((local_height) * ((ptr_cE3zj60d[(1) - (1)]) - (1))));

	    double cr = (double) ((ptr_Ul4Ys0Mt[(1) - (1)]) + (((X) - ((ptr_Ul4Ys0Mt[(3) - (1)]) / (2))) / (zoom)));
            double ci = (double) ((ptr_Ul4Ys0Mt[(2) - (1)]) + (((Y) - ((ptr_Ul4Ys0Mt[(4) - (1)]) / (2))) / (zoom)));
	    //


            double zi = (double) (ci);
            double zr = (double) (cr);


            int64_t n = (int64_t) (0);

	    float m = zr * zr + zi * zi;
            /* int64_t b1 = (int64_t) (((n) < (max_iterations)) + ((((zr) * (zr)) + ((zi) * (zi))) < ((conv_limit) * (conv_limit)))); */
            
            /* while ((b1) >= (2)) */
	    /* printf("%d\n", max_iterations); */

            for (n = 0; n < max_iterations && m < conv_limit * conv_limit; n++)
	    {
                double tmp = (double) ((((zr) * (zr)) - ((zi) * (zi))) + (cr));
                zi = ((2) * (zr) * (zi)) + (ci);
                zr = tmp;
                /* n = (n) + (1); */
		m = zr*zr + zi*zi;
                /* b1 = ((n) <= (max_iterations)) + ((((zr) * (zr)) + ((zi) * (zi))) <= ((conv_limit) * (conv_limit))); */
            }
            ;
	    
	    /* printf("n: %d\n max_iter: %d\n", n, max_iterations); */
            if ((n) < (max_iterations))
            {
		    ptr_o2BQqRir[((y) + (((x) - (1)) * (ld_o2BQqRir))) - (1)] = 255 * (1.0 * n / (max_iterations));
            } else
            {
                ptr_o2BQqRir[((y) + (((x) - (1)) * (ld_o2BQqRir))) - (1)] = 0;
            }
            ;
        }
        ;
    }
    ;
}


