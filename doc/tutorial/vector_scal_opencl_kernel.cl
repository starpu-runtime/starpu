/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 *
 * Permission is granted to copy, distribute and/or modify this document
 * under the terms of the GNU Free Documentation License, Version 1.3
 * or any later version published by the Free Software Foundation;
 * with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
 * See the GNU Free Documentation License in COPYING.GFDL for more details.
 */

__kernel void vector_mult_opencl(__global float* val, int nx, float factor)
{
        const int i = get_global_id(0);
        if (i < nx) {
                val[i] *= factor;
        }
}

