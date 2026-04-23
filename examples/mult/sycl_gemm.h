/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2026-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int syclblasSgemm(unsigned nxC, unsigned nyC, unsigned nyA,
                  float alpha, const float *subA, unsigned ldA,
                  const float *subB, unsigned ldB, float beta,
                  float *subC, unsigned ldC);

int syclblasDgemm(unsigned nxC, unsigned nyC, unsigned nyA,
                  double alpha, const double *subA, unsigned ldA,
                  const double *subB, unsigned ldB, double beta,
                  double *subC, unsigned ldC);

#ifdef __cplusplus
}
#endif

