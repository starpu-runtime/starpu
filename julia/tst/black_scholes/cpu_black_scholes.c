/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
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
#include <stdint.h>
#include <starpu.h>
#include <math.h>


static inline double normcdf(double x)
{
	
	return (1.0 + erf(x/sqrt(2.0)))/2.0;
}

void cpu_black_scholes(void *descr[], void *arg)
{ 
	double *S, *K, *R, *T, *SIG, *CRES, *PRES;

	uint32_t nxS;
	
	
	S = (double *)STARPU_MATRIX_GET_PTR(descr[0]);
	K = (double *)STARPU_MATRIX_GET_PTR(descr[1]);
	R = (double *)STARPU_MATRIX_GET_PTR(descr[2]);
	T = (double *)STARPU_MATRIX_GET_PTR(descr[3]);
	SIG = (double *)STARPU_MATRIX_GET_PTR(descr[4]);
	CRES = (double *)STARPU_MATRIX_GET_PTR(descr[5]);
	PRES = (double *)STARPU_MATRIX_GET_PTR(descr[6]);
	
	nxS = STARPU_MATRIX_GET_NX(descr[0]);

	
	uint32_t i;
	for (i = 0; i < nxS; i++){
				
		double d1 = (log(S[i] / K[i]) + (R[i] + pow(SIG[i], 2.0) * 0.5) * T[i]) / (SIG[i] * sqrt(T[i]));
		double d2 = (log(S[i] / K[i]) + (R[i] - pow(SIG[i], 2.0) * 0.5) * T[i]) / (SIG[i] * sqrt(T[i]));
		
		CRES[i] = S[i] * normcdf(d1) - K[i] * exp(-R[i] * T[i]) * normcdf(d2);
		PRES[i] = -S[i] * normcdf(-d1) + K[i] * exp(-R[i] * T[i]) * normcdf(-d2);
	}
}
