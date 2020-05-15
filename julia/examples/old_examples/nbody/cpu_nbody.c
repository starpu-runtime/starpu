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
#include <stdint.h>
#include <starpu.h>

struct Param {
	unsigned taskx;
	double epsilon;
};

void cpu_nbody(void *descr[], void *arg)
{
	struct Param *params = arg;

	double *P;
	double *subA;
	double *M;

	uint32_t nxP, nxA, nxM;
	uint32_t ldP, ldA, ldM;

	P = (double *)STARPU_MATRIX_GET_PTR(descr[0]);
	subA = (double *)STARPU_MATRIX_GET_PTR(descr[1]);
	M = (double *)STARPU_MATRIX_GET_PTR(descr[2]);

	nxP = STARPU_MATRIX_GET_NX(descr[0]);
	nxA = STARPU_MATRIX_GET_NX(descr[1]);
	nxM = STARPU_MATRIX_GET_NX(descr[2]);

	ldP = STARPU_MATRIX_GET_LD(descr[0]);
	ldA = STARPU_MATRIX_GET_LD(descr[1]);
	ldM = STARPU_MATRIX_GET_LD(descr[2]);

	double epsilon = params->epsilon;

	unsigned id = nxA * params->taskx;

	uint32_t i,j;
	
	for (i = 0; i < nxA; i++){
		double sumaccx = 0;
		double sumaccy = 0;
		
		for (j = 0; j < nxP; j++){
			
			if (j != i + id){
				
				double dx = P[j] - P[i + id];
				double dy = P[j + ldP] - P[i + id + ldP];

				double modul = sqrt(dx * dx + dy * dy);

				sumaccx = sumaccx + 6.67e-11 * M[j] * dx / pow(modul + epsilon, 3);
				sumaccy = sumaccy + 6.67e-11 * M[j] * dy / pow(modul + epsilon, 3);
			}

		}
		subA[i] = sumaccx;
		subA[i + ldA] = sumaccy;
	}
}

void cpu_nbody2(void *descr[], void *arg)
{
	double *subP;
	double *subV;
	double *subA;

	uint32_t nxP, nxV, nxA;
	uint32_t ldP, ldV, ldA;

	subP = (double *)STARPU_MATRIX_GET_PTR(descr[0]);
	subV = (double *)STARPU_MATRIX_GET_PTR(descr[1]);
	subA = (double *)STARPU_MATRIX_GET_PTR(descr[2]);

	nxP = STARPU_MATRIX_GET_NX(descr[0]);
	nxV = STARPU_MATRIX_GET_NX(descr[1]);
	nxA = STARPU_MATRIX_GET_NX(descr[2]);
	
	ldP = STARPU_MATRIX_GET_LD(descr[0]);
	ldV = STARPU_MATRIX_GET_LD(descr[1]);
	ldA = STARPU_MATRIX_GET_LD(descr[2]);
	
	
	unsigned i,dt;
	dt = 3600;
	for (i = 0; i < nxP; i++){
	
		subV[i] = subV[i] + dt*subA[i];
		subV[i + ldV] = subV[i + ldV] + dt*subA[i + ldA];

		subP[i] = subP[i] + dt*subV[i];
		subP[i + ldP] = subP[i + ldP] + dt*subV[i + ldV];
	}
}
	      
