/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Corentin Salingue
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

//! [To be included. You should update doxygen if you see this text.]

/* Try to write into disk memory
 * Use mechanism to push datas from main ram to disk ram
 */

#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* size of one vector */
#define	NX	(30*1000000/sizeof(double))
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)


int main(int argc, char **argv)
{
	double * A,*B,*C,*D,*E,*F;

	/* limit main ram to force to push in disk */
	setenv("STARPU_LIMIT_CPU_MEM", "160", 1);

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);

	if (ret == -ENODEV) goto enodev;

	/* register a disk */
	int new_dd = starpu_disk_register(&starpu_disk_unistd_ops, (void *) "/tmp/", 1024*1024*200);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT) goto enoent;

	/* allocate two memory spaces */
	starpu_malloc_flags((void **)&A, NX*sizeof(double), STARPU_MALLOC_COUNT);
	starpu_malloc_flags((void **)&F, NX*sizeof(double), STARPU_MALLOC_COUNT);

	FPRINTF(stderr, "TEST DISK MEMORY \n");

	unsigned int j;
	/* initialization with bad values */
	for(j = 0; j < NX; ++j)
	{
		A[j] = j;
		F[j] = -j;
	}

	starpu_data_handle_t vector_handleA, vector_handleB, vector_handleC, vector_handleD, vector_handleE, vector_handleF;

	/* register vector in starpu */
	starpu_vector_data_register(&vector_handleA, STARPU_MAIN_RAM, (uintptr_t)A, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleB, -1, (uintptr_t) NULL, NX, sizeof(double));	
	starpu_vector_data_register(&vector_handleC, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleD, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleE, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleF, STARPU_MAIN_RAM, (uintptr_t)F, NX, sizeof(double));

	/* copy vector A->B, B->C... */
	starpu_data_cpy(vector_handleB, vector_handleA, 0, NULL, NULL);
	starpu_data_cpy(vector_handleC, vector_handleB, 0, NULL, NULL);
	starpu_data_cpy(vector_handleD, vector_handleC, 0, NULL, NULL);
	starpu_data_cpy(vector_handleE, vector_handleD, 0, NULL, NULL);
	starpu_data_cpy(vector_handleF, vector_handleE, 0, NULL, NULL);

	/* StarPU does not need to manipulate the array anymore so we can stop
 	 * monitoring it */

	/* free them */
	starpu_data_unregister(vector_handleA);
	starpu_data_unregister(vector_handleB);
	starpu_data_unregister(vector_handleC);
	starpu_data_unregister(vector_handleD);
	starpu_data_unregister(vector_handleE);
	starpu_data_unregister(vector_handleF);

	/* check if computation is correct */
	int try = 1;
	for (j = 0; j < NX; ++j)
		if (A[j] != F[j])
		{
			printf("Fail A %f != F %f \n", A[j], F[j]);
			try = 0;
		}

	/* free last vectors */
	starpu_free_flags(A, NX*sizeof(double), STARPU_MALLOC_COUNT);
	starpu_free_flags(F, NX*sizeof(double), STARPU_MALLOC_COUNT);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	if(try)
		FPRINTF(stderr, "TEST SUCCESS\n");
	else
		FPRINTF(stderr, "TEST FAIL\n");
	return (try ? EXIT_SUCCESS : EXIT_FAILURE);

enodev:
	return 77;
enoent:
	return 77;
}

//! [To be included. You should update doxygen if you see this text.]
