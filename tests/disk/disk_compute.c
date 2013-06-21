/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013 Corentin Salingue
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

/* Try to write into disk memory
 * Use mechanism to push datas from main ram to disk ram
 */

#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NX (100)

int main(int argc, char **argv)
{
	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);

	if (ret == -ENODEV) goto enodev;

	/* register a disk */
	int new_dd = starpu_disk_register(&starpu_disk_stdio_ops, (void *) "/tmp/", 1024*1024*1);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT) goto enoent;
	
	unsigned dd = (unsigned) new_dd;

	printf("TEST DISK MEMORY \n");

	/* Imagine, you want to compute datas */
	int *A;
	int *C;

	starpu_malloc_flags((void **)&A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_malloc_flags((void **)&C, NX*sizeof(int), STARPU_MALLOC_COUNT);
 
	unsigned int j;
	/* you register them in a vector */
	for(j = 0; j < NX; ++j)
	{
		A[j] = j;
	}




	/* you create a file to store the vector ON the disk */
	FILE * f = fopen("/tmp/STARPU_DISK_COMPUTE_DATA", "wb+");
	/* fail */
	if (f == NULL)
		goto enoent;

	/* store it in the file */
	fwrite(A, sizeof(int), NX, f);

	/* close the file */
	fclose(f);

	/* And now, you want to use your datas in StarPU */
	/* Open the file ON the disk */
	void * data = starpu_disk_open(dd, (void *) "STARPU_DISK_COMPUTE_DATA", NX*sizeof(int));

	starpu_data_handle_t vector_handleA, vector_handleC;

	/* register vector in starpu */
	starpu_vector_data_register(&vector_handleA, dd, (uintptr_t) data, NX, sizeof(int));

	/* and do what you want with it, here we copy it into an other vector */ 
	starpu_vector_data_register(&vector_handleC, STARPU_MAIN_RAM, (uintptr_t) C, NX, sizeof(int));	

	starpu_data_cpy(vector_handleC, vector_handleA, 0, NULL, NULL);

	/* free them */
	starpu_data_unregister(vector_handleA);
	starpu_data_unregister(vector_handleC);

	/* close it in StarPU */
	starpu_disk_close(dd, data, NX*sizeof(int));
	
	int try = 1;
	for (j = 0; j < NX; ++j)
		if (A[j] != C[j])
		{
			printf("Fail A %d != C %d \n", A[j], C[j]);
			try = 0;
		}

	starpu_free_flags(A, NX*sizeof(double), STARPU_MALLOC_COUNT);
	starpu_free_flags(C, NX*sizeof(double), STARPU_MALLOC_COUNT);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	if(try)
		printf("TEST SUCCESS\n");
	else
		printf("TEST FAIL\n");
	return (try ? EXIT_SUCCESS : EXIT_FAILURE);

enodev:
	return 77;
enoent:
	return 77;
}
