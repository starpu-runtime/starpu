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

#define NX (30*1000000/sizeof(int))

int main(int argc, char **argv)
{
	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);

	if (ret == -ENODEV) goto enodev;

	/* register a disk */
	int new_dd = starpu_disk_register(&starpu_disk_stdio_ops, (void *) "/tmp/", 1024*1024*40);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT) goto enoent;
	
	unsigned dd = (unsigned) new_dd;

	printf("TEST DISK MEMORY \n");

	/* Imagine, you want to compute datas */
	int A[NX];
	int C[NX];
	unsigned int j;
	/* you register them in a vector */
	for(j = 0; j < NX; ++j)
	{
		A[j] = j;
	}

	/* you create a file to store the vector ON the disk */
	FILE * f = fopen("/tmp/STARPU_DISK_COMPUTE_DATA", "rb+");
	/* fail */
	if (f == NULL)
		goto enoent;


	/* store it in the file */
	fwrite((void *) A, sizeof(int), NX, f);

	/* close the file */
	fclose(f);

	/* And now, you want to use your datas in StarPU */
	/* Open the file ON the disk */
	void * data = starpu_disk_open(dd, (void *) "STARPU_DISK_COMPUTE_DATA", NX*sizeof(int));

	starpu_data_handle_t vector_handleA, vector_handleB;

	/* register vector in starpu */
	starpu_vector_data_register(&vector_handleA, dd, (uintptr_t)A, NX, sizeof(int));

	/* and do what you want with it, here we copy it into an other vector */ 
	starpu_vector_data_register(&vector_handleB, STARPU_MAIN_RAM, (uintptr_t) NULL, NX, sizeof(int));	

	starpu_data_cpy(vector_handleB, vector_handleA, 0, NULL, NULL);

	/* free them */
	starpu_data_unregister(vector_handleA);
	starpu_data_unregister(vector_handleB);

	/* close it in StarPU */
	starpu_disk_close(dd, data, NX*sizeof(int));

	/* check if it's correct */
	f = fopen("/tmp/STARPU_DISK_COMPUTE_DATA", "rb+");
	/* fail */
	if (f == NULL)
		goto enoent;
	int size = fread(C, sizeof(int), NX, f);

	int try = 1;
	for (j = 0; j < NX; ++j)
		if (A[j] != C[j])
		{
			printf("Fail A %d != C %d \n", A[j], C[j]);
			try = 0;
		}

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
