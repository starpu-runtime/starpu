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

#include <fcntl.h>
#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include "../helper.h"

#ifdef STARPU_HAVE_HDF5
#include <hdf5.h>
#endif

/*
 * Try to write into disk memory
 * Use mechanism to push datas from main ram to disk ram
 * Here we just simulate performing a dumb computation C=A+0, i.e. a mere copy
 * actually
 */

#define NX (16*1024)

int dotest(struct starpu_disk_ops *ops, char *base)
{
	int *A, *C;

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);

	if (ret == -ENODEV) goto enodev;

	/* Initialize path and name */
	const char *name_file_start = "STARPU_DISK_COMPUTE_DATA_";
	const char *name_file_end = "STARPU_DISK_COMPUTE_DATA_RESULT_";

	char * path_file_start = malloc(strlen(base) + 1 + strlen(name_file_start) + 1);
	strcpy(path_file_start, base);
	strcat(path_file_start, "/");
	strcat(path_file_start, name_file_start);

	char * path_file_end = malloc(strlen(base) + 1 + strlen(name_file_end) + 1);
	strcpy(path_file_end, base);
	strcat(path_file_end, "/");
	strcat(path_file_end, name_file_end);

	/* register a disk */
	int new_dd = starpu_disk_register(ops, (void *) base, STARPU_DISK_SIZE_MIN);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT) goto enoent;

	unsigned dd = (unsigned) new_dd;

	/* allocate two memory spaces */
	starpu_malloc_flags((void **)&A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_malloc_flags((void **)&C, NX*sizeof(int), STARPU_MALLOC_COUNT);

	FPRINTF(stderr, "TEST DISK MEMORY \n");

	unsigned int j;
	/* you register them in a vector */
	for(j = 0; j < NX; ++j)
	{
		A[j] = j;
		C[j] = 0;
	}

	/* you create a file to store the vector ON the disk */
	FILE * f = fopen(path_file_start, "wb+");
	if (f == NULL)
		goto enoent2;

	/* store it in the file */
	fwrite(A, sizeof(int), NX, f);

	/* close the file */
	fclose(f);

	int descriptor = open(path_file_start, O_RDWR);
	if (descriptor < 0)
		goto enoent2;
#ifdef STARPU_HAVE_WINDOWS
	_commit(descriptor);
#else
	fsync(descriptor);
#endif
	close(descriptor);

	/* create a file to store result */
	f = fopen(path_file_end, "wb+");
	if (f == NULL)
		goto enoent2;

	/* replace all datas by 0 */
	fwrite(C, sizeof(int), NX, f);

	/* close the file */
	fclose(f);

        descriptor = open(path_file_end, O_RDWR);
#ifdef STARPU_HAVE_WINDOWS
        _commit(descriptor);
#else
        fsync(descriptor);
#endif
	close(descriptor);

	/* And now, you want to use your datas in StarPU */
	/* Open the file ON the disk */
	void * data = starpu_disk_open(dd, (void *) name_file_start, NX*sizeof(int));
	void * data_result = starpu_disk_open(dd, (void *) name_file_end, NX*sizeof(int));

	starpu_data_handle_t vector_handleA, vector_handleC;

	/* register vector in starpu */
	starpu_vector_data_register(&vector_handleA, dd, (uintptr_t) data, NX, sizeof(int));

	/* and do what you want with it, here we copy it into an other vector */
	starpu_vector_data_register(&vector_handleC, dd, (uintptr_t) data_result, NX, sizeof(int));

	starpu_data_cpy(vector_handleC, vector_handleA, 0, NULL, NULL);

	/* free them */
	starpu_data_unregister(vector_handleA);
	starpu_data_unregister(vector_handleC);

	/* close them in StarPU */
	starpu_disk_close(dd, data, NX*sizeof(int));
	starpu_disk_close(dd, data_result, NX*sizeof(int));

	/* check results */
	f = fopen(path_file_end, "rb+");
	if (f == NULL)
		goto enoent2;
	/* take datas */
	size_t read = fread(C, sizeof(int), NX, f);
        STARPU_ASSERT(read == NX);

	/* close the file */
	fclose(f);

	int try = 1;
	for (j = 0; j < NX; ++j)
		if (A[j] != C[j])
		{
			FPRINTF(stderr, "Fail A %d != C %d \n", A[j], C[j]);
			try = 0;
		}

	starpu_free_flags(A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_free_flags(C, NX*sizeof(int), STARPU_MALLOC_COUNT);

	unlink(path_file_start);
	unlink(path_file_end);

	free(path_file_start);
	free(path_file_end);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	if(try)
		FPRINTF(stderr, "TEST SUCCESS\n");
	else
		FPRINTF(stderr, "TEST FAIL\n");
	return try ? EXIT_SUCCESS : EXIT_FAILURE;

enodev:
	return STARPU_TEST_SKIPPED;
enoent2:
	starpu_free_flags(A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_free_flags(C, NX*sizeof(int), STARPU_MALLOC_COUNT);
enoent:
	unlink(path_file_start);
	unlink(path_file_end);

	free(path_file_start);
	free(path_file_end);

	FPRINTF(stderr, "Couldn't write data: ENOENT\n");
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}

#ifdef STARPU_HAVE_HDF5
int dotest_hdf5(struct starpu_disk_ops *ops, char *base)
{
	int *A, *C;
        herr_t status;

        /* Open and close file, just to create an empty file */
        FILE * f = fopen(base, "wb+");
        if (!f)
                goto h5fail2;
        fclose(f);

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);

	if (ret == -ENODEV) goto h5enodev;

	/* Initialize path */
	const char *path_obj_start = "STARPU_DISK_COMPUTE_DATA_";
	const char *path_obj_end = "STARPU_DISK_COMPUTE_DATA_RESULT_";

	/* register a disk */
	int new_dd = starpu_disk_register(ops, (void *) base, STARPU_DISK_SIZE_MIN);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT) goto h5enoent;

	unsigned dd = (unsigned) new_dd;

	/* allocate two memory spaces */
	starpu_malloc_flags((void **)&A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_malloc_flags((void **)&C, NX*sizeof(int), STARPU_MALLOC_COUNT);

	FPRINTF(stderr, "TEST DISK MEMORY \n");

	unsigned int j;
	/* you register them in a vector */
	for(j = 0; j < NX; ++j)
	{
		A[j] = j;
		C[j] = 0;
	}

	/* Open HDF5 file to store data */
        hid_t file = H5Fopen(base, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file < 0)
                goto h5enoent2;

	/* store initial data in the file */
        hsize_t dims[1] = {NX};
        hid_t dataspace = H5Screate_simple(1, dims, NULL);
        if (dataspace < 0)
        {
                H5Fclose(file);
                goto h5fail;
        }

        hid_t dataset = H5Dcreate2(file, path_obj_start, H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset < 0)
        {
                H5Sclose(dataspace);
                H5Fclose(file);
                goto h5fail;
        }

        status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, A);

	/* close the resources before checking the writing */
        H5Dclose(dataset);

        if (status < 0)
        {
                H5Fclose(file);
                goto h5fail;
        }

        /* intialize results in file */
        dataset = H5Dcreate2(file, path_obj_end, H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset < 0)
        {
                H5Sclose(dataspace);
                H5Fclose(file);
                goto h5fail;
        }

        status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, A);

	/* close the resources before checking the writing */
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Fclose(file);

        if (status < 0)
                goto h5fail;

	/* And now, you want to use your datas in StarPU */
	/* Open the file ON the disk */
	void * data = starpu_disk_open(dd, (void *) path_obj_start, NX*sizeof(int));
	void * data_result = starpu_disk_open(dd, (void *) path_obj_end, NX*sizeof(int));

	starpu_data_handle_t vector_handleA, vector_handleC;

	/* register vector in starpu */
	starpu_vector_data_register(&vector_handleA, dd, (uintptr_t) data, NX, sizeof(int));

	/* and do what you want with it, here we copy it into an other vector */
	starpu_vector_data_register(&vector_handleC, dd, (uintptr_t) data_result, NX, sizeof(int));

	starpu_data_cpy(vector_handleC, vector_handleA, 0, NULL, NULL);

	/* free them */
	starpu_data_unregister(vector_handleA);
	starpu_data_unregister(vector_handleC);

	/* close them in StarPU */
	starpu_disk_close(dd, data, NX*sizeof(int));
	starpu_disk_close(dd, data_result, NX*sizeof(int));

	/* check results */
        file = H5Fopen(base, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file < 0)
                goto h5enoent2;

        dataset = H5Dopen2(file, path_obj_end, H5P_DEFAULT);
        if (dataset < 0)
        {
                H5Fclose(file);
                goto h5fail;
        }

        status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, C);

	/* close the resources before checking the writing */
        H5Dclose(dataset);
        H5Fclose(file);

        if (status < 0)
                goto h5fail;

	int try = 1;
	for (j = 0; j < NX; ++j)
		if (A[j] != C[j])
		{
			FPRINTF(stderr, "Fail A %d != C %d \n", A[j], C[j]);
			try = 0;
		}

	starpu_free_flags(A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_free_flags(C, NX*sizeof(int), STARPU_MALLOC_COUNT);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

        unlink(base);

	if(try)
		FPRINTF(stderr, "TEST SUCCESS\n");
	else
		FPRINTF(stderr, "TEST FAIL\n");
	return (try ? EXIT_SUCCESS : EXIT_FAILURE);

h5enodev:
        unlink(base);
	return STARPU_TEST_SKIPPED;
h5enoent2:
	starpu_free_flags(A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_free_flags(C, NX*sizeof(int), STARPU_MALLOC_COUNT);
h5enoent:
	FPRINTF(stderr, "Couldn't write data: ENOENT\n");
	starpu_shutdown();
        unlink(base);
	return STARPU_TEST_SKIPPED;
h5fail:
	starpu_free_flags(A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_free_flags(C, NX*sizeof(int), STARPU_MALLOC_COUNT);

	starpu_shutdown();
        unlink(base);
h5fail2:
	FPRINTF(stderr, "Something goes wrong with HDF5 dataset/dataspace/write \n");
        return EXIT_FAILURE;

}
#endif

static int merge_result(int old, int new)
{
	if (new == EXIT_FAILURE)
		return EXIT_FAILURE;
	if (old == 0)
		return 0;
	return new;
}

int main(void)
{
	int ret = 0;
	int ret2;
	char s[128];
	char *ptr;

#ifdef STARPU_HAVE_SETENV
	setenv("STARPU_CALIBRATE_MINIMUM", "1", 1);
#endif

	snprintf(s, sizeof(s), "/tmp/%s-disk-XXXXXX", getenv("USER"));
	ptr = _starpu_mkdtemp(s);
	if (!ptr)
	{
		FPRINTF(stderr, "Cannot make directory '%s'\n", s);
		return STARPU_TEST_SKIPPED;
	}

	ret = merge_result(ret, dotest(&starpu_disk_stdio_ops, s));
	ret = merge_result(ret, dotest(&starpu_disk_unistd_ops, s));
#ifdef STARPU_LINUX_SYS
	if ((NX * sizeof(int)) % getpagesize() == 0)
	{
		ret = merge_result(ret, dotest(&starpu_disk_unistd_o_direct_ops, s));
	}
	else
	{
		ret = merge_result(ret, STARPU_TEST_SKIPPED);
	}
#endif
#ifdef STARPU_HAVE_HDF5
        char hdf5_base[128];
        strcpy(hdf5_base, s);
        strcat(hdf5_base, "/STARPU_HDF5_file.h5");

        ret = merge_result(ret, dotest_hdf5(&starpu_disk_hdf5_ops, hdf5_base));
#endif

	ret2 = rmdir(s);
	if (ret2 < 0)
		STARPU_CHECK_RETURN_VALUE(-errno, "rmdir '%s'\n", s);
	return ret;
}
