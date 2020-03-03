/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * Use mechanism to push data from disk to disk.
 */

#define NX (16*1024)

int dotest(struct starpu_disk_ops *ops, char *base)
{
	int *A, *C;

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);
	if (ret == -ENODEV) goto enodev;

	/* Initialize path and name */
	const char *name_file_start = "STARPU_DISK_COMPUTE_DATA";
	const char *name_dir_src = "src";
	const char *name_dir_dst = "dst";

	char * path_file_start = malloc(strlen(base) + 1 + strlen(name_dir_src) + 1 + strlen(name_file_start) + 1);
	strcpy(path_file_start, base);
	strcat(path_file_start, "/");
	strcat(path_file_start, name_dir_src);
	strcat(path_file_start, "/");
	strcat(path_file_start, name_file_start);

        char * base_src = malloc(strlen(base) + 1 + strlen(name_dir_src) + 1);
        strcpy(base_src, base);
        strcat(base_src, "/");
        strcat(base_src, name_dir_src);

        char * base_dst = malloc(strlen(base) + 1 + strlen(name_dir_dst) + 1);
        strcpy(base_dst, base);
        strcat(base_dst, "/");
        strcat(base_dst, name_dir_dst);

	/* register a disks */
	int disk_src = starpu_disk_register(ops, (void *) base_src, STARPU_DISK_SIZE_MIN);
	if (disk_src == -ENOENT) goto enoent;

	int disk_dst = starpu_disk_register(ops, (void *) base_dst, STARPU_DISK_SIZE_MIN);
	if (disk_dst == -ENOENT) goto enoent;

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

	/* And now, you want to use your datas in StarPU */
	/* Open the file ON the disk */
	void * data = starpu_disk_open(disk_src, (void *) name_file_start, NX*sizeof(int));

	starpu_data_handle_t vector_handleA;
	starpu_vector_data_register(&vector_handleA, disk_src, (uintptr_t) data, NX, sizeof(int));

	/* Move and invalidate copy to an other disk */
	starpu_data_acquire_on_node(vector_handleA, disk_dst, STARPU_RW);
	starpu_data_release_on_node(vector_handleA, disk_dst);

	starpu_data_acquire_on_node(vector_handleA, disk_src, STARPU_RW);
	starpu_data_release_on_node(vector_handleA, disk_src);

	/* free them */
	starpu_data_unregister(vector_handleA);

	/* close them in StarPU */
	starpu_disk_close(disk_src, data, NX*sizeof(int));

	/* check results */
	f = fopen(path_file_start, "rb+");
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

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

	unlink(path_file_start);
	rmdir(base_src);

	free(base_src);
	free(base_dst);
	free(path_file_start);

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
	free(base_src);
	free(base_dst);
	free(path_file_start);

	FPRINTF(stderr, "Couldn't write data: ENOENT\n");
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}

#ifdef STARPU_HAVE_HDF5
int dotest_hdf5(struct starpu_disk_ops *ops, char *base)
{
	int *A, *C;
        herr_t status;


	/* Initialize path */
	const char *path_obj_start = "STARPU_DISK_COMPUTE_DATA";
	const char *name_hdf5_start = "STARPU_HDF5_src_file.h5";
	const char *name_hdf5_end = "STARPU_HDF5_dst_file.h5";

        char * hdf5_base_src = malloc(strlen(base) + 1 + strlen(name_hdf5_start) + 1);
        strcpy(hdf5_base_src, base);
        strcat(hdf5_base_src, "/");
        strcat(hdf5_base_src, name_hdf5_start);

        char * hdf5_base_dst = malloc(strlen(base) + 1 + strlen(name_hdf5_end) + 1);
        strcpy(hdf5_base_dst, base);
        strcat(hdf5_base_dst, "/");
        strcat(hdf5_base_dst, name_hdf5_end);

        /* Open and close files, just to create empty files */
        FILE * file_src = fopen(hdf5_base_src, "wb+");
        if (!file_src)
                goto h5fail2;
        fclose(file_src);

        FILE * file_dst = fopen(hdf5_base_dst, "wb+");
        if (!file_dst)
	{
                goto h5fail2;
	}
        fclose(file_dst);

	/* Initialize StarPU with default configuration */
	int ret = starpu_init(NULL);
	if (ret == -ENODEV) goto h5enodev;

	/* register disks */
	int disk_src = starpu_disk_register(ops, (void *) hdf5_base_src, STARPU_DISK_SIZE_MIN);
	if (disk_src == -ENOENT) goto h5enoent;

	int disk_dst = starpu_disk_register(ops, (void *) hdf5_base_dst, STARPU_DISK_SIZE_MIN);
	if (disk_dst == -ENOENT) goto h5enoent;

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
        hid_t file = H5Fopen(hdf5_base_src, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file < 0)
                goto h5fail;

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

        H5Sclose(dataspace);
        H5Fclose(file);

	/* Open the file ON the disk */
	void * data = starpu_disk_open(disk_src, (void *) path_obj_start, NX*sizeof(int));

	starpu_data_handle_t vector_handleA;
	starpu_vector_data_register(&vector_handleA, disk_src, (uintptr_t) data, NX, sizeof(int));

	/* Move and invalidate copy to an other disk */
	starpu_data_acquire_on_node(vector_handleA, disk_dst, STARPU_RW);
	starpu_data_release_on_node(vector_handleA, disk_dst);

	starpu_data_acquire_on_node(vector_handleA, disk_src, STARPU_RW);
	starpu_data_release_on_node(vector_handleA, disk_src);

	starpu_data_unregister(vector_handleA);

	/* close them in StarPU */
	starpu_disk_close(disk_src, data, NX*sizeof(int));

	/* check results */
        file = H5Fopen(hdf5_base_src, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file < 0)
                goto h5fail;

        dataset = H5Dopen2(file, path_obj_start, H5P_DEFAULT);
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

        unlink(hdf5_base_src);
        unlink(hdf5_base_dst);

	free(hdf5_base_src);
	free(hdf5_base_dst);

	if(try)
		FPRINTF(stderr, "TEST SUCCESS\n");
	else
		FPRINTF(stderr, "TEST FAIL\n");
	return (try ? EXIT_SUCCESS : EXIT_FAILURE);

h5fail:
	starpu_free_flags(A, NX*sizeof(int), STARPU_MALLOC_COUNT);
	starpu_free_flags(C, NX*sizeof(int), STARPU_MALLOC_COUNT);
h5enoent:
	FPRINTF(stderr, "Couldn't write data: ENOENT\n");
	starpu_shutdown();
h5enodev:
        unlink(hdf5_base_src);
        unlink(hdf5_base_dst);
	free(hdf5_base_src);
	free(hdf5_base_dst);
	return STARPU_TEST_SKIPPED;
h5fail2:
	free(hdf5_base_src);
	free(hdf5_base_dst);
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
        ret = merge_result(ret, dotest_hdf5(&starpu_disk_hdf5_ops, s));
#endif

	ret2 = rmdir(s);
	if (ret2 < 0)
		STARPU_CHECK_RETURN_VALUE(-errno, "rmdir '%s'\n", s);
	return ret;
}
