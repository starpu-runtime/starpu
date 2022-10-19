/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <hdfs.h>
#include <fr_labri_hpccloud_starpu_StarPU.h>
#include "starpujni-hdfs.h"
#include "starpujni-common.h"

#define DISK_PREFIX "/tmp/starpujni"
enum writemode
{
	APPEND = 0,
	CREATE = 1
};

#define START_TEST() \
	do {		   \
		fprintf(stderr, "***\n"   \
			 "*** STARTING NATIVE TEST %s\n"	\
			 "***\n", __FUNCTION__);		\
	} while(0)

#define VECTOR_SIZE 1000
#define NB_HANDLES 1000
static void test_swap(void)
{
	int i;
	starpu_data_handle_t handles[NB_HANDLES];

	START_TEST();

	for (i = 0; i < NB_HANDLES; i++)
	{
		int j;
		int *vec;
		starpu_vector_data_register(handles + i, -1, 0, VECTOR_SIZE, sizeof(uintptr_t));
		starpu_data_acquire(handles[i], STARPU_W);
		vec = (int *) starpu_vector_get_local_ptr(handles[i]);
		for (j = 0; j < VECTOR_SIZE; j++)
			vec[j] = j;
		starpu_data_release(handles[i]);
	}
	for (i = 0; i < NB_HANDLES; i++)
		starpu_data_unregister(handles[i]);
}

static hdfsFile s_crt_file_for_write(hdfsFS fs, const char *filename, enum writemode mode)
{
	int m = O_WRONLY | (mode == CREATE ? O_CREAT : O_APPEND);
	hdfsFile writeFile;

#if STARPUJNI_HADOOP_MAJOR >= 3 && STARPUJNI_HADOOP_MINOR >= 1
	struct hdfsStreamBuilder *sb = hdfsStreamBuilderAlloc(fs, filename, m);

	if (sb != NULL)
		writeFile = hdfsStreamBuilderBuild(sb);
	else
		writeFile = NULL;
#else
	writeFile = hdfsOpenFile(fs, filename, m, 0, 0, 0);
#endif

	if (!writeFile)
	{
		fprintf(stderr, "Failed to open %s for writing!\n", filename);
		exit(-1);
	}
	return writeFile;
}

static void test_hdfs(void)
{
	static const char *test_string = "Hello, World !";
	size_t test_string_len = strlen(test_string);
	START_TEST();
	hdfsFS fs = hdfsConnect("default", 9000);
	const char *path = DISK_PREFIX "/testfile.txt";

	hdfsFile stream = s_crt_file_for_write(fs, path, CREATE);
	hdfsWrite(fs, stream, (void *) test_string, test_string_len + 1);
	if (hdfsFlush(fs, stream))
	{
		fprintf(stderr, "Failed to 'flush' %s\n", path);
		exit(-1);
	}
	hdfsCloseFile(fs, stream);

	size_t bufsize = test_string_len + 1;
	char *buffer = malloc(bufsize);
	stream = hdfsOpenFile(fs, path, O_RDONLY, 0, 0, 0);

	if (hdfsRead(fs, stream, (void *) buffer, bufsize) != bufsize)
	{
		fprintf(stderr, "Failed to read %ld bytes from %s\n", bufsize, path);
		exit(-1);
	}
	if (strcmp(buffer, test_string) != 0)
	{
		buffer[bufsize - 1] = 0;
		fprintf(stderr, "invalid read buffer '%s'.\n", buffer);
		exit(-1);
	}
	free(buffer);
	hdfsCloseFile(fs, stream);
}

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

static void test_hdfs_disk(void)
{
	START_TEST();

	/* size of one vector */
	static const int NX = (30 * 1000000 / sizeof(double));
	double *A, /**B, *C, *D, *E, */ *F;

	/* allocate two memory spaces */
	starpu_malloc_flags((void **) &A, NX * sizeof(double), STARPU_MALLOC_COUNT);
	starpu_malloc_flags((void **) &F, NX * sizeof(double), STARPU_MALLOC_COUNT);
	FPRINTF(stderr, "TEST DISK MEMORY \n");

	unsigned int j;
	/* initialization with bad values */
	for (j = 0; j < NX; ++j)
	{
		A[j] = j;
		F[j] = -j;
	}
	starpu_data_handle_t vector_handleA, vector_handleB, vector_handleC, vector_handleD, vector_handleE, vector_handleF;
	/* register vector in starpu */
	starpu_vector_data_register(&vector_handleA, STARPU_MAIN_RAM, (uintptr_t) A, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleB, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleC, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleD, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleE, -1, (uintptr_t) NULL, NX, sizeof(double));
	starpu_vector_data_register(&vector_handleF, STARPU_MAIN_RAM, (uintptr_t) F, NX, sizeof(double));
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
			FPRINTF(stderr, "Fail A %f != F %f \n", A[j], F[j]);
			try = 0;
		}
	/* free last vectors */
	starpu_free_flags(A, NX * sizeof(double), STARPU_MALLOC_COUNT);
	starpu_free_flags(F, NX * sizeof(double), STARPU_MALLOC_COUNT);

	/* terminate StarPU, no task can be submitted after */
	if (try)
		FPRINTF(stderr, "TEST SUCCESS\n");
	else
		FPRINTF(stderr, "TEST FAIL\n");
}

JNIEXPORT jboolean JNICALL STARPUJNI_FUNCNAME(StarPU, runNativeTests)(JNIEnv *env, jclass cls)
{
	/* limit main ram to force to push in disk */
	setenv("STARPU_LIMIT_CPU_MEM", "160", 1);
	/* Initialize StarPU with default configuration */
	/* register a disk */
	if (getenv("STARPUJNI_NO_SRAND48") == NULL)
		srand48(time(NULL)+getpid());
	int new_dd = starpu_disk_register(&starpujni_disk_hdfs_ops,
					  (void *) DISK_PREFIX, 1024 * 1024 * 200);
	/* can't write on /tmp/ */
	if (new_dd == -ENOENT)
	{
		FPRINTF(stderr, "Can not create disk\n");
		return JNI_FALSE;
	}

	test_swap();
	test_hdfs();
	test_hdfs_disk();

	return JNI_TRUE;
}
