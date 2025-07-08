/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2024-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <sys/mman.h>
#include <starpu_mpi.h>
#include "helper.h"
#include <common/config.h>

static FILE *_f_buffer;
static char _f_buffer_name[40];
static int check=1, display=1, silent=0;
#define vector 0
#define matrix 1
#define block  2
#define tensor 3
#define ndim   4
#define nb_tests 5
static int tests_data[nb_tests] = {1, 1, 1, 1, 1};
static int test_small=1, test_large=1;
static int rank_comm, size_comm;

static void dump()
{
	fclose(_f_buffer);
	char *buffer = 0;
	long length;
	FILE *f = fopen(_f_buffer_name, "rb");

	assert(f);
	if (f)
	{
		fseek(f, 0, SEEK_END);
		length = ftell(f);
		if (length)
		{
			fseek(f, 0, SEEK_SET);
			buffer = malloc(length);
			assert(fread(buffer, 1, length, f) > 0);
		}
		fclose(f);
		unlink(_f_buffer_name);
	}
	if (rank_comm != 0)
	{
		MPI_Send(&length, 1, MPI_LONG, 0, rank_comm, MPI_COMM_WORLD);
		MPI_Send(buffer, length, MPI_CHAR, 0, rank_comm, MPI_COMM_WORLD);
	}
	else
	{
		int x;
		for(x=0 ; x<length ; x++) fprintf(stderr, "%c", buffer[x]);

		for(x=1 ; x<size_comm ; x++)
		{
			MPI_Recv(&length, 1, MPI_LONG, x, x, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			char *_data = malloc(length);
			MPI_Recv(_data, length, MPI_CHAR, x, x, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			int y;
			for(y=0 ; y<length ; y++) fprintf(stderr, "%c", _data[y]);
			free(_data);
		}
	}
	free(buffer);
}

static void create_file()
{
	char *tmp = "starpu_XXXXXX";
	strcpy(_f_buffer_name, tmp);

#ifdef STARPU_HAVE_WINDOWS
	_mktemp(_f_buffer_name);
#else
	int id = mkstemp(_f_buffer_name);
	STARPU_ASSERT_MSG(id >= 0, "Error when creating temp file");
#endif
	_f_buffer = fopen(_f_buffer_name, "w");
}

#define FPRINTF_BUFFER_INIT() do { create_file(); } while(0)
#define FPRINTF_BUFFER(fmt, ...) do { if (!silent) fprintf(_f_buffer, fmt, ## __VA_ARGS__); } while(0)
#define FPRINTF_BUFFER_DUMP() do { dump(); } while(0)

struct type_function
{
	void (*init_func)(void *);
	void (*next_func)(void *);
	void (*empty_func)(void *);
	int (*compare_func)(void *buffer, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn, void *value);
	void (*print_func)(FILE *f, void *buffer, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn);
	void (*print_val_func)(FILE *f, void *value);
	void (*set_func)(void *buffer, void *value, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn);
	size_t elem_size;
	void *value;
	void *empty_value;
};

void init_func_char(void *x)
{
	char *_x = (char *)x;
	*_x = 'a';
}

void next_func_char(void *x)
{
	char *_x = (char *)x;
	*_x = *_x + 1;
	if (*_x > 'z')
		init_func_char(x);
}

void set_func_char(void *buffer, void *value, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn)
{
	char *_buffer = (char *)buffer;
	char *_value = (char *)value;
	//FPRINTF_BUFFER("setting %zu %zu %zu %zu %zu at pos %zu to %c\n", x, y, z, t, n, n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x, *_value);
	_buffer[n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x] = *_value;
}

void empty_func_char(void *x)
{
	char *_x = (char *)x;
	*_x = 0;
}

int compare_func_char(void *buffer, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn, void *value)
{
	char *_buffer = (char *)buffer;
	char *_value = (char *)value;
	return (_buffer[n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x] == *_value);
}

void print_func_char(FILE *f, void *buffer, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn)
{
	char *_buffer = (char *)buffer;
	fprintf(f, "'%c' ", _buffer[n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x]);
}

void print_val_func_char(FILE *f, void *value)
{
	char *_value = (char *)value;
	fprintf(f, "'%c' ", *_value);
}

void init_func_int(void *x)
{
	int *_x = (int *)x;
	*_x = 1;
}

void next_func_int(void *x)
{
	int *_x = (int *)x;
	*_x = *_x + 1;
	if (*_x > 1000)
		init_func_int(x);
}

void set_func_int(void *buffer, void *value, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn)
{
	int *_buffer = (int *)buffer;
	int *_value = (int *)value;
	//	FPRINTF_BUFFER("setting %zu %zu %zu at pos %zu to %d\n", x, y, z, t, n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x, *_value);
	_buffer[n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x] = *_value;
}

void empty_func_int(void *x)
{
	int *_x = (int *)x;
	*_x = 0;
}

int compare_func_int(void *buffer, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn, void *value)
{
	int *_buffer = (int *)buffer;
	int *_value = (int *)value;
	return (_buffer[n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x] == *_value);
}

void print_func_int(FILE *f, void *buffer, size_t x, size_t y, size_t z, size_t t, size_t n, size_t ldy, size_t ldz, size_t ldt, size_t ldn)
{
	int *_buffer = (int *)buffer;
	fprintf(f, "%4d ", _buffer[n*ldn*ldt*ldz*ldy+t*ldt*ldz*ldy+z*ldz*ldy+y*ldy+x]);
}

void print_val_func_int(FILE *f, void *value)
{
	int *_value = (int *)value;
	fprintf(f, "'%4d' ", *_value);
}

struct type_function funcs_int =
{
	.init_func = init_func_int,
	.next_func = next_func_int,
	.empty_func = empty_func_int,
	.compare_func = compare_func_int,
	.print_func = print_func_int,
	.print_val_func = print_val_func_int,
	.set_func = set_func_int,
	.elem_size = sizeof(int)
};

struct type_function funcs_char =
{
	.init_func = init_func_char,
	.next_func = next_func_char,
	.empty_func = empty_func_char,
	.compare_func = compare_func_char,
	.print_func = print_func_char,
	.set_func = set_func_char,
	.elem_size = sizeof(char)
};

void print_buffer(char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	if (silent || !display) return;

	size_t n;
	for(n=0 ; n<nn ; n++)
	{
		size_t t;
		for(t=0 ; t<nt ; t++)
		{
			size_t z;
			for(z=0 ; z<nz ; z++)
			{
				size_t y;
				for(y=0 ; y<ny ; y++)
				{
					if (y > 5 && y < ny-5)
					{
						if (y == 6)
							FPRINTF_BUFFER("...\n");
					}
					else
					{
						size_t x;
						for(x = 0; x < nx; x++)
						{
							if (x > 10 && x < nx-10)
							{
								if (x == 11)
									FPRINTF_BUFFER(" ... ");
							}
							else
							{
								funcs.print_func(_f_buffer, buffer, x, y, z, t, n, ldy, ldz, ldt, ldn);
							}
						}
						FPRINTF_BUFFER("\n");
					}
				}
				FPRINTF_BUFFER("\n");
			}
			FPRINTF_BUFFER("\n");
		}
	}
}

void init_buffer(char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, size_t buffer_size, struct type_function funcs)
{
	if (rank_comm == 0)
	{
		funcs.init_func(funcs.value);

		size_t n;
		for(n=0 ; n<nn ; n++)
		{
			size_t t;
			for(t=0 ; t<ldn ; t++)
			{
				size_t z;
				for(z=0 ; z<ldt ; z++)
				{
					size_t y;
					for(y=0 ; y<ldz ; y++)
					{
						size_t x;
						for(x=0 ; x<ldy ; x++)
						{
							if (t < nt && z < nz && y < ny && x < nx)
							{
								funcs.set_func(buffer, funcs.value, x, y, z, t, n, ldy, ldz, ldt, ldn);
							}
							funcs.next_func(funcs.value);
						}
					}
				}
			}
		}

		//print_buffer(buffer, nx, ny, nz, nt, nn, ldy, ldz, ldt, ldn, funcs);
		print_buffer(buffer, ldy, ldz, ldt, ldn, nn, ldy, ldz, ldt, ldn, funcs);
	}
}

int check_buffer(char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	if (rank_comm == 1)
	{
		//print_buffer(buffer, nx, ny, nz, nt, nn, ldy, ldz, ldt, ldn, funcs);
		print_buffer(buffer, ldy, ldz, ldt, ldn, nn, ldy, ldz, ldt, ldn, funcs);

		if (!check) return 0;

		funcs.init_func(funcs.value);
		funcs.empty_func(funcs.empty_value);

		size_t n;
		for(n=0 ; n<nn ; n++)
		{
			size_t t;
			for(t=0 ; t<ldn ; t++)
			{
				size_t z;
				for(z=0 ; z<ldt ; z++)
				{
					size_t y;
					for(y=0 ; y<ldz ; y++)
					{
						size_t x;
						for(x = 0; x < ldy; x++)
						{
							if (t < nt && z < nz && y < ny && x < nx)
							{
								int ret = funcs.compare_func(buffer, x, y, z, t, n, ldy, ldz, ldt, ldn, funcs.value);
								if (ret == 0)
								{
									if (!silent)
									{
										fprintf(_f_buffer, "[starpu][%s][assert failure] [rank_comm %d] Expected value [%zu,%zu,%zu,%zu,%zu] ", __starpu_func__, rank_comm, x, y, z, t, n);
										//funcs.print_val_func(_f_buffer, funcs.value);
										fprintf(_f_buffer, "is not received value ");
										funcs.print_func(_f_buffer, buffer, x, y, z, t, n, ldy, ldz, ldt, ldn);
										fprintf(_f_buffer, "\n");
									}
									return -1;
								}
							}
							else
							{
								int ret = funcs.compare_func(buffer, x, y, z, t, n, ldy, ldz, ldt, ldn, funcs.empty_value);
								if (ret == 0)
								{
									if (!silent)
									{
										fprintf(_f_buffer, "[starpu][%s][assert failure] [rank_comm %d] Value [%zu,%zu,%zu,%zu,%zu] should not have been updated to \n", __starpu_func__, rank_comm, x, y, z, t, n);
										funcs.print_func(_f_buffer, buffer, x, y, z, t, n, ldy, ldz, ldt, ldn);
										fprintf(_f_buffer, "\n");
									}
									return -1;
								}
							}
							funcs.next_func(funcs.value);
						}
					}
				}
			}
		}
	}
	return 0;
}

struct data_function
{
	void (*data_register_func)(starpu_data_handle_t *data_handle, char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs);
	size_t (*data_get_size_func)(size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs);
	char *data_name;
};

void vector_data_register(starpu_data_handle_t *data_handle, char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	(void)ny;
	(void)nz;
	(void)ldy;
	(void)ldz;
	starpu_vector_data_register(data_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, nx, funcs.elem_size);
}

size_t vector_data_get_size(size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	return nx*funcs.elem_size;
}

struct data_function funcs_vector =
{
	.data_register_func = vector_data_register,
	.data_get_size_func = vector_data_get_size,
	.data_name = "vector"
};

void matrix_data_register(starpu_data_handle_t *data_handle, char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	(void)nz;
	(void)ldz;
	starpu_matrix_data_register(data_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, ldy, nx, ny, funcs.elem_size);
}

size_t matrix_data_get_size(size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	return ldy*ny*funcs.elem_size;
}

struct data_function funcs_matrix =
{
	.data_register_func = matrix_data_register,
	.data_get_size_func = matrix_data_get_size,
	.data_name = "matrix"
};

void block_data_register(starpu_data_handle_t *data_handle, char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	starpu_block_data_register(data_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, ldy, ldz*ldy, nx, ny, nz, funcs.elem_size);
}

size_t block_data_get_size(size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	return ldz*ldy*nz*funcs.elem_size;
}

struct data_function funcs_block =
{
	.data_register_func = block_data_register,
	.data_get_size_func = block_data_get_size,
	.data_name = "block"
};

void tensor_data_register(starpu_data_handle_t *data_handle, char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	starpu_tensor_data_register(data_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, ldy, ldz*ldy, ldz*ldy*ldt, nx, ny, nz, nt, funcs.elem_size);
}

size_t tensor_data_get_size(size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	return ldt*ldz*ldy*nt*funcs.elem_size;
}

struct data_function funcs_tensor =
{
	.data_register_func = tensor_data_register,
	.data_get_size_func = tensor_data_get_size,
	.data_name = "tensor"
};

void ndim_data_register(starpu_data_handle_t *data_handle, char *buffer, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	size_t nns[5] = {nx, ny, nz, nt, nn};
	size_t lds[5] = {1, ldy, ldy*ldz, ldy*ldz*ldt, ldy*ldz*ldt*ldn};
	starpu_ndim_data_register(data_handle, STARPU_MAIN_RAM, (uintptr_t)buffer, lds, nns, 5, funcs.elem_size);
}

size_t ndim_data_get_size(size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn, struct type_function funcs)
{
	return ldn*ldt*ldz*ldy*nn*funcs.elem_size;
}

struct data_function funcs_ndim =
{
	.data_register_func = ndim_data_register,
	.data_get_size_func = ndim_data_get_size,
	.data_name = "ndim"
};

int check_dataset(struct data_function data_funcs, struct type_function type_funcs, size_t nx, size_t ny, size_t nz, size_t nt, size_t nn, size_t ldy, size_t ldz, size_t ldt, size_t ldn)
{
	char *buffer;
	starpu_data_handle_t data_handle=NULL;
	int ret;
	size_t size;

	size = data_funcs.data_get_size_func(nx, ny, nz, nt, nn, ldy, ldz, ldt, ldn, type_funcs);
	FPRINTF_BUFFER("\nCheck with %s [nx=%zu(ldy=%zu),ny=%zu(ldz=%zu),nz=%zu(ldt=%zu),nt=%zu(ldn=%zu),nn=%zu] elements of size %zu ... --> whole size %zu (INT_MAX %d)\n", data_funcs.data_name, nx, ldy, ny, ldz, nz, ldt, nt, ldn, nn, type_funcs.elem_size, size, INT_MAX);

	buffer = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
	if (buffer == MAP_FAILED)
	{
		FPRINTF_BUFFER("cannot allocate memory for %zu bytes ...\n", size);
		perror("mmap");
		return -1;
	}
	memset(buffer, 0, size);
	type_funcs.value = calloc(1, type_funcs.elem_size);
	type_funcs.empty_value = calloc(1, type_funcs.elem_size);
	init_buffer(buffer, nx, ny, nz, nt, nn, ldy, ldz, ldt, ldn, size, type_funcs);

	data_funcs.data_register_func(&data_handle, buffer, nx, ny, nz, nt, nn, ldy, ldz, ldt, ldn, type_funcs);

	ret = 0;
	if (rank_comm == 0)
	{
		ret = starpu_mpi_send(data_handle, 1, 42, MPI_COMM_WORLD);
	}
	else if (rank_comm == 1)
	{
		ret = starpu_mpi_recv(data_handle, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_send/recv");

	starpu_data_unregister(data_handle);
	ret = check_buffer(buffer, nx, ny, nz, nt, nn, ldy, ldz, ldt, ldn, type_funcs);
	munmap(buffer, size);
	free(type_funcs.value);
	free(type_funcs.empty_value);
	FPRINTF_BUFFER("Check with %s [nx=%zu(ldy=%zu),ny=%zu(ldz=%zu),nz=%zu(ldt=%zu),nt=%zu(ldn=%zu),nn=%zu] elements of size %zu ... DONE with %s\n", data_funcs.data_name, nx, ldy, ny, ldz, nz, ldt, nt, ldn, nn, type_funcs.elem_size, ret==0?"SUCCESS":"FAILURE");
	return ret;
}

int _starpu_test_datatype();

int main(int argc, char **argv)
{
	int ret;
	struct starpu_conf conf;
	int mpi_init, i;

	for(i=1 ; i<argc ; i++)
	{
		if (strcmp(argv[i], "--no-check") == 0) check = 0;
		if (strcmp(argv[i], "--no-display") == 0) display = 0;

		if (strcmp(argv[i], "--no-small") == 0) test_small = 0;
		if (strcmp(argv[i], "--no-large") == 0) test_large = 0;

		if (strcmp(argv[i], "--no-vector") == 0) tests_data[vector] = 0;
		if (strcmp(argv[i], "--no-matrix") == 0) tests_data[matrix] = 0;
		if (strcmp(argv[i], "--no-block") == 0)  tests_data[block] = 0;
		if (strcmp(argv[i], "--no-tensor") == 0) tests_data[tensor] = 0;
		if (strcmp(argv[i], "--no-ndim") == 0)   tests_data[ndim] = 0;

		if (strcmp(argv[i], "--only-vector") == 0) { memset(tests_data, 0, nb_tests*sizeof(int)); tests_data[vector] = 1; }
		if (strcmp(argv[i], "--only-matrix") == 0) { memset(tests_data, 0, nb_tests*sizeof(int)); tests_data[matrix] = 1; }
		if (strcmp(argv[i], "--only-block") == 0)  { memset(tests_data, 0, nb_tests*sizeof(int)); tests_data[block] = 1; }
		if (strcmp(argv[i], "--only-tensor") == 0) { memset(tests_data, 0, nb_tests*sizeof(int)); tests_data[tensor] = 1; }
		if (strcmp(argv[i], "--only-ndim") == 0)   { memset(tests_data, 0, nb_tests*sizeof(int)); tests_data[ndim] = 1; }
	}
	if (getenv("STARPU_SSILENT"))
		silent = 1;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_sc = -1;
	conf.ntcpip_sc = -1;

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, &conf);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank_comm);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &size_comm);

	if (size_comm < 2)
	{
		if (rank_comm == 0)
			FPRINTF(stderr, "We need at least 2 processes (size %d).\n", size_comm);

		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return 0;
	}

	FPRINTF_BUFFER_INIT();
	FPRINTF_BUFFER("hello from rank_comm %d\n", rank_comm);

	if (rank_comm > 1)
		goto end;

#ifndef STARPU_HAVE_MPI_TYPE_VECTOR_C
	{
		// As the function MPI_Type_vector_c is not defined,
		// StarPU will use MPI_Type_create_struct. Before
		// running the test, let's check if this function can
		// properly handle large types
		ret = _starpu_test_datatype();
		if (ret == 0)
		{
			FPRINTF(stderr, "Function MPI_Type_create_struct fails with large types.\n");
			fclose(_f_buffer);
			starpu_mpi_shutdown();
			if (!mpi_init)
				MPI_Finalize();
			return 0;
		}
	}
#endif

#if defined(STARPU_MPI_MINIMAL_TESTS) || defined(STARPU_QUICK_CHECK)
	ret = check_dataset(funcs_vector, funcs_char, (size_t)INT_MAX+12, 1, 1, 1, 1, (size_t)INT_MAX+12, 1, 1, 1);
	goto end;
#endif

	if (tests_data[vector])
	{
		if (test_small)
		{
			ret = check_dataset(funcs_vector, funcs_char, 26, 1, 1, 1, 1, 26, 1, 1, 1);
			if (ret == -1) goto end;
		}
		if (test_large)
		{
			ret = check_dataset(funcs_vector, funcs_char, (size_t)INT_MAX, 1, 1, 1, 1, (size_t)INT_MAX, 1, 1, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_vector, funcs_int, (size_t)INT_MAX, 1, 1, 1, 1, (size_t)INT_MAX, 1, 1, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_vector, funcs_char, (size_t)INT_MAX+12, 1, 1, 1, 1, (size_t)INT_MAX+12, 1, 1, 1);
			if (ret == -1) goto end;
		}
	}

	if (tests_data[matrix])
	{
		if (test_small)
		{
			ret = check_dataset(funcs_matrix, funcs_char, 4, 3, 1, 1, 1, 4, 3, 1, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_matrix, funcs_char, 3, 5, 1, 1, 1, 10, 5, 1, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_matrix, funcs_int, 3, 5, 1, 1, 1, 10, 5, 1, 1);
			if (ret == -1) goto end;
		}
		if (test_large)
		{
			ret = check_dataset(funcs_matrix, funcs_char, (size_t)INT_MAX, 1, 1, 1, 1, (size_t)INT_MAX, 1, 1, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_matrix, funcs_int, (size_t)INT_MAX+100, 1, 1, 1, 1, (size_t)INT_MAX+100, 1, 1, 1);
			if (ret == -1) goto end;
		}
	}

	if (tests_data[block])
	{
		if (test_small)
		{
			ret = check_dataset(funcs_block, funcs_int, 6, 2, 4, 1, 1, 6, 2, 4, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_block, funcs_char, 5, 2, 7, 1, 1, 6, 3, 7, 1);
			if (ret == -1) goto end;
		}
		if (test_large)
		{
			ret = check_dataset(funcs_block, funcs_char, (size_t)INT_MAX, 1, 1, 1, 1, (size_t)INT_MAX, 1, 1, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_block, funcs_char, (size_t)INT_MAX+10, 1, 1, 1, 1, (size_t)INT_MAX+10, 1, 1, 1);
			if (ret == -1) goto end;
		}
	}

	if (tests_data[tensor])
	{
		if (test_small)
		{
			ret = check_dataset(funcs_tensor, funcs_int, 6, 4, 2, 4, 1, 6, 4, 2, 4);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_tensor, funcs_char, 6, 4, 2, 4, 1, 6, 5, 3, 4);
			if (ret == -1) goto end;
		}
		if (test_large)
		{
			ret = check_dataset(funcs_tensor, funcs_char, ((size_t)INT_MAX+10)/20, 2, 4, 3, 1, ((size_t)INT_MAX+10)/20, 2, 4, 1);
			if (ret == -1) goto end;
		}
	}

	if (tests_data[ndim])
	{
		if (test_small)
		{
			ret = check_dataset(funcs_ndim, funcs_char, 3, 2, 3, 1, 2, 3, 2, 3, 1);
			if (ret == -1) goto end;
			ret = check_dataset(funcs_ndim, funcs_char, 2, 2, 3, 1, 2, 3, 2, 3, 1);
			if (ret == -1) goto end;
		}
		if (test_large)
		{
			ret = check_dataset(funcs_ndim, funcs_char, ((size_t)INT_MAX+10)/20, 2, 4, 3, 2, ((size_t)INT_MAX+10)/20, 2, 4, 3);
			if (ret == -1) goto end;
		}
	}

end:
	FPRINTF_BUFFER_DUMP();
	starpu_mpi_shutdown();

 enodev:
	if (!mpi_init)
		MPI_Finalize();

	return rank_comm == 0 ? ret ==-ENODEV ? 0 : ret : 0;
}
