/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2017       Erwan Leria
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

/*
 * Simple parallel GEMM implementation: partition the output matrix in the two
 * dimensions, and the input matrices in the corresponding dimension, and
 * perform the output computations in parallel.
 */
#ifndef TYPE
#error "Do not compile xgemm.c directly, compile sgemm.c or dgemm.c"
#endif

//To randomize the order of tasks
#include <time.h>
#include <stdlib.h>
#define RANDOM_TASK_ORDER
#define RECURSIVE_MATRIX_LAYOUT
#define RANDOM_DATA_ACCESS
#define SEED
#include <starpu_data_maxime.h>

#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <starpu.h>
#include <starpu_fxt.h>

#include <common/blas.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <starpu_cublas_v2.h>
static const TYPE p1 = 1.0;
static const TYPE m1 = -1.0;
static const TYPE v0 = 0.0;
#endif

static unsigned niter = 10;
static unsigned nsleeps = 1;
static unsigned nslicesx = 4;
static unsigned nslicesy = 4;
static unsigned nslicesz = 4;
#if defined(STARPU_QUICK_CHECK) && !defined(STARPU_SIMGRID)
static unsigned xdim = 256;
static unsigned ydim = 256;
static unsigned zdim = 64;
#else
static unsigned xdim = 960*4;
static unsigned ydim = 960*4;
static unsigned zdim = 960*4;
#endif
static unsigned check = 0;
static unsigned bound = 0;
static unsigned print_hostname = 0;
static unsigned tiled = 0;

static TYPE *A, *B, *C;
static starpu_data_handle_t A_handle, B_handle, C_handle;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); fflush(stdout); }} while(0)

static int check_output(void)
{
	/* compute C = C - AB */
	CPU_GEMM("N", "N", ydim, xdim, zdim, (TYPE)-1.0f, A, ydim, B, zdim, (TYPE)1.0f, C, ydim);

	/* make sure C = 0 */
	TYPE err;
	err = CPU_ASUM(xdim*ydim, C, 1);

	if (err < EPSILON*xdim*ydim*zdim)
	{
		FPRINTF(stderr, "Results are OK\n");
		return 0;
	}
	else
	{
		int max;
		max = CPU_IAMAX(xdim*ydim, C, 1);

		FPRINTF(stderr, "There were errors ... err = %f\n", err);
		FPRINTF(stderr, "Max error : %e\n", C[max]);
		return 1;
	}
}

static void init_problem_data(void)
{
#ifndef STARPU_SIMGRID
	unsigned i,j;
#endif

	starpu_malloc_flags((void **)&A, zdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&B, xdim*zdim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_malloc_flags((void **)&C, xdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

#ifndef STARPU_SIMGRID
	/* fill the A and B matrices */
	for (j=0; j < ydim; j++)
	{
		for (i=0; i < zdim; i++)
		{
			A[j+i*ydim] = (TYPE)(starpu_drand48());
		}
	}

	for (j=0; j < zdim; j++)
	{
		for (i=0; i < xdim; i++)
		{
			B[j+i*zdim] = (TYPE)(starpu_drand48());
		}
	}

	for (j=0; j < ydim; j++)
	{
		for (i=0; i < xdim; i++)
		{
			C[j+i*ydim] = (TYPE)(0);
		}
	}
#endif
}

static void partition_mult_data(void)
{
	unsigned x, y, z;

	starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A,
		ydim, ydim, zdim, sizeof(TYPE));
	starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B,
		zdim, zdim, xdim, sizeof(TYPE));
	starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C,
		ydim, ydim, xdim, sizeof(TYPE));

	struct starpu_data_filter vert;
	memset(&vert, 0, sizeof(vert));
	vert.filter_func = starpu_matrix_filter_vertical_block;
	vert.nchildren = nslicesx;

	struct starpu_data_filter horiz;
	memset(&horiz, 0, sizeof(horiz));
	horiz.filter_func = starpu_matrix_filter_block;
	horiz.nchildren = nslicesy;

	if (tiled)
	{
		struct starpu_data_filter vertA;
		memset(&vertA, 0, sizeof(vertA));
		vertA.filter_func = starpu_matrix_filter_vertical_block;
		vertA.nchildren = nslicesz;

		struct starpu_data_filter horizB;
		memset(&horizB, 0, sizeof(horizB));
		horizB.filter_func = starpu_matrix_filter_block;
		horizB.nchildren = nslicesz;

		starpu_data_map_filters(A_handle, 2, &vertA, &horiz);
		starpu_data_map_filters(B_handle, 2, &vert, &horizB);
		starpu_data_map_filters(C_handle, 2, &vert, &horiz);

		for (y = 0; y < nslicesy; y++)
		for (z = 0; z < nslicesz; z++)
			starpu_data_set_coordinates(starpu_data_get_sub_data(A_handle, 2, z, y), 2, z, y);

		for (x = 0; x < nslicesx; x++)
		for (z = 0; z < nslicesz; z++)
			starpu_data_set_coordinates(starpu_data_get_sub_data(B_handle, 2, x, z), 2, x, z);
	}
	else
	{
		starpu_data_partition(B_handle, &vert);
		starpu_data_partition(A_handle, &horiz);

		starpu_data_map_filters(C_handle, 2, &vert, &horiz);
	}

	for (x = 0; x < nslicesx; x++)
	for (y = 0; y < nslicesy; y++)
		starpu_data_set_coordinates(starpu_data_get_sub_data(C_handle, 2, x, y), 2, x, y);
}

#ifdef STARPU_USE_CUDA
static void cublas_mult(void *descr[], void *arg, const TYPE *beta)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *subC = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned nxC = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned nyC = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldC = STARPU_MATRIX_GET_LD(descr[2]);

	cudaStream_t stream = starpu_cuda_get_local_stream();

	if (nxC == ldC)
		cudaMemsetAsync(subC, 0, sizeof(*subC) * nxC * nyC, stream);
	else
	{
		unsigned i;
		for (i = 0; i < nyC; i++)
			cudaMemsetAsync(subC + i*ldC, 0, sizeof(*subC) * nxC, stream);
	}

	cublasStatus_t status = CUBLAS_GEMM(starpu_cublas_get_local_handle(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			nxC, nyC, nyA,
			&p1, subA, ldA, subB, ldB,
			beta, subC, ldC);
	if (status != CUBLAS_STATUS_SUCCESS)
		STARPU_CUBLAS_REPORT_ERROR(status);
}

static void cublas_gemm0(void *descr[], void *arg)
{
	cublas_mult(descr, arg, &v0);
}

static void cublas_gemm(void *descr[], void *arg)
{
	cublas_mult(descr, arg, &p1);
}
#endif

void cpu_mult(void *descr[], void *arg, TYPE beta)
{
	(void)arg;
	TYPE *subA = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
	TYPE *subB = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
	TYPE *subC = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);

	unsigned nxC = STARPU_MATRIX_GET_NX(descr[2]);
	unsigned nyC = STARPU_MATRIX_GET_NY(descr[2]);
	unsigned nyA = STARPU_MATRIX_GET_NY(descr[0]);

	unsigned ldA = STARPU_MATRIX_GET_LD(descr[0]);
	unsigned ldB = STARPU_MATRIX_GET_LD(descr[1]);
	unsigned ldC = STARPU_MATRIX_GET_LD(descr[2]);

	int worker_size = starpu_combined_worker_get_size();

	if (nxC == ldC)
		memset(subC, 0, sizeof(*subC) * nxC * nyC);
	else
	{
		unsigned i;
		for (i = 0; i < nyC; i++)
			memset(subC + i*ldC, 0, sizeof(*subC) * nxC);
	}

	if (worker_size == 1)
	{
		/* Sequential CPU task */
		CPU_GEMM("N", "N", nxC, nyC, nyA, (TYPE)1.0, subA, ldA, subB, ldB, beta, subC, ldC);
	}
	else
	{
		/* Parallel CPU task */
		unsigned rank = starpu_combined_worker_get_rank();

		unsigned block_size = (nyC + worker_size - 1)/worker_size;
		unsigned new_nyC = STARPU_MIN(nyC, block_size*(rank+1)) - block_size*rank;

		STARPU_ASSERT(nyC == STARPU_MATRIX_GET_NY(descr[1]));

		TYPE *new_subB = &subB[block_size*rank];
		TYPE *new_subC = &subC[block_size*rank];

		CPU_GEMM("N", "N", nxC, new_nyC, nyA, (TYPE)1.0, subA, ldA, new_subB, ldB, beta, new_subC, ldC);
	}
}

void cpu_gemm0(void *descr[], void *arg)
{
	cpu_mult(descr, arg, 0.);
}

void cpu_gemm(void *descr[], void *arg)
{
	cpu_mult(descr, arg, 1.);
}

static struct starpu_perfmodel starpu_gemm_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = STARPU_GEMM_STR(gemm)
};

static struct starpu_codelet cl_gemm0 =
{
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_gemm0},
	.cpu_funcs_name = {"cpu_gemm0"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cublas_gemm0},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	//~ .modes = {STARPU_R, STARPU_R, STARPU_RW},
	//~ .modes = {STARPU_R, STARPU_R, STARPU_R},
	.model = &starpu_gemm_model
};

static struct starpu_codelet cl_gemm =
{
	.type = STARPU_SEQ, /* changed to STARPU_SPMD if -spmd is passed */
	.max_parallelism = INT_MAX,
	.cpu_funcs = {cpu_gemm},
	.cpu_funcs_name = {"cpu_gemm"},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cublas_gemm},
#elif defined(STARPU_SIMGRID)
	.cuda_funcs = {(void*)1},
#endif
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	//~ .modes = {STARPU_R, STARPU_R, STARPU_RW},
	.modes = {STARPU_R, STARPU_R, STARPU_R},
	.model = &starpu_gemm_model
};

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-3d") == 0)
		{
			tiled = 1;
		}

		else if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			nslicesy = nslicesx;
			nslicesz = nslicesx;
		}

		else if (strcmp(argv[i], "-nblocksx") == 0)
		{
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nblocksy") == 0)
		{
			char *argptr;
			nslicesy = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nblocksz") == 0)
		{
			char *argptr;
			nslicesz = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-x") == 0)
		{
			char *argptr;
			xdim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-xy") == 0)
		{
			char *argptr;
			xdim = ydim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-y") == 0)
		{
			char *argptr;
			ydim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-z") == 0)
		{
			char *argptr;
			zdim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-size") == 0)
		{
			char *argptr;
			xdim = ydim = zdim = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-iter") == 0)
		{
			char *argptr;
			niter = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nsleeps") == 0)
		{
			char *argptr;
			nsleeps = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-bound") == 0)
		{
			bound = 1;
		}

		else if (strcmp(argv[i], "-hostname") == 0)
		{
			print_hostname = 1;
		}

		else if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		else if (strcmp(argv[i], "-spmd") == 0)
		{
			cl_gemm0.type = STARPU_SPMD;
		}

		else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			fprintf(stderr,"Usage: %s [-3d] [-nblocks n] [-nblocksx x] [-nblocksy y] [-nblocksz z] [-x x] [-y y] [-xy n] [-z z] [-size size] [-iter iter] [-bound] [-check] [-spmd] [-hostname] [-nsleeps nsleeps]\n", argv[0]);
			if (tiled)
				fprintf(stderr,"Currently selected: %ux%u * %ux%u and %ux%ux%u blocks, %u iterations, %u sleeps\n", zdim, ydim, xdim, zdim, nslicesx, nslicesy, nslicesz, niter, nsleeps);
			else
				fprintf(stderr,"Currently selected: %ux%u * %ux%u and %ux%u blocks, %u iterations, %u sleeps\n", zdim, ydim, xdim, zdim, nslicesx, nslicesy, niter, nsleeps);
			exit(EXIT_SUCCESS);
		}
		else
		{
			fprintf(stderr,"Unrecognized option %s\n", argv[i]);
			exit(EXIT_FAILURE);
		}
	}
}

/* Don't do this at home, kids, this is really dumb!  */
starpu_data_handle_t dumb_victim_selector(unsigned node)
{
	static unsigned next_evicted; // index of next data to evict, to avoid getting stuck. Yes this is awful.
	starpu_data_handle_t handle;
	unsigned x, y, z, index = 0;

	if (tiled) {
		if (next_evicted == nslicesy*nslicesz + nslicesx+nslicesz + nslicesx*nslicesy)
			next_evicted = 0;

		for (y = 0; y < nslicesy; y++)
		for (z = 0; z < nslicesz; z++)
		{
			if (index++ < next_evicted)
				continue;
			handle = starpu_data_get_sub_data(A_handle, 2, z, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (z = 0; z < nslicesz; z++)
		{
			if (index++ < next_evicted)
				continue;
			handle = starpu_data_get_sub_data(B_handle, 2, x, z);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;
			handle = starpu_data_get_sub_data(C_handle, 2, x, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}
	}
	else
	{
		if (next_evicted == 3*nslicesx*nslicesy)
			next_evicted = 0;

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;

			handle = starpu_data_get_sub_data(A_handle, 1, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;

			handle = starpu_data_get_sub_data(B_handle, 1, x);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;

			handle = starpu_data_get_sub_data(C_handle, 2, x, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}
	}

	/* Uh :/ */
	//~ fprintf(stderr,"uh, no evictable data\n");
	next_evicted = 0;
	return NULL;

done:
	next_evicted = index;
	//~ fprintf(stderr,"evicting %p\n", handle);
	return handle;
}

//~ starpu_data_is_on_node : savoir si une donnée est bien sur la mémoire
/* une idée pourrait être dans hfp simuler la mémoire et voir quand elle sera pleine et aura besoin d'évincer,
 * et la j'appelle cette fonction
 * pre_exec_hook pour connaitre la tâche en cours
/* Evicting data used in the longest time  */
starpu_data_handle_t belady_victim_selector(unsigned node)
{
	//~ printf("Début belady_victim_selector\n");
	static unsigned next_evicted; // index of next data to evict, to avoid getting stuck. Yes this is awful.
	starpu_data_handle_t handle;
	unsigned x, y, z, index;
	int nb_data_next_task = 0; int i = 0; int j = 0;
	int nb_data_on_node = 0; /* Number of data loaded on memory. Needed to init the tab containing data on node */
	int * a_ne_pas_evincer;
	if (task_currently_treated != NULL) {
		for (i = 0; i < nb_different_data; i++) {
			if (starpu_data_is_on_node(all_data_needed[i], node)) {
				nb_data_on_node++;
			}
		}
		printf("Il y a %d data on node :\n",nb_data_on_node);
		starpu_data_handle_t * data_on_node = malloc(nb_data_on_node*sizeof(all_data_needed[0]));
		for (i = 0; i < nb_different_data; i++) {
			if (starpu_data_is_on_node(all_data_needed[i], node)) {
				data_on_node[j] = all_data_needed[i];
				printf("%p\n",data_on_node[j]);
				j++;
			}
		}
		//~ printf("La tâche en cours est %p, index numéro %d, position %d dans le tableau d'ordre des données\n",task_currently_treated, index_task_currently_treated, task_position_in_data_use_order[index_task_currently_treated]);
		
		//~ printf("Mes données étaient :\n");
		//~ printf("%p %p %p\n",data_use_order[task_position_in_data_use_order[index_task_currently_treated]-3],data_use_order[task_position_in_data_use_order[index_task_currently_treated]-2],data_use_order[task_position_in_data_use_order[index_task_currently_treated]-1]);
		//~ printf("Les données suivantes sont\n");
		//~ printf("%p %p %p\n",data_use_order[task_position_in_data_use_order[index_task_currently_treated]],data_use_order[task_position_in_data_use_order[index_task_currently_treated]+1],data_use_order[task_position_in_data_use_order[index_task_currently_treated]+2]);
		
		nb_data_next_task = task_position_in_data_use_order[index_task_currently_treated + 1] - task_position_in_data_use_order[index_task_currently_treated];
		printf("Nb data next task : %d\n",nb_data_next_task);
		int a_ne_pas_evincer[nb_data_next_task];
		//Remplissage des données à ne pas évincer car elles sont utilisé par la prochaine tâche
		for (i = 0; i < nb_data_next_task; i++) { a_ne_pas_evincer[i] = 0; }
		for (i = 0; i < nb_data_next_task; i++) { 
			for (j = 0; j < nb_data_on_node; j++) { 
				if(data_use_order[task_position_in_data_use_order[index_task_currently_treated + i]] == data_on_node[j]) { 
					a_ne_pas_evincer[i] = 1; 
				} 
			} 
		}
		for (i = 0; i < nb_data_next_task; i++) {	
			/* On regarde si la donnée est pas déjà sur M par hasard */
			if (starpu_data_is_on_node(data_use_order[task_position_in_data_use_order[index_task_currently_treated + i]], node)) {
				printf("La donnée %p est déjà sur M\n",data_use_order[task_position_in_data_use_order[index_task_currently_treated + i]]);
				break;
			}
			else {
				for (j = 1; j < total_nb_data; j++) {
					if(starpu_data_is_on_node(data_use_order[total_nb_data - j], node) && 
					a_ne_pas_evincer[i] != 1) {
						//evincer
						printf("On évince :\n");
						break;
					}
				}
			}
		}
	}
	
	//Savoir a quelle tache on est donne ou on est dans la liste des taches ordonné.
	//De la on peut deduire la donnée utilisé dans le + longtemps qui est sur node
	//Mais attention a ne pas evincer une donnée qui sera utilisé par la tache suivante
	//Pour savoir le nb de data que utilise la prochaine simplement regardé la diff entre
	//task_position_in_data_use_order[index_task_currently_treated] et task_position_in_data_use_order[index_task_currently_treated+1]
	
	if (tiled) {
		if (next_evicted == nslicesy*nslicesz + nslicesx+nslicesz + nslicesx*nslicesy)
			next_evicted = 0;

		for (y = 0; y < nslicesy; y++)
		for (z = 0; z < nslicesz; z++)
		{
			if (index++ < next_evicted)
				continue;
			handle = starpu_data_get_sub_data(A_handle, 2, z, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (z = 0; z < nslicesz; z++)
		{
			if (index++ < next_evicted)
				continue;
			handle = starpu_data_get_sub_data(B_handle, 2, x, z);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;
			handle = starpu_data_get_sub_data(C_handle, 2, x, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}
	}
	else
	{
		if (next_evicted == 3*nslicesx*nslicesy)
			next_evicted = 0;

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;

			handle = starpu_data_get_sub_data(A_handle, 1, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;

			handle = starpu_data_get_sub_data(B_handle, 1, x);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}

		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			if (index++ < next_evicted)
				continue;

			handle = starpu_data_get_sub_data(C_handle, 2, x, y);
			if (starpu_data_is_on_node(handle, node) && starpu_data_can_evict(handle, node))
				goto done;
		}
	}

	/* Uh :/ */
	//~ fprintf(stderr,"uh, no evictable data\n");
	next_evicted = 0;
	return NULL;

done:
	next_evicted = index;
	//~ fprintf(stderr,"evicting %p\n", handle);
	return handle;
}

int main(int argc, char **argv)
{
	//Ajout pour randomiser l'ordre d'arrivé des tâches srandom(starpu_get_env_number_default("SEED",1));
	
	//Ajout pour le Z layout
	int x_z_layout = 0; int i_bis = 0; int x_z_layout_i = 0; int j_bis = 0; int y_z_layout = 0; int y_z_layout_i = 0;
	
	double start, end;
	int ret;

	parse_args(argc, argv);

#ifdef STARPU_QUICK_CHECK
	niter /= 10;
#endif

	starpu_fxt_autostart_profiling(0);
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_cublas_init();

	init_problem_data();
	partition_mult_data();
	
	//Mettre un if belady=1 pour plus tard ici (avec ma fct du coup)
	if (starpu_get_env_number_default("BELADY",0) == 1) { starpu_data_register_victim_selector(belady_victim_selector); }
	//~ starpu_data_register_victim_selector(dumb_victim_selector);

	PRINTF("# ");
	if (print_hostname)
		PRINTF("node\t");
	PRINTF("x\ty\tz\tms\tGFlops");
	if (bound)
		PRINTF("\tTms\tTGFlops\tTims\tTiGFlops");
	PRINTF("\n");

	unsigned sleeps;
	for (sleeps = 0; sleeps < nsleeps; sleeps++)
	{
		if (bound)
			starpu_bound_start(0, 0);

		starpu_fxt_start_profiling();
		start = starpu_timing_now();

		unsigned x, y, z, iter;
		/* Matrice 3D */
		if (tiled)
		{
			for (iter = 0; iter < niter; iter++)
			{
				starpu_pause(); /* To get all tasks at once */
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					starpu_data_handle_t Ctile = starpu_data_get_sub_data(C_handle, 2, x, y);
					//~ starpu_pause();
					for (z = 0; z < nslicesz; z++)
					{
						struct starpu_task *task = starpu_task_create();

						if (z == 0)
							//~ task->cl = &cl_gemm0;
							task->cl = &cl_gemm;
						else
							task->cl = &cl_gemm;

						task->handles[0] = starpu_data_get_sub_data(A_handle, 2, z, y);
						task->handles[1] = starpu_data_get_sub_data(B_handle, 2, x, z);
						task->handles[2] = Ctile;

						task->flops = 2ULL * (xdim/nslicesx) * (ydim/nslicesy) * (zdim/nslicesz);

						ret = starpu_task_submit(task);
						if (ret == -ENODEV)
						{
						     check = 0;
						     ret = 77;
						     goto enodev;
						}
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
					}
					//~ starpu_resume();
					starpu_data_wont_use(Ctile);
				}
				starpu_resume(); /* Because I paused above */
				starpu_task_wait_for_all();
			}
		} else if (starpu_get_env_number_default("RANDOM_TASK_ORDER",0) == 1 && starpu_get_env_number_default("RECURSIVE_MATRIX_LAYOUT",0) == 0 && starpu_get_env_number_default("RANDOM_DATA_ACCESS",0) == 0) {
			//~ printf("random\n");
			//If environment variable RANDOM_TASK_ORDER == 1
			int i = 0; int j = 0; unsigned tab_x[nslicesx][nslicesx]; unsigned tab_y[nslicesy][nslicesy]; int temp = 0; int k = 0; int n = 0;
			for (iter = 0; iter < niter; iter++)
			{
				for (i= 0; i < nslicesx; i++) { for (j = 0; j < nslicesx; j++) { tab_x[i][j] = i; } }
				for (i= 0; i < nslicesy; i++) { for (j = 0; j < nslicesy; j++) { tab_y[i][j] = j; } }

				//Shuffle
				for( n=0; n< nslicesx*nslicesy; n++)
				{	
					k = n;
					k += random() % ((nslicesx*nslicesy) - n);
					temp = tab_x[n%nslicesx][n/nslicesx];
					tab_x[n%nslicesx][n/nslicesx] = tab_x[k%nslicesx][k/nslicesx];
					tab_x[k%nslicesx][k/nslicesx] = temp;	
					temp = tab_y[n%nslicesy][n/nslicesy];
					tab_y[n%nslicesy][n/nslicesy] = tab_y[k%nslicesy][k/nslicesy];
					tab_y[k%nslicesy][k/nslicesy] = temp;
				} 			
				//printf des tableaux
				//~ printf("\n");
				//~ printf("Tableau x : \n");
				//~ for (i= 0; i < nslicesx; i++) { for (j = 0; j < nslicesx; j++) { printf(" %3d ",tab_x[i][j]); } printf("\n"); }
				//~ printf("Tableau y : \n");
				//~ for (i= 0; i < nslicesy; i++) { for (j = 0; j < nslicesy; j++) { printf(" %3d ",tab_y[i][j]); } printf("\n"); }
				
				starpu_pause();
				for (i = 0; i < nslicesx; i++)
				for (j = 0; j < nslicesy; j++)
				{
					struct starpu_task *task = starpu_task_create();

					task->cl = &cl_gemm0;
					
					task->handles[0] = starpu_data_get_sub_data(A_handle, 1, tab_y[i][j]);
					task->handles[1] = starpu_data_get_sub_data(B_handle, 1, tab_x[i][j]);
					task->handles[2] = starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]);
					task->flops = 2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim;

					ret = starpu_task_submit(task);
					if (ret == -ENODEV)
					{
						 ret = 77;
						 goto enodev;
					}
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
					starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]));
				}
				starpu_resume();
				starpu_task_wait_for_all();
			}
			//End If environment variable RANDOM_TASK_ORDER == 1
		}
		else if (starpu_get_env_number_default("RECURSIVE_MATRIX_LAYOUT",0) == 1 && starpu_get_env_number_default("RANDOM_DATA_ACCESS",0) == 0) {
			// RECURSIVE_MATRIX_LAYOUT == 1, cela annule le random=1
			int i = 0; int j = 0; unsigned tab_x[nslicesx][nslicesx]; unsigned tab_y[nslicesy][nslicesy]; int temp = 0; int k = 0; int n = 0;
			for (iter = 0; iter < niter; iter++)
			{
				for (i= 0; i < nslicesx; i++) { for (j = 0; j < nslicesx; j++) { tab_x[i][j] = i; } }
				for (i= 0; i < nslicesy; i++) { for (j = 0; j < nslicesy; j++) { tab_y[i][j] = j; } }
			
				//printf des tableaux
				//~ printf("Au début \n");
				//~ printf("Tableau x : \n");
				//~ for (i= 0; i < nslicesx; i++) { for (j = 0; j < nslicesx; j++) { printf(" %3d ",tab_x[i][j]); } printf("\n"); }
				//~ printf("Tableau y : \n");
				//~ for (i= 0; i < nslicesy; i++) { for (j = 0; j < nslicesy; j++) { printf(" %3d ",tab_y[i][j]); } printf("\n"); }
				
				x_z_layout = 0; x_z_layout_i = 0; i_bis = 0;
				for (i= 0; i < nslicesx; i++) { for (j = 0; j < nslicesx; j++) {
					if (i_bis%2 == 1) { x_z_layout_i = nslicesx/2; }
					if (j >= 4) { x_z_layout = (j/4)*2; }
					tab_x[i][j] = j%2 + x_z_layout + x_z_layout_i;
				} x_z_layout = 0; x_z_layout_i = 0; if (i%2 == 1) { i_bis++; } }
				
				//~ x_z_layout = 0; x_z_layout_i = 0; i_bis = 0;
				//~ for (i= 0; i < nslicesy; i++) { for (j = 0; j < nslicesy; j++) {
					//~ if (i_bis%2 == 1) { x_z_layout_i = nslicesy/2; }
					//~ if (j >= 4) { x_z_layout = (j/4)*2; }
					//~ tab_y[j][i] = j%2 + x_z_layout + x_z_layout_i;
				//~ } x_z_layout = 0; x_z_layout_i = 0; if (i%2 == 1) { i_bis++; } }
				
				y_z_layout_i = 0; i_bis = 0; j_bis = 0; y_z_layout = 0;
				for (i= 0; i < nslicesy; i++) { for (j = 0; j < nslicesy; j++) {
					//~ if (i >= nslicesy/2) { y_z_layout_i = nslicesy/2; }
					if (i >= 4) { y_z_layout_i = 4*(i/4); }
					//~ if (i%2 == 1) { y_z_layout_i += nslicesy/4; }
					if (j_bis%2 == 1) { y_z_layout = 1; }
					if (i%2 == 1) { y_z_layout += 2; }
					tab_y[i][j] = y_z_layout + y_z_layout_i;
					if (j%2 == 1) { j_bis++; }
					y_z_layout = 0;
					y_z_layout_i = 0;
				} y_z_layout = 0;  if (i%2 == 1) { i_bis++; } }
				
		
				
				//printf des tableaux
				//~ printf("A la fin \n");
				//~ printf("Tableau x : \n");
				//~ for (i= 0; i < nslicesx; i++) { for (j = 0; j < nslicesx; j++) { printf(" %3d ",tab_x[i][j]); } printf("\n"); }
				//~ printf("Tableau y : \n");
				//~ for (i= 0; i < nslicesy; i++) { for (j = 0; j < nslicesy; j++) { printf(" %3d ",tab_y[i][j]); } printf("\n"); }
				
				starpu_pause();
				for (i = 0; i < nslicesx; i++)
				{
					for (j = 0; j < nslicesy; j++)
					{
						struct starpu_task *task = starpu_task_create();
						task->cl = &cl_gemm0;
						
						task->handles[0] = starpu_data_get_sub_data(A_handle, 1, tab_y[i][j]);
						task->handles[1] = starpu_data_get_sub_data(B_handle, 1, tab_x[i][j]);
						task->handles[2] = starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]);
						task->flops = 2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim;
						ret = starpu_task_submit(task); if (ret == -ENODEV) { ret = 77; goto enodev; }
						STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit"); 
						starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, tab_x[i][j], tab_y[i][j]));
					}
				     //~ check = 0;
				     //~ ret = 77;
				     //~ goto enodev;
				}
				starpu_resume();
				starpu_task_wait_for_all();
			}
			//End If RECURSIVE_MATRIX_LAYOUT == 1
		}
		else if (starpu_get_env_number_default("RANDOM_DATA_ACCESS",0) == 1) {
			for (iter = 0; iter < niter; iter++)
			{
				starpu_pause();
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					struct starpu_task *task = starpu_task_create();

					task->cl = &cl_gemm0;
					//random x et y mais meme nombre de tâches inf a nslicesx et y pour la matrice A et B seulement
					task->handles[0] = starpu_data_get_sub_data(A_handle, 1, random()%nslicesy);
					task->handles[1] = starpu_data_get_sub_data(B_handle, 1, random()%nslicesx);
					task->handles[2] = starpu_data_get_sub_data(C_handle, 2, x, y);		
					
					task->flops = 2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim;

					ret = starpu_task_submit(task);
					if (ret == -ENODEV)
					{
						 ret = 77;
						 goto enodev;
					}
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
					starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, x, y));
				}
				starpu_resume();

				starpu_task_wait_for_all();
			}	
		}
		else { 
			//If environment variable RANDOM_TASK_ORDER == 0
			for (iter = 0; iter < niter; iter++)
			{
				starpu_pause();
				for (x = 0; x < nslicesx; x++)
				for (y = 0; y < nslicesy; y++)
				{
					struct starpu_task *task = starpu_task_create();

					task->cl = &cl_gemm0;
					//random x et y mais meme nombre de tâches inf a nslicesx et y pour la matrice A et B seulement
					task->handles[0] = starpu_data_get_sub_data(A_handle, 1, y);
					task->handles[1] = starpu_data_get_sub_data(B_handle, 1, x);
					task->handles[2] = starpu_data_get_sub_data(C_handle, 2, x, y);		
					
					task->flops = 2ULL * (xdim/nslicesx) * (ydim/nslicesy) * zdim;

					ret = starpu_task_submit(task);
					if (ret == -ENODEV)
					{
						 ret = 77;
						 goto enodev;
					}
					STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
					starpu_data_invalidate_submit(starpu_data_get_sub_data(C_handle, 2, x, y));
				}
				starpu_resume();

				starpu_task_wait_for_all();
			}	
			//End If environment variable RANDOM_TASK_ORDER == 0
		}

		end = starpu_timing_now();
		starpu_fxt_stop_profiling();

		if (bound)
			starpu_bound_stop();

		double timing = end - start;
		double min, min_int;
		double flops = 2.0*((unsigned long long)niter)*((unsigned long long)xdim)
				   *((unsigned long long)ydim)*((unsigned long long)zdim);

		if (bound)
			starpu_bound_compute(&min, &min_int, 1);

		if (print_hostname)
		{
			char hostname[255];
			gethostname(hostname, 255);
			PRINTF("%s\t", hostname);
		}
		PRINTF("%u\t%u\t%u\t%.0f\t%.1f", xdim, ydim, zdim, timing/niter/1000.0, flops/timing/1000.0);
		if (bound)
			PRINTF("\t%.0f\t%.1f\t%.0f\t%.1f", min, flops/min/1000000.0, min_int, flops/min_int/1000000.0);
		PRINTF("\n");

		if (sleeps < nsleeps-1)
		{
			sleep(10);
		}
	}

enodev:
	{
		unsigned x, y;
		for (x = 0; x < nslicesx; x++)
		for (y = 0; y < nslicesy; y++)
		{
			starpu_data_handle_t subhandle = starpu_data_get_sub_data(C_handle, 2, x, y);
			starpu_data_acquire(subhandle, STARPU_W);
			//~ starpu_data_acquire(subhandle, STARPU_R);
			starpu_data_release(subhandle);
		}
	}
	starpu_data_unpartition(C_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(B_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(A_handle, STARPU_MAIN_RAM);

	starpu_data_unregister(A_handle);
	starpu_data_unregister(B_handle);
	starpu_data_unregister(C_handle);

#ifndef STARPU_SIMGRID
	if (check)
		ret = check_output();
#endif

	starpu_free_flags(A, zdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(B, xdim*zdim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	starpu_free_flags(C, xdim*ydim*sizeof(TYPE), STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);

	starpu_cublas_shutdown();
	starpu_shutdown();

	return ret;
}
