/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#include <starpu_mpi.h>
#include "mpi_cholesky.h"
#include "mpi_cholesky_models.h"

/*
 *	Create the codelets
 */

static starpu_codelet cl11 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = chol_cpu_codelet_update_u11,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u11,
#endif
	.nbuffers = 1,
	.model = &chol_model_11
};

static starpu_codelet cl21 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = chol_cpu_codelet_update_u21,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u21,
#endif
	.nbuffers = 2,
	.model = &chol_model_21
};

static starpu_codelet cl22 =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = chol_cpu_codelet_update_u22,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u22,
#endif
	.nbuffers = 3,
	.model = &chol_model_22
};

/* Returns the MPI node number where data indexes index is */
int my_distrib(int x, int y, int nb_nodes) {
        return (x+y) % nb_nodes;
}

/*
 *	code to bootstrap the factorization
 *	and construct the DAG
 */
static void dw_cholesky(float *matA, unsigned size, unsigned ld, unsigned nblocks, int rank, int nodes)
{
	struct timeval start;
	struct timeval end;
        starpu_data_handle **data_handles;
        int x, y;

	/* create all the DAG nodes */
	unsigned i,j,k;

        data_handles = malloc(nblocks*sizeof(starpu_data_handle *));
        for(x=0 ; x<nblocks ; x++) data_handles[x] = malloc(nblocks*sizeof(starpu_data_handle));

	gettimeofday(&start, NULL);
        for(x = 0; x < nblocks ;  x++) {
                for (y = 0; y < nblocks; y++) {
                        int mpi_rank = my_distrib(x, y, nodes);
                        if (mpi_rank == rank) {
                                //fprintf(stderr, "[%d] Owning data[%d][%d]\n", rank, x, y);
                                starpu_matrix_data_register(&data_handles[x][y], 0, (uintptr_t)&(matA[((size/nblocks)*y) + ((size/nblocks)*x) * ld]),
                                                            ld, size/nblocks, size/nblocks, sizeof(float));
                        }
                        else if (rank == mpi_rank+1 || rank == mpi_rank-1) {
                                /* I don't own that index, but will need it for my computations */
                                //fprintf(stderr, "[%d] Neighbour of data[%d][%d]\n", rank, x, y);
                                starpu_matrix_data_register(&data_handles[x][y], -1, (uintptr_t)NULL,
                                                            ld, size/nblocks, size/nblocks, sizeof(float));
                        }
                        else {
                                /* I know it's useless to allocate anything for this */
                                data_handles[x][y] = NULL;
                        }
                        if (data_handles[x][y])
                                starpu_data_set_rank(data_handles[x][y], mpi_rank);
                }
        }

	for (k = 0; k < nblocks; k++)
        {
                int prio = STARPU_DEFAULT_PRIO;
                if (!noprio) prio = STARPU_MAX_PRIO;

                starpu_mpi_insert_task(MPI_COMM_WORLD, &cl11,
                                       STARPU_PRIORITY, prio,
                                       STARPU_RW, data_handles[k][k],
                                       0);

		for (j = k+1; j<nblocks; j++)
		{
                        prio = STARPU_DEFAULT_PRIO;
                        if (!noprio&& (j == k+1)) prio = STARPU_MAX_PRIO;
                        starpu_mpi_insert_task(MPI_COMM_WORLD, &cl21,
                                               STARPU_PRIORITY, prio,
                                               STARPU_R, data_handles[k][k],
                                               STARPU_RW, data_handles[k][j],
                                               0);

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
                                {
                                        prio = STARPU_DEFAULT_PRIO;
                                        if (!noprio && (i == k + 1) && (j == k +1) ) prio = STARPU_MAX_PRIO;
                                        starpu_mpi_insert_task(MPI_COMM_WORLD, &cl22,
                                                               STARPU_PRIORITY, prio,
                                                               STARPU_R, data_handles[k][i],
                                                               STARPU_R, data_handles[k][j],
                                                               STARPU_RW, data_handles[i][j],
                                                               0);
                                }
			}
		}
        }

        starpu_task_wait_for_all();

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "Computation took (in ms)\n");
	printf("%2.2f\n", timing/1000);

	double flop = (1.0f*size*size*size)/3.0f;
	fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

void initialize_system(float **A, unsigned dim, unsigned pinned, int *rank, int *nodes)
{
	starpu_init(NULL);
	starpu_mpi_initialize_extended(1, rank, nodes);
	starpu_helper_cublas_init();

	if (pinned)
	{
		starpu_malloc((void **)A, (size_t)dim*dim*sizeof(float));
	}
	else {
		*A = malloc(dim*dim*sizeof(float));
	}
}

int main(int argc, char **argv)
{
	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1)
	 * */

	float *mat;
        int rank, nodes;

	parse_args(argc, argv);
	mat = malloc(size*size*sizeof(float));
	initialize_system(&mat, size, pinned, &rank, &nodes);

	unsigned i,j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			//mat[j +i*size] = ((i == j)?1.0f*size:0.0f);
		}
	}


        if (display) {
                printf("[%d] Input :\n", rank);

                for (j = 0; j < size; j++)
                {
                        for (i = 0; i < size; i++)
                        {
                                if (i <= j) {
                                        printf("%2.2f\t", mat[j +i*size]);
                                }
                                else {
                                        printf(".\t");
                                }
                        }
                        printf("\n");
                }
        }

	dw_cholesky(mat, size, size, nblocks, rank, nodes);

        starpu_helper_cublas_shutdown();
	starpu_mpi_shutdown();
	starpu_shutdown();

        if (display) {
                printf("[%d] Results :\n", rank);

                for (j = 0; j < size; j++)
		{
                        for (i = 0; i < size; i++)
			{
                                if (i <= j) {
                                        printf("%2.2f\t", mat[j +i*size]);
                                }
                                else {
                                        printf(".\t");
                                }
                        }
                        printf("\n");
                }
        }

	fprintf(stderr, "[%d] compute explicit LLt ...\n", rank);
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
		{
			if (i > j) {
				mat[j+i*size] = 0.0f; // debug
			}
		}
	}
	float *test_mat = malloc(size*size*sizeof(float));
	STARPU_ASSERT(test_mat);

	SSYRK("L", "N", size, size, 1.0f,
				mat, size, 0.0f, test_mat, size);

	fprintf(stderr, "[%d] comparing results ...\n", rank);
        if (display) {
                for (j = 0; j < size; j++)
		{
                        for (i = 0; i < size; i++)
			{
                                if (i <= j) {
                                        printf("%2.2f\t", test_mat[j +i*size]);
                                }
                                else {
                                        printf(".\t");
                                }
                        }
                        printf("\n");
                }
        }

        int x, y;
        for(x = 0; x < nblocks ;  x++)
	{
                for (y = 0; y < nblocks; y++)
		{
                        int mpi_rank = my_distrib(x, y, nodes);
                        if (mpi_rank == rank) {
                                for (i = (size/nblocks)*x ; i < (size/nblocks)*x+(size/nblocks); i++)
                                {
                                        for (j = (size/nblocks)*y ; j < (size/nblocks)*y+(size/nblocks); j++)
                                        {
                                                if (i <= j)
                                                {
                                                        float orig = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
                                                        float err = abs(test_mat[j +i*size] - orig);
                                                        if (err > 0.00001) {
                                                                fprintf(stderr, "[%d] Error[%d, %d] --> %2.2f != %2.2f (err %2.2f)\n", rank, i, j, test_mat[j +i*size], orig, err);
                                                                assert(0);
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

	return 0;
}
