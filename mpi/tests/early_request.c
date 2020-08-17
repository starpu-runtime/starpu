/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <starpu_mpi.h>
#include "helper.h"

#define NUM_EL 5
#define NUM_LOOPS 10

/*
 * This testcase written by J-M Couteyen allows to test that several
 * early requests for a given source and tag can be posted to StarPU
 * by the application before data arrive.
 *
 * In this test case, multiples processes (called "domains") exchanges
 * informations between multiple "elements" multiple times, with
 * different sizes (in order to catch error more easily).
 * The communications are independent between the elements (each one
 * as its proper tag), but must occur in the submitted order for an
 * element taken independtly.
*/

struct element
{
	int tag;
	int foreign_domain;

	int array_send[100];
	int array_recv[100];

	starpu_data_handle_t ensure_submitted_order_send;
	starpu_data_handle_t ensure_submitted_order_recv;
	starpu_data_handle_t send;
	starpu_data_handle_t recv;
};

/* functions/codelet to fill the bufferss*/
void fill_tmp_buffer(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	int *tmp = (int *) STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;

	for (i=0; i<nx; i++)
		tmp[i]=nx+i;
}

static struct starpu_codelet fill_tmp_buffer_cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = {fill_tmp_buffer, NULL},
	.nbuffers = 1,
	.modes = {STARPU_W},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "fill_tmp_buffer"
};

void read_ghost(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	int *tmp = (int *) STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx=STARPU_VECTOR_GET_NX(buffers[0]);
	int i;
	for(i=0; i<nx;i++)
	{
		assert(tmp[i]==nx+i);
	}
}

static struct starpu_codelet read_ghost_value_cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = {read_ghost, NULL},
	.nbuffers = 1,
	.modes = {STARPU_R},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "read_ghost_value"
};

/*codelet to ensure submitted order for a given element*/
void noop(void *buffers[], void *cl_arg)
{
	(void)buffers;
	(void)cl_arg;
}

void submitted_order_fun(void *buffers[], void *cl_arg)
{
	(void)buffers;
	(void)cl_arg;
}

static struct starpu_codelet submitted_order_rw =
{
	.where = STARPU_CPU,
	.cpu_funcs = {submitted_order_fun, NULL},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_RW},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "submitted_order_enforcer"
};

static struct starpu_codelet submitted_order =
{
	.where = STARPU_CPU,
	.cpu_funcs = {submitted_order_fun, NULL},
	.nbuffers = 2,
	.modes = {STARPU_RW, STARPU_W},
#ifdef STARPU_SIMGRID
	.model = &starpu_perfmodel_nop,
#endif
	.name = "submitted_order_enforcer"
};

void init_element(struct element *el, int size, int foreign_domain)
{
	el->tag=size;
	el->foreign_domain=foreign_domain;

	int mpi_rank;
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &mpi_rank);

	starpu_vector_data_register(&el->recv, 0, (uintptr_t)el->array_recv, size, sizeof(int));
	starpu_vector_data_register(&el->send, 0, (uintptr_t)el->array_send, size, sizeof(int));
	starpu_void_data_register(&el->ensure_submitted_order_send);
	starpu_void_data_register(&el->ensure_submitted_order_recv);
}

void free_element(struct element *el)
{
	starpu_data_unregister(el->recv);
	starpu_data_unregister(el->send);
	starpu_data_unregister(el->ensure_submitted_order_send);
	starpu_data_unregister(el->ensure_submitted_order_recv);
}

void insert_work_for_one_element(struct element *el)
{
	starpu_data_handle_t tmp_recv;
	starpu_data_handle_t tmp_send;

	starpu_vector_data_register(&tmp_recv, -1, 0, el->tag, sizeof(int));
	starpu_vector_data_register(&tmp_send, -1, 0, el->tag, sizeof(int));

	//Emulate the work to fill the send buffer
	starpu_insert_task(&fill_tmp_buffer_cl,
			   STARPU_W,tmp_send,
			   0);
	//Send operation
	starpu_insert_task(&submitted_order_rw,
			   STARPU_RW,el->ensure_submitted_order_send,
			   STARPU_RW,tmp_send,
			   0);
	starpu_mpi_isend_detached(tmp_send,el->foreign_domain,el->tag, MPI_COMM_WORLD, NULL, NULL);
	starpu_insert_task(&submitted_order_rw,
			   STARPU_RW,el->ensure_submitted_order_send,
			   STARPU_RW,tmp_send,
			   0);

	//Recv operation for current element
	starpu_insert_task(&submitted_order,
			   STARPU_RW,el->ensure_submitted_order_recv,
			   STARPU_W,tmp_recv,
			   0);
	starpu_mpi_irecv_detached(tmp_recv,el->foreign_domain,el->tag, MPI_COMM_WORLD, NULL, NULL);
	//Emulate the "reading" of the recv value.
	starpu_insert_task(&read_ghost_value_cl,
			   STARPU_R,tmp_recv,
			   0);

	starpu_data_unregister_submit(tmp_send);
	starpu_data_unregister_submit(tmp_recv);
}

/*main program*/
int main(int argc, char * argv[])
{
	/* Init */
	int ret;
	int mpi_rank, mpi_size;
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);

	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &mpi_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &mpi_size);

	if (starpu_cpu_worker_get_count() == 0)
	{
		if (mpi_rank == 0)
			FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_mpi_shutdown();
		if (!mpi_init)
			MPI_Finalize();
		return STARPU_TEST_SKIPPED;
	}

	/*element initialization : domains are connected as a ring for this test*/
	int num_elements=NUM_EL;
	struct element * el_left=malloc(num_elements*sizeof(el_left[0]));
	struct element * el_right=malloc(num_elements*sizeof(el_right[0]));
	int i;
	for(i=0;i<num_elements;i++)
	{
		init_element(el_left+i,i+1,((mpi_rank-1)+mpi_size)%mpi_size);
		init_element(el_right+i,i+1,(mpi_rank+1)%mpi_size);
	}

	/* Communication loop */
	for (i=0; i<NUM_LOOPS; i++) //number of "computations loops"
	{
		int e;
		for (e=0;e<num_elements;e++) //Do something for each elements
		{
			insert_work_for_one_element(el_right+e);
			insert_work_for_one_element(el_left+e);
		}
	}
	/* End */
	starpu_task_wait_for_all();

	for(i=0;i<num_elements;i++)
	{
		free_element(el_left+i);
		free_element(el_right+i);
	}
	free(el_left);
	free(el_right);

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();
	FPRINTF(stderr, "No assert until end\n");
	return 0;
}
