/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <math.h>
#include "helper.h"

#ifndef STARPU_USE_MPI_MPI
int main(int argc, char **argv)
{
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	if (!mpi_init)
		MPI_Finalize();

	return 0;
}

#else

#include <mpi/starpu_mpi_early_data.h>
#include <mpi/starpu_mpi_early_request.h>
#include <mpi/starpu_mpi_mpi_backend.h>

void early_data()
{
	struct _starpu_mpi_early_data_handle *edh[2];
	struct _starpu_mpi_envelope envelope[2];
	struct _starpu_mpi_node_tag node_tag[2];
	struct _starpu_mpi_early_data_handle *early;
	struct _starpu_mpi_early_data_handle_tag_hashlist *hash;

	memset(&node_tag[0], 0, sizeof(struct _starpu_mpi_node_tag));
	node_tag[0].node.rank = 1;
	node_tag[0].node.comm = MPI_COMM_WORLD;
	node_tag[0].data_tag = 42;

	memset(&node_tag[1], 0, sizeof(struct _starpu_mpi_node_tag));
	node_tag[1].node.rank = 2;
	node_tag[1].node.comm = MPI_COMM_WORLD;
	node_tag[1].data_tag = 84;

	envelope[0].data_tag = node_tag[0].data_tag;
	edh[0] = _starpu_mpi_early_data_create(&envelope[0], node_tag[0].node.rank, node_tag[0].node.comm);

	envelope[1].data_tag = node_tag[1].data_tag;
	edh[1] = _starpu_mpi_early_data_create(&envelope[1], node_tag[1].node.rank, node_tag[1].node.comm);

	_starpu_mpi_early_data_add(edh[0]);
	_starpu_mpi_early_data_add(edh[1]);

	hash = _starpu_mpi_early_data_extract(&node_tag[1]);
	STARPU_ASSERT(_starpu_mpi_early_data_handle_list_size(&hash->list) == 1);
	early = _starpu_mpi_early_data_handle_list_pop_front(&hash->list);
	STARPU_ASSERT(early->node_tag.node.comm == node_tag[1].node.comm && early->node_tag.node.rank == node_tag[1].node.rank && early->node_tag.data_tag == node_tag[1].data_tag);
	STARPU_ASSERT(_starpu_mpi_early_data_handle_list_size(&hash->list) == 0);

	early = _starpu_mpi_early_data_find(&node_tag[0]);
	STARPU_ASSERT(early->node_tag.node.comm == node_tag[0].node.comm && early->node_tag.node.rank == node_tag[0].node.rank && early->node_tag.data_tag == node_tag[0].data_tag);
}

void early_request()
{
	struct _starpu_mpi_req req[2];
	struct _starpu_mpi_req *early;
	struct _starpu_mpi_early_request_tag_hashlist *hash;

	memset(&req[0].node_tag, 0, sizeof(struct _starpu_mpi_node_tag));
	req[0].node_tag.node.rank = 1;
	req[0].node_tag.node.comm = MPI_COMM_WORLD;
	req[0].node_tag.data_tag = 42;

	memset(&req[1].node_tag, 0, sizeof(struct _starpu_mpi_node_tag));
	req[1].node_tag.node.rank = 2;
	req[1].node_tag.node.comm = MPI_COMM_WORLD;
	req[1].node_tag.data_tag = 84;

	_starpu_mpi_early_request_enqueue(&req[1]);
	_starpu_mpi_early_request_enqueue(&req[0]);

	early = _starpu_mpi_early_request_dequeue(req[0].node_tag.data_tag, req[0].node_tag.node.rank, req[0].node_tag.node.comm);
	STARPU_ASSERT(early->node_tag.data_tag == req[0].node_tag.data_tag && early->node_tag.node.rank == req[0].node_tag.node.rank && early->node_tag.node.comm == req[0].node_tag.node.comm);

	hash = _starpu_mpi_early_request_extract(req[1].node_tag.data_tag, req[1].node_tag.node.rank, req[1].node_tag.node.comm);
	STARPU_ASSERT(_starpu_mpi_req_list_size(&hash->list) == 1);
	early = _starpu_mpi_req_list_pop_front(&hash->list);
	STARPU_ASSERT(_starpu_mpi_req_list_size(&hash->list) == 0);
	STARPU_ASSERT(early->node_tag.data_tag == req[1].node_tag.data_tag && early->node_tag.node.rank == req[1].node_tag.node.rank && early->node_tag.node.comm == req[1].node_tag.node.comm);
}

int main(int argc, char **argv)
{
	int ret, rank, size, i;
	starpu_data_handle_t tab_handle[4];
	int mpi_init;

	MPI_INIT_THREAD(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_init);
	ret = starpu_mpi_init_conf(&argc, &argv, mpi_init, MPI_COMM_WORLD, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

	early_data();
	early_request();

	starpu_mpi_shutdown();

	if (!mpi_init)
		MPI_Finalize();
	return 0;
}

#endif
