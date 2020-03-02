/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "implicit-stencil.h"

#define BIND_LAST 1

/*
 * Schedule tasks for updates and saves
 */

/*
 * NB: iter = 0: initialization phase, TAG_U(z, 0) = TAG_INIT
 *
 * dir is -1 or +1.
 */

#if 0
# define DEBUG(fmt, ...) fprintf(stderr,fmt,##__VA_ARGS__)
#else
# define DEBUG(fmt, ...)
#endif

#if defined(STARPU_USE_MPI) && !defined(STARPU_USE_MPI_MASTER_SLAVE)
#include <starpu_mpi.h>
#define starpu_insert_task(...) starpu_mpi_insert_task(MPI_COMM_WORLD, __VA_ARGS__)
#endif

/*
 * Schedule initialization tasks
 */

void create_task_memset(unsigned sizex, unsigned sizey, unsigned z)
{
	struct block_description *descr = get_block_description(z);
    struct starpu_codelet *codelet = &cl_memset;

    int ret = starpu_insert_task(
            codelet,
            STARPU_VALUE,   &sizex,  sizeof(unsigned),
            STARPU_VALUE,   &sizey,  sizeof(unsigned),
            STARPU_VALUE,   &z,  sizeof(unsigned),
            STARPU_W,   descr->layers_handle[0],
            STARPU_W,   descr->layers_handle[1],
            STARPU_W,   descr->boundaries_handle[T][0],
            STARPU_W,   descr->boundaries_handle[T][1],
            STARPU_W,   descr->boundaries_handle[B][0],
            STARPU_W,   descr->boundaries_handle[B][1],
                0);

    if (ret)
    {
        FPRINTF(stderr, "Could not submit task save: %d\n", ret);
        if (ret == -ENODEV)
            exit(77);
        STARPU_ABORT();
    }
}

void create_task_initlayer(unsigned sizex, unsigned sizey, unsigned z)
{
	struct block_description *descr = get_block_description(z);
    struct starpu_codelet *codelet = &cl_initlayer;

    int ret = starpu_insert_task(
            codelet,
            STARPU_VALUE,   &sizex,  sizeof(unsigned),
            STARPU_VALUE,   &sizey,  sizeof(unsigned),
            STARPU_VALUE,   &z,  sizeof(unsigned),
            STARPU_W,   descr->layers_handle[0],
                0);

    if (ret)
    {
        FPRINTF(stderr, "Could not submit task save: %d\n", ret);
        if (ret == -ENODEV)
            exit(77);
        STARPU_ABORT();
    }
}

/*
 * Schedule saving boundaries of blocks to communication buffers
 */

static void create_task_save_local(unsigned z, int dir)
{
	struct block_description *descr = get_block_description(z);
	struct starpu_codelet *codelet;
	int ret;

	codelet = (dir == -1)?&save_cl_bottom:&save_cl_top;
	ret = starpu_insert_task(
				 codelet,
				 STARPU_VALUE,   &z,  sizeof(unsigned),
				 STARPU_R,   descr->layers_handle[0],
				 STARPU_R,   descr->layers_handle[1],
				 STARPU_W,   descr->boundaries_handle[(1-dir)/2][0],
				 STARPU_W,   descr->boundaries_handle[(1-dir)/2][1],
				 STARPU_PRIORITY,    STARPU_MAX_PRIO,
				 0);

	if (ret)
	{
		FPRINTF(stderr, "Could not submit task save: %d\n", ret);
		if (ret == -ENODEV)
			exit(77);
		STARPU_ABORT();
	}
}

/*
 * Schedule update computation in computation buffer
 */

void create_task_update(unsigned iter, unsigned z, int local_rank)
{
	STARPU_ASSERT(iter != 0);

	unsigned old_layer = (K*(iter-1)) % 2;
	unsigned new_layer = (old_layer + 1) % 2;

	struct block_description *descr = get_block_description(z);
	struct block_description *bottom_neighbour = descr->boundary_blocks[B];
	struct block_description *top_neighbour = descr->boundary_blocks[T];

	struct starpu_codelet *codelet = &cl_update;

    // Simple-level prio
    //int prio = ((bottom_neighbour->mpi_node != local_rank) || (top_neighbour->mpi_node != local_rank )) ? STARPU_MAX_PRIO : STARPU_DEFAULT_PRIO;

    // Two-level prio
    int prio = ((bottom_neighbour->mpi_node != local_rank) || (top_neighbour->mpi_node != local_rank )) ? STARPU_MAX_PRIO :
               ((bottom_neighbour->boundary_blocks[B]->mpi_node != local_rank) || (top_neighbour->boundary_blocks[T]->mpi_node != local_rank )) ? STARPU_MAX_PRIO-1 : STARPU_DEFAULT_PRIO;

    int ret = starpu_insert_task(
            codelet,
            STARPU_VALUE,   &z,  sizeof(unsigned),
	        STARPU_RW,      descr->layers_handle[old_layer],
	        STARPU_RW,      descr->layers_handle[new_layer],
	        STARPU_R,       bottom_neighbour->boundaries_handle[T][old_layer],
	        STARPU_R,       bottom_neighbour->boundaries_handle[T][new_layer],
	        STARPU_R,       top_neighbour->boundaries_handle[B][old_layer],
	        STARPU_R,       top_neighbour->boundaries_handle[B][new_layer],
            STARPU_PRIORITY,    prio,
                0);
	if (ret)
	{
		FPRINTF(stderr, "Could not submit task update block: %d\n", ret);
		if (ret == -ENODEV)
			exit(77);
		STARPU_ABORT();
	}
}

/*
 * Create all the tasks
 */
void create_tasks(int rank)
{
	int iter;
	int bz;
	int niter = get_niter();
	int nbz = get_nbz();

	for (iter = 0; iter <= niter; iter++)
	{
	     for (bz = 0; bz < nbz; bz++)
	     {
		    if ((iter > 0) && ((get_block_mpi_node(bz) == rank)|| (get_block_mpi_node(bz+1) == rank)|| (get_block_mpi_node(bz-1) == rank)))
			    create_task_update(iter, bz, rank);
	     }

	     for (bz = 0; bz < nbz; bz++)
	     {
		     if (iter != niter)
		     {
			     int node_z = get_block_mpi_node(bz);
			     int node_z_and_b = get_block_mpi_node(bz-1);
			     int node_z_and_t = get_block_mpi_node(bz+1);

			     if ((node_z == rank) || ((node_z != node_z_and_b) && (node_z_and_b == rank)))
				     create_task_save_local(bz, +1);

			     if ((node_z == rank) || ((node_z != node_z_and_t) && (node_z_and_t == rank)))
				     create_task_save_local(bz, -1);
		     }
	     }
	}
}
