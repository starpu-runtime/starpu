/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <math.h>

/* Manage block and tags allocation */

static struct block_description *blocks;
static size_t sizex, sizey, sizez;
static size_t nbz;
static size_t *block_sizes_z;

/*
 *	Tags for various codelet completion
 */

/*
 * common tag format:
 */
static starpu_tag_t tag_common(int z, int dir, int type)
{
	return (((((starpu_tag_t)type) << 4) | ((dir+1)/2)) << 32)|(starpu_tag_t)z;
}

/* Completion of last update tasks */
starpu_tag_t TAG_FINISH(int z)
{
	z = (z + nbz)%nbz;

	starpu_tag_t tag = tag_common(z, 0, 1);
	return tag;
}

/* Completion of the save codelet for MPI send/recv */
starpu_tag_t TAG_START(int z, int dir)
{
	z = (z + nbz)%nbz;

	starpu_tag_t tag = tag_common(z, dir, 2);
	return tag;
}

/*
 * common MPI tag format:
 */
static int mpi_tag_common(int z, int dir, int layer_or_boundary, int buffer)
{
	return (z<<12) | (layer_or_boundary << 8) | ((((1+dir)/2))<<4) | buffer;
}

int MPI_TAG_LAYERS(int z, int buffer)
{
	z = (z + nbz)%nbz;

	/* No direction for layers ; layer is 0 */
	int tag = mpi_tag_common(z, 0, 0, buffer);

	return tag;
}

int MPI_TAG_BOUNDARIES(int z, int dir, int buffer)
{
	z = (z + nbz)%nbz;

	int tag = mpi_tag_common(z, dir, 1, buffer);

	return tag;
}


/*
 *	Block descriptors
 */

/* Compute the size of the different blocks */
static void compute_block_sizes(void)
{
	block_sizes_z = (size_t *) malloc(nbz*sizeof(size_t));
	STARPU_ASSERT(block_sizes_z);

	/* Perhaps the last chunk is smaller */
	size_t default_block_size = (sizez+nbz-1)/nbz;
	size_t remaining = sizez;

	size_t b;
	for (b = 0; b < nbz; b++)
	{
		block_sizes_z[b] = MIN(default_block_size, remaining);
		remaining -= block_sizes_z[b];
	}

	STARPU_ASSERT(remaining == 0);
}

size_t get_block_size(int bz)
{
	return block_sizes_z[bz];
}

struct block_description *get_block_description(int z)
{
	z = (z + nbz)%nbz;

	STARPU_ASSERT(&blocks[z]);

	return &blocks[z];
}

int get_block_mpi_node(int z)
{
	z = (z + nbz)%nbz;
	return blocks[z].mpi_node;
}

void create_blocks_array(size_t _sizex, size_t _sizey, size_t _sizez, size_t _nbz)
{
	/* Store the parameters */
	nbz = _nbz;
	sizex = _sizex;
	sizey = _sizey;
	sizez = _sizez;

	/* Create a grid of block descriptors */
	blocks = (struct block_description *) calloc(nbz, sizeof(struct block_description));
	STARPU_ASSERT(blocks);

	/* What is the size of the different blocks ? */
	compute_block_sizes();

	size_t bz;
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description * block =
				get_block_description(bz);

		/* Which block is it ? */
		block->bz = bz;

		/* For simplicity, we store which are the neighbours blocks */
		block->boundary_blocks[B] = get_block_description((bz-1+nbz)%nbz);
		block->boundary_blocks[T] = get_block_description((bz+1)%nbz);
	}
}

void free_blocks_array()
{
	free(blocks);
	free(block_sizes_z);
}

/*
 *	Initialization of the blocks
 */

void assign_blocks_to_workers(int rank)
{
	size_t bz;

	/* NB: perhaps we could count a GPU as multiple workers */

	/* how many workers are there ? */
	/*unsigned nworkers = starpu_worker_get_count();*/

	/* how many blocks are on that MPI node ? */
//	size_t nblocks = 0;
//	for (bz = 0; bz < nbz; bz++)
//	{
//		struct block_description *block =
//				get_block_description(bz);
//
//		if (block->mpi_node == rank)
//			nblocks++;
//	}

	/* how many blocks per worker ? */
	/*size_t nblocks_per_worker = (nblocks + nworkers - 1)/nworkers;*/

	/* we now attribute up to nblocks_per_worker blocks per workers */
	unsigned attributed = 0;
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block =
				get_block_description(bz);

		if (block->mpi_node == rank)
		{
			unsigned workerid;
			/* Manage initial block distribution between CPU and GPU */
		#if 0
			#if 1
			/* GPUs then CPUs */
			if (attributed < 3*18)
				workerid = attributed / 18;
			else
				workerid = 3+ (attributed - 3*18) / 2;
			#else
			/* GPUs interleaved with CPUs */
			if ((attributed % 20) <= 1)
				workerid = 3 + attributed / 20;
			else if (attributed < 60)
				workerid = attributed / 20;
			else
				workerid = (attributed - 60)/2 + 6;
			#endif
		#else
			/* Only GPUS */
			workerid = (attributed / 21) % 3;
		#endif
			/*= attributed/nblocks_per_worker;*/

			block->preferred_worker = workerid;

			attributed++;
		}
	}
}



void assign_blocks_to_mpi_nodes(int world_size)
{
	size_t nzblocks_per_process = (nbz + world_size - 1) / world_size;

	size_t bz;
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block =
				get_block_description(bz);

		block->mpi_node = bz / nzblocks_per_process;
	}
}

static size_t allocated = 0;

static void allocate_block_on_node(starpu_data_handle_t *handleptr, size_t bz, TYPE **ptr, size_t nx, size_t ny, size_t nz)
{
	int ret;
	size_t block_size = nx*ny*nz*sizeof(TYPE);

	/* Allocate memory */
#if 1
	ret = starpu_malloc_flags((void **)ptr, block_size, STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
	STARPU_ASSERT(ret == 0);
#else
	*ptr = malloc(block_size);
	STARPU_ASSERT(*ptr);
#endif

	allocated += block_size;

//#ifndef STARPU_SIMGRID
//	/* Fill the blocks with 0 */
//	memset(*ptr, 0, block_size);
//#endif

	/* Register it to StarPU */
	starpu_block_data_register(handleptr, STARPU_MAIN_RAM, (uintptr_t)*ptr, nx, nx*ny, nx, ny, nz, sizeof(TYPE));

	starpu_data_set_coordinates(*handleptr, 1, bz);
}

static void free_block_on_node(starpu_data_handle_t handleptr, size_t nx, size_t ny, size_t nz)
{
	void *ptr = (void *) starpu_block_get_local_ptr(handleptr);
	size_t block_size = nx*ny*nz*sizeof(TYPE);
	starpu_data_unregister(handleptr);
	starpu_free_flags(ptr, block_size, STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
}

void display_memory_consumption(int rank, double time)
{
	FPRINTF(stderr, "%zu B of memory were allocated on node %d in %f ms\n", allocated, rank, time/1000);
}

void allocate_memory_on_node(int rank)
{
	size_t bz;

	/* Correctly allocate and declare all data handles to StarPU. */
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block = get_block_description(bz);
		int node = block->mpi_node;
		size_t size_bz = block_sizes_z[bz];

		if (node == rank)
		{
			/* Main blocks */
			allocate_block_on_node(&block->layers_handle[0], bz, &block->layers[0],
					       (sizex + 2*K), (sizey + 2*K), (size_bz + 2*K));
			allocate_block_on_node(&block->layers_handle[1], bz, &block->layers[1],
					       (sizex + 2*K), (sizey + 2*K), (size_bz + 2*K));

			/* Boundary blocks : Top */
			allocate_block_on_node(&block->boundaries_handle[T][0], bz, &block->boundaries[T][0],
					       (sizex + 2*K), (sizey + 2*K), K);
			allocate_block_on_node(&block->boundaries_handle[T][1], bz, &block->boundaries[T][1],
					       (sizex + 2*K), (sizey + 2*K), K);

			/* Boundary blocks : Bottom */
			allocate_block_on_node(&block->boundaries_handle[B][0], bz, &block->boundaries[B][0],
					       (sizex + 2*K), (sizey + 2*K), K);
			allocate_block_on_node(&block->boundaries_handle[B][1], bz, &block->boundaries[B][1],
					       (sizex + 2*K), (sizey + 2*K), K);
		}
		/* Register void blocks to StarPU, that StarPU-MPI will request to
		 * neighbour nodes if needed for the local computation */
		else
		{
			/* Main blocks */
			starpu_block_data_register(&block->layers_handle[0], -1, (uintptr_t) NULL, (sizex + 2*K), (sizex + 2*K)*(sizey + 2*K), (sizex + 2*K), (sizey + 2*K), (size_bz + 2*K), sizeof(TYPE));
			starpu_block_data_register(&block->layers_handle[1], -1, (uintptr_t) NULL, (sizex + 2*K), (sizex + 2*K)*(sizey + 2*K), (sizex + 2*K), (sizey + 2*K), (size_bz + 2*K), sizeof(TYPE));

			/* Boundary blocks : Top */
			starpu_block_data_register(&block->boundaries_handle[T][0], -1, (uintptr_t) NULL, (sizex + 2*K), (sizex + 2*K)*(sizey + 2*K), (sizex + 2*K), (sizey + 2*K), K, sizeof(TYPE));
			starpu_block_data_register(&block->boundaries_handle[T][1], -1, (uintptr_t) NULL, (sizex + 2*K), (sizex + 2*K)*(sizey + 2*K), (sizex + 2*K), (sizey + 2*K), K, sizeof(TYPE));

			/* Boundary blocks : Bottom */
			starpu_block_data_register(&block->boundaries_handle[B][0], -1, (uintptr_t) NULL, (sizex + 2*K), (sizex + 2*K)*(sizey + 2*K), (sizex + 2*K), (sizey + 2*K), K, sizeof(TYPE));
			starpu_block_data_register(&block->boundaries_handle[B][1], -1, (uintptr_t) NULL, (sizex + 2*K), (sizex + 2*K)*(sizey + 2*K), (sizex + 2*K), (sizey + 2*K), K, sizeof(TYPE));
		}

#if defined(STARPU_USE_MPI)  && !defined(STARPU_USE_MPI_SERVER_CLIENT)
		/* Register all data to StarPU-MPI, even the ones that are not
		 * allocated on the local node. */

		/* Main blocks */
		starpu_mpi_data_register(block->layers_handle[0], MPI_TAG_LAYERS(bz, 0), node);
		starpu_mpi_data_register(block->layers_handle[1], MPI_TAG_LAYERS(bz, 1), node);

		/* Boundary blocks : Top */
		starpu_mpi_data_register(block->boundaries_handle[T][0], MPI_TAG_BOUNDARIES(bz, T, 0), node);
		starpu_mpi_data_register(block->boundaries_handle[T][1], MPI_TAG_BOUNDARIES(bz, T, 1), node);

		/* Boundary blocks : Bottom */
		starpu_mpi_data_register(block->boundaries_handle[B][0], MPI_TAG_BOUNDARIES(bz, B, 0), node);
		starpu_mpi_data_register(block->boundaries_handle[B][1], MPI_TAG_BOUNDARIES(bz, B, 1), node);
#endif
	}

	/* Initialize all the data in parallel */
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block = get_block_description(bz);
		int node = block->mpi_node;

		if (node == rank)
		{
			/* Set all the data to 0 */
			create_task_memset(sizex, sizey, bz);

			/* Initialize the first layer with some random data */
			create_task_initlayer(sizex, sizey, bz);
		}
	}
	starpu_task_wait_for_all();
}

void free_memory_on_node(int rank)
{
	size_t bz;
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block = get_block_description(bz);

		int node = block->mpi_node;

		/* Main blocks */
		if (node == rank)
		{
			free_block_on_node(block->layers_handle[0], (sizex + 2*K), (sizey + 2*K), K);
			free_block_on_node(block->layers_handle[1], (sizex + 2*K), (sizey + 2*K), K);
		}
		else
		{
			starpu_data_unregister(block->layers_handle[0]);
			starpu_data_unregister(block->layers_handle[1]);
		}

		/* Boundary blocks : Top */
		if (node == rank)
		{
			free_block_on_node(block->boundaries_handle[T][0], (sizex + 2*K), (sizey + 2*K), K);
			free_block_on_node(block->boundaries_handle[T][1], (sizex + 2*K), (sizey + 2*K), K);
		}
		else
		{
			starpu_data_unregister(block->boundaries_handle[T][0]);
			starpu_data_unregister(block->boundaries_handle[T][1]);
		}

		/* Boundary blocks : Bottom */
		if (node == rank)
		{
			free_block_on_node(block->boundaries_handle[B][0], (sizex + 2*K), (sizey + 2*K), K);
			free_block_on_node(block->boundaries_handle[B][1], (sizex + 2*K), (sizey + 2*K), K);
		}
		else
		{
			starpu_data_unregister(block->boundaries_handle[B][0]);
			starpu_data_unregister(block->boundaries_handle[B][1]);
		}
	}
}

/* check how many cells are alive */
void check(int rank)
{
	size_t bz;
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block = get_block_description(bz);

		int node = block->mpi_node;

		/* Main blocks */
		if (node == rank)
		{
			size_t size_bz = block_sizes_z[bz];
#ifdef LIFE
			size_t x, y, z;
			size_t sum = 0;
			for (x = 0; x < sizex; x++)
				for (y = 0; y < sizey; y++)
					for (z = 0; z < size_bz; z++)
						sum += block->layers[0][(K+x)+(K+y)*(sizex + 2*K)+(K+z)*(sizex+2*K)*(sizey+2*K)];
			printf("block %zu got %zu/%zu alive\n", bz, sum, sizex*sizey*size_bz);
#endif
		}
	}
}
