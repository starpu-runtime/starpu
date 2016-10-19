/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  Mathieu Lirzin <mthl@openmailbox.org>
 * Copyright (C) 2016  Inria
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

#ifndef __DRIVER_MPI_SOURCE_H__
#define __DRIVER_MPI_SOURCE_H__

#include <drivers/mp_common/mp_common.h>

#ifdef STARPU_USE_MPI_MASTER_SLAVE

/* Array of structures containing all the informations useful to send
 * and receive informations with devices */
extern struct _starpu_mp_node *mpi_ms_nodes[STARPU_MAXMICDEVS];

unsigned _starpu_mpi_src_get_device_count();
void *_starpu_mpi_src_worker(void *arg);
void _starpu_mpi_exit_useless_node(int devid);

void _starpu_mpi_source_init(struct _starpu_mp_node *node);
void _starpu_mpi_source_deinit(struct _starpu_mp_node *node);

///* Send *MSG which can be a command or data, to a MPI sink. */
//extern void _starpu_mpi_source_send(const struct _starpu_mp_node *node,
//				    void *msg, int len);
//
///* Receive *MSG which can be an answer or data, to a MPI sink. */
//extern void _starpu_mpi_source_recv(const struct _starpu_mp_node *node,
//				    void *msg, int len);
//
///* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
// * node to the address pointed by DST in the DST_NODE memory node */
//extern int _starpu_mpi_copy_src_to_sink(void *src,
//					unsigned src_node STARPU_ATTRIBUTE_UNUSED,
//					void *dst, unsigned dst_node,
//					size_t size);
//
///* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
// * node to the address pointed by DST in the DST_NODE memory node */
//extern int _starpu_mpi_copy_sink_to_src(void *src, unsigned src_node, void *dst,
//					unsigned dst_node STARPU_ATTRIBUTE_UNUSED,
//					size_t size);
//
//extern int _starpu_mpi_copy_sink_to_sink(void *src, unsigned src_node,
//					 void *dst, unsigned dst_node,
//					 size_t size);
//
///* Get a pointer which points at the implementation to be called by MPI node. */
//extern void (*_starpu_mpi_get_kernel_from_job(const struct _starpu_mp_node *,
//					      struct _starpu_job *j))(void);

#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#endif	/* __DRIVER_MPI_SOURCE_H__ */
