/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdarg.h>
#include <mpi.h>

#include <starpu.h>
#include <starpu_data.h>
#include <common/utils.h>
#include <common/hash.h>
#include <util/starpu_insert_task_utils.h>

//#define STARPU_MPI_VERBOSE	1
#include <starpu_mpi_private.h>

/* Whether we are allowed to keep copies of remote data. Does not work
 * yet: the sender has to know whether the receiver has it, keeping it
 * in an array indexed by node numbers. */
//#define MPI_CACHE

#ifdef MPI_CACHE
static struct starpu_htbl32_node_s **sent_data = NULL;
static struct starpu_htbl32_node_s **received_data = NULL;

static void _starpu_mpi_task_init(int nb_nodes) {
        int i;

        _STARPU_MPI_DEBUG("Initialising hash table for cache\n");
        sent_data = malloc(nb_nodes * sizeof(struct starpu_htbl32_node_s *));
        for(i=0 ; i<nb_nodes ; i++) sent_data[i] = NULL;
        received_data = malloc(nb_nodes * sizeof(struct starpu_htbl32_node_s *));
        for(i=0 ; i<nb_nodes ; i++) received_data[i] = NULL;
}
#endif

int starpu_mpi_insert_task(MPI_Comm comm, starpu_codelet *codelet, ...) {
        int arg_type;
        va_list varg_list;
        int me, do_execute;
	size_t arg_buffer_size = 0;
        int dest;

        _STARPU_MPI_LOG_IN();

	MPI_Comm_rank(comm, &me);

#ifdef MPI_CACHE
        if (sent_data == NULL) {
                int size;
                MPI_Comm_size(comm, &size);
                _starpu_mpi_task_init(size);
        }
#endif

        /* Get the number of buffers and the size of the arguments */
	va_start(varg_list, codelet);
        arg_buffer_size = starpu_insert_task_get_arg_size(varg_list);

	/* Find out whether we are to execute the data because we own the data to be written to. */
        do_execute = -1;
	va_start(varg_list, codelet);
	while ((arg_type = va_arg(varg_list, int)) != 0) {
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH) {
                        starpu_data_handle data = va_arg(varg_list, starpu_data_handle);
                        if (arg_type & STARPU_W) {
                                if (!data) {
                                        /* We don't have anything allocated for this.
                                         * The application knows we won't do anything
                                         * about this task */
                                        /* Yes, the app could actually not call
                                         * insert_task at all itself, this is just a
                                         * safeguard. */
                                        _STARPU_MPI_DEBUG("oh oh\n");
                                        _STARPU_MPI_LOG_OUT();
                                        return;
                                }
                                int mpi_rank = starpu_data_get_rank(data);
                                if (mpi_rank == me) {
                                        if (do_execute == 0) {
                                                _STARPU_ERROR("erh? incoherent!\n");
                                        }
                                        else {
                                                do_execute = 1;
                                        }
                                }
                                else if (mpi_rank != -1) {
                                        if (do_execute == 1) {
                                                _STARPU_ERROR("erh? incoherent!\n");
                                        }
                                        else {
                                                do_execute = 0;
                                                dest = mpi_rank;
                                                /* That's the rank which needs the data to be sent to */
                                        }
                                }
                        }
                }
		else if (arg_type==STARPU_VALUE) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK) {
			va_arg(varg_list, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY) {
			va_arg(varg_list, int);
		}
	}
	va_end(varg_list);
        assert(do_execute != -1);

        /* Send and receive data as requested */
	va_start(varg_list, codelet);
	while ((arg_type = va_arg(varg_list, int)) != 0) {
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH) {
                        starpu_data_handle data = va_arg(varg_list, starpu_data_handle);
                        if (arg_type & STARPU_R) {
                                int mpi_rank = starpu_data_get_rank(data);
                                /* The task needs to read this data */
                                if (do_execute && mpi_rank != me && mpi_rank != -1) {
                                        /* I will have to execute but I don't have the data, receive */
#ifdef MPI_CACHE
                                        uint32_t key = _starpu_crc32_be(data, 0);
                                        void *already_received = _starpu_htbl_search_32(received_data[mpi_rank], key);
                                        if (!already_received) {
                                                _starpu_htbl_insert_32(&received_data[mpi_rank], key, data);
                                        }
                                        else {
                                                _STARPU_MPI_DEBUG("Do not receive data %p from node %d as it is already available\n", data, mpi_rank);
                                        }
                                        if (!already_received)
                                                //if (!starpu_allocated(data))

#endif
                                                {
                                                        _STARPU_MPI_DEBUG("Receive data %p from %d\n", data, mpi_rank);
                                                        starpu_mpi_irecv_detached(data, mpi_rank, 0, comm, NULL, NULL);
                                                }
                                }
                                if (!do_execute && mpi_rank == me) {
                                        /* Somebody else will execute it, and I have the data, send it. */
                                        /* FIXME CACHE: we need to know whether the receiver has it. */
#ifdef MPI_CACHE
                                        uint32_t key = _starpu_crc32_be(data, 0);
                                        void *already_sent = _starpu_htbl_search_32(sent_data[dest], key);
                                        if (!already_sent) {
                                                _starpu_htbl_insert_32(&sent_data[dest], key, data);
                                        }
                                        else {
                                                _STARPU_MPI_DEBUG("Do not sent data %p to node %d as it has already been sent\n", data, dest);
                                        }
                                        if (!already_sent)
#endif
                                                {
                                                        _STARPU_MPI_DEBUG("Send data %p to %d\n", data, dest);
                                                        starpu_mpi_isend_detached(data, dest, 0, comm, NULL, NULL);
                                                }
                                }
                        }
                }
		else if (arg_type==STARPU_VALUE) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK) {
			va_arg(varg_list, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY) {
			va_arg(varg_list, int);
		}
        }
	va_end(varg_list);

	if (do_execute) {
                _STARPU_MPI_DEBUG("Execution of the codelet\n");
                va_start(varg_list, codelet);
                struct starpu_task *task = starpu_task_create();
                int ret = starpu_insert_task_create_and_submit(arg_buffer_size, codelet, &task, varg_list);
                _STARPU_MPI_DEBUG("ret: %d\n", ret);
                STARPU_ASSERT(ret==0);
        }

	/* No need to handle W, as we assume (and check) that task
	 * write in data that they own */

	va_start(varg_list, codelet);
	while ((arg_type = va_arg(varg_list, int)) != 0) {
		if (arg_type==STARPU_R || arg_type==STARPU_W || arg_type==STARPU_RW || arg_type == STARPU_SCRATCH) {
                        starpu_data_handle data = va_arg(varg_list, starpu_data_handle);
#ifdef MPI_CACHE
                        if (arg_type & STARPU_W) {
                                if (do_execute) {
                                        /* FIXME: I need to note that all
                                         * copies I've sent to neighbours are
                                         * now invalid */
                                }
                                else {
                                        int mpi_rank = starpu_data_get_rank(data);
                                        uint32_t key = _starpu_crc32_be(data, 0);
                                        void *already_received = _starpu_htbl_search_32(received_data[mpi_rank], key);
                                        if (already_received) {
                                        /* Somebody else will write to the data, so discard our cached copy if any */
                                        /* TODO: starpu_mpi could just remember itself. */
                                                _STARPU_MPI_DEBUG("Clear cache for data %p\n", data);
                                                //                                        if (starpu_data_allocated(data))
#warning do a submit deallocate
                                                //starpu_data_deallocate(data);
                                        }
                                }
                        }
#else
                        /* We allocated a temporary buffer for the received data, now drop it */
                        if ((arg_type & STARPU_R) && do_execute) {
                                int mpi_rank = starpu_data_get_rank(data);
                                if (mpi_rank != me && mpi_rank != -1) {
                                        //starpu_data_deallocate(data);
                                }
                        }
#endif
                }
		else if (arg_type==STARPU_VALUE) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK) {
			va_arg(varg_list, void (*)(void *));
		}
		else if (arg_type==STARPU_CALLBACK_ARG) {
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY) {
			va_arg(varg_list, int);
		}
        }
	va_end(varg_list);
        _STARPU_MPI_LOG_OUT();
}
