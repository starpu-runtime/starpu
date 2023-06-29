/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef FT_STARPU_STARPU_MPI_FT_STATS_H
#define FT_STARPU_STARPU_MPI_FT_STATS_H

#include <common/list.h>
#include <common/utils.h>

#ifdef STARPU_USE_MPI_FT

#ifdef __cplusplus
extern "C"
{
#endif

extern starpu_pthread_mutex_t _ft_stats_mutex;

extern int cp_data_msgs_sent_count;
extern size_t cp_data_msgs_sent_total_size;
extern int cp_data_msgs_received_count;
extern size_t cp_data_msgs_received_total_size;

extern int cp_data_msgs_sent_cached_count;
extern size_t cp_data_msgs_sent_cached_total_size;
extern int cp_data_msgs_received_cached_count;
extern size_t cp_data_msgs_received_cached_total_size;
extern int cp_data_msgs_received_cp_cached_count;
extern size_t cp_data_msgs_received_cp_cached_total_size;

extern int ft_service_msgs_sent_count;
extern size_t ft_service_msgs_sent_total_size;
extern int ft_service_msgs_received_count;
extern size_t ft_service_msgs_received_total_size;

extern struct size_sample_list cp_data_in_memory_list; //over time
extern size_t cp_data_in_memory_size_total;
extern size_t cp_data_in_memory_size_max_at_t;

static inline void stat_init();
static inline void _starpu_ft_stats_shutdown();
static inline void _starpu_ft_stats_write_to_fd();
static inline void _starpu_ft_stats_send_data(size_t size);
static inline void _starpu_ft_stats_send_data_cached(size_t size);;
static inline void _starpu_ft_stats_recv_data(size_t size);
static inline void _starpu_ft_stats_recv_data_cached(size_t size);
static inline void _starpu_ft_stats_recv_data_cp_cached(size_t size);
static inline void _starpu_ft_stats_service_msg_send(size_t size);
static inline void _starpu_ft_stats_service_msg_recv(size_t size);
static inline void _starpu_ft_stats_add_cp_data_in_memory(size_t size);
static inline void _starpu_ft_stats_free_cp_data_in_memory(size_t size);

#ifdef STARPU_USE_MPI_FT_STATS
#define _STARPU_MPI_FT_STATS_INIT() do{ stat_init(); }while(0)
#define _STARPU_MPI_FT_STATS_SHUTDOWN() do{ _starpu_ft_stats_shutdown(); }while(0)
#define _STARPU_MPI_FT_STATS_WRITE_TO_FD(fd) do{ _starpu_ft_stats_write_to_fd(fd); }while(0)
#define _STARPU_MPI_FT_STATS_SEND_CP_DATA(size) do{ _starpu_ft_stats_send_data(size); }while(0)
#define _STARPU_MPI_FT_STATS_CANCEL_SEND_CP_DATA(size) do{ _starpu_ft_stats_cancel_send_data(size); }while(0)
#define _STARPU_MPI_FT_STATS_SEND_CACHED_CP_DATA(size) do{ _starpu_ft_stats_send_data_cached(size); }while(0)
#define _STARPU_MPI_FT_STATS_RECV_CP_DATA(size) do{ _starpu_ft_stats_recv_data(size); }while(0)
#define _STARPU_MPI_FT_STATS_CANCEL_RECV_CP_DATA(size) do{ _starpu_ft_stats_cancel_recv_data(size); }while(0)
#define _STARPU_MPI_FT_STATS_RECV_CACHED_CP_DATA(size) do{ _starpu_ft_stats_recv_data_cached(size); }while(0)
#define _STARPU_MPI_FT_STATS_RECV_CP_CACHED_CP_DATA(size) do{ _starpu_ft_stats_recv_data_cp_cached(size); }while(0)
#define _STARPU_MPI_FT_STATS_SEND_FT_SERVICE_MSG(size) do{ _starpu_ft_stats_service_msg_send(size); }while(0)
#define _STARPU_MPI_FT_STATS_RECV_FT_SERVICE_MSG(size) do{ _starpu_ft_stats_service_msg_recv(size); }while(0)
#define _STARPU_MPI_FT_STATS_STORE_CP_DATA(size) do{ _starpu_ft_stats_add_cp_data_in_memory(size); }while(0)
#define _STARPU_MPI_FT_STATS_DISCARD_CP_DATA(size) do{ _starpu_ft_stats_free_cp_data_in_memory(size); }while(0)

#else //_STARPU_MPI_FT_STATS
#define _STARPU_MPI_FT_STATS_INIT() do{}while(0)
#define _STARPU_MPI_FT_STATS_SHUTDOWN() do{}while(0)
#define _STARPU_MPI_FT_STATS_WRITE_TO_FD(fd) do{}while(0)
#define _STARPU_MPI_FT_STATS_SEND_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_CANCEL_SEND_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_SEND_CACHED_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_RECV_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_CANCEL_RECV_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_RECV_CACHED_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_RECV_CP_CACHED_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_SEND_FT_SERVICE_MSG(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_RECV_FT_SERVICE_MSG(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_STORE_CP_DATA(size) do{}while(0)
#define _STARPU_MPI_FT_STATS_DISCARD_CP_DATA(size) do{}while(0)

#endif //_STARPU_MPI_FT_STATS

LIST_TYPE(size_sample, \
	  size_t size;
)

static inline void stat_init()
{
	STARPU_PTHREAD_MUTEX_INIT(&_ft_stats_mutex, NULL);
	size_sample_list_init(&cp_data_in_memory_list);
	cp_data_msgs_sent_count = 0;
	cp_data_msgs_sent_total_size = 0;
	cp_data_msgs_received_count = 0;
	cp_data_msgs_received_total_size = 0;

	cp_data_msgs_sent_cached_count = 0;
	cp_data_msgs_sent_cached_total_size = 0;
	cp_data_msgs_received_cached_count = 0;
	cp_data_msgs_received_cached_total_size = 0;
	cp_data_msgs_received_cp_cached_count = 0;
	cp_data_msgs_received_cp_cached_total_size = 0;

	ft_service_msgs_sent_count = 0;
	ft_service_msgs_sent_total_size = 0;
	ft_service_msgs_received_count = 0;
	ft_service_msgs_received_total_size = 0;

	cp_data_in_memory_size_total = 0;
	cp_data_in_memory_size_max_at_t = 0;
}

static inline void _starpu_ft_stats_send_data(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_msgs_sent_count++;
	cp_data_msgs_sent_total_size+=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_cancel_send_data(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_msgs_sent_count--;
	cp_data_msgs_sent_total_size-=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_send_data_cached(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_msgs_sent_cached_count++;
	cp_data_msgs_sent_cached_total_size+=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_recv_data(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_msgs_received_count++;
	cp_data_msgs_received_total_size+=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_cancel_recv_data(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_msgs_received_count--;
	cp_data_msgs_received_total_size-=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_recv_data_cached(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_msgs_received_cached_count++;
	cp_data_msgs_received_cached_total_size+=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_recv_data_cp_cached(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_msgs_received_cp_cached_count++;
	cp_data_msgs_received_cp_cached_total_size+=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_service_msg_send(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	ft_service_msgs_sent_count++;
	ft_service_msgs_sent_total_size+=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_service_msg_recv(size_t size)
{
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	ft_service_msgs_received_count++;
	ft_service_msgs_received_total_size+=size;
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_add_cp_data_in_memory(size_t size)
{
	size_t tmp;
	struct size_sample *tmp_sample, *sample = malloc(sizeof(struct size_sample));
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	cp_data_in_memory_size_total+=size;
	tmp_sample = size_sample_list_back(&cp_data_in_memory_list);
	tmp = (NULL==tmp_sample?0:tmp_sample->size);
	tmp+=size;
	if (tmp>cp_data_in_memory_size_max_at_t)
	{
		cp_data_in_memory_size_max_at_t = tmp;
	}
	sample->size = tmp;
	size_sample_list_push_back(&cp_data_in_memory_list, sample);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _starpu_ft_stats_free_cp_data_in_memory(size_t size)
{
	size_t tmp;
	struct size_sample* sample = malloc(sizeof(struct size_sample));
	STARPU_ASSERT_MSG((int)size != -1, "Cannot count a data of size -1. An error has occurred.\n");
	STARPU_PTHREAD_MUTEX_LOCK(&_ft_stats_mutex);
	tmp = size_sample_list_back(&cp_data_in_memory_list)->size;
	tmp-=size;
	sample->size = tmp;
	size_sample_list_push_back(&cp_data_in_memory_list, sample);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_ft_stats_mutex);
}

static inline void _ft_stats_free_cp_data_in_memory_list()
{
	struct size_sample *next, *sample = size_sample_list_begin(&cp_data_in_memory_list);
	while (sample != size_sample_list_end(&cp_data_in_memory_list))
	{
		next = size_sample_list_next(sample);
		size_sample_list_erase(&cp_data_in_memory_list, sample);
		free(sample);
		sample = next;
	}
}

static inline void _starpu_ft_stats_write_to_fd(FILE* fd)
{
	// HEADER
	fprintf(fd, "TYPE\tCP_DATA_NORMAL_COUNT\tCP_DATA_NORMAL_TOTAL_SIZE\tCP_DATA_CACHED_COUNT\tCP_DATA_CACHED_SIZE\tFT_SERVICE_MSGS_COUNT\tFT_SERVICE_MSGS_TOTAL_SIZE\n");
	// DATA
	fprintf(fd, "SEND\t%d\t"                 "%ld\t"                    "%d\t"               "%ld\t"               "%d\t"                 "%ld\n",
	        cp_data_msgs_sent_count, cp_data_msgs_sent_total_size, cp_data_msgs_sent_cached_count, cp_data_msgs_sent_cached_total_size, ft_service_msgs_sent_count, ft_service_msgs_sent_total_size);
	fprintf(fd, "RECV\t%d\t"                 "%ld\t"                    "%d\t"               "%ld\t"               "%d\t"                 "%ld\n",
	        cp_data_msgs_received_count, cp_data_msgs_received_total_size, cp_data_msgs_received_cached_count, cp_data_msgs_received_cached_total_size+cp_data_msgs_received_cp_cached_total_size, ft_service_msgs_received_count, ft_service_msgs_received_total_size);
	fprintf(fd, "\n");
	fprintf(fd, "IN_MEM_CP_DATA_TOTAL:%lu\n", cp_data_in_memory_size_total);
	fprintf(fd, "\n");
	fprintf(fd, "IN_MEM_CP_DATA_MAX_AT_T:%lu\n", cp_data_in_memory_size_max_at_t);
	fprintf(fd, "\n");
//	fprintf(fd, "IN_MEM_CP_DATA_TRACKING\n");
//	struct size_sample *sample = size_sample_list_begin(&cp_data_in_memory_list);
//	while (sample != size_sample_list_end(&cp_data_in_memory_list))
//	{
//		fprintf(fd, "%ld\n", sample->size);
//		sample = size_sample_list_next(sample);
//	}
//	fprintf(fd, "\n");
}

static inline void _starpu_ft_stats_shutdown()
{
	_ft_stats_free_cp_data_in_memory_list();
}

#ifdef __cplusplus
}
#endif

#endif // STARPU_USE_MPI_FT
#endif //FT_STARPU_STARPU_MPI_FT_STATS_H
