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

#ifndef __STARPU_MPI_PRIVATE_H__
#define __STARPU_MPI_PRIVATE_H__

#include <starpu.h>
#include <common/config.h>
#include <common/uthash.h>
#include <starpu_mpi.h>
#include <starpu_mpi_fxt.h>
#include <common/list.h>
#include <common/prio_list.h>
#include <common/starpu_spinlock.h>
#include <core/simgrid.h>

/** @file */

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_SIMGRID
extern starpu_pthread_wait_t _starpu_mpi_thread_wait;
extern starpu_pthread_queue_t _starpu_mpi_thread_dontsleep;

struct _starpu_simgrid_mpi_req
{
	MPI_Request *request;
	MPI_Status *status;
	starpu_pthread_queue_t *queue;
	unsigned *done;
};

int _starpu_mpi_simgrid_mpi_test(unsigned *done, int *flag);
void _starpu_mpi_simgrid_wait_req(MPI_Request *request, MPI_Status *status, starpu_pthread_queue_t *queue, unsigned *done);
#endif

extern int _starpu_debug_rank;
char *_starpu_mpi_get_mpi_error_code(int code);
extern int _starpu_mpi_comm_debug;

#ifdef STARPU_MPI_VERBOSE
extern int _starpu_debug_level_min;
extern int _starpu_debug_level_max;
void _starpu_mpi_set_debug_level_min(int level);
void _starpu_mpi_set_debug_level_max(int level);
#endif
extern int _starpu_mpi_fake_world_size;
extern int _starpu_mpi_fake_world_rank;
extern int _starpu_mpi_use_prio;
extern int _starpu_mpi_nobind;
extern int _starpu_mpi_thread_cpuid;
extern int _starpu_mpi_use_coop_sends;
void _starpu_mpi_env_init(void);

#ifdef STARPU_NO_ASSERT
#  define STARPU_MPI_ASSERT_MSG(x, msg, ...) do { if (0) { (void) (x); }} while(0)
#else
#  if defined(__CUDACC__) && defined(STARPU_HAVE_WINDOWS)
int _starpu_debug_rank;
#    define STARPU_MPI_ASSERT_MSG(x, msg, ...)									\
	do													\
	{ 													\
		if (STARPU_UNLIKELY(!(x))) 									\
		{												\
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			fprintf(stderr, "\n[%d][starpu_mpi][%s][assert failure] " msg "\n\n", _starpu_debug_rank, __starpu_func__, ## __VA_ARGS__); *(int*)NULL = 0; \
		} \
	} while(0)
#  else
#    define STARPU_MPI_ASSERT_MSG(x, msg, ...)	\
	do \
	{ \
		if (STARPU_UNLIKELY(!(x))) \
		{ \
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			fprintf(stderr, "\n[%d][starpu_mpi][%s][assert failure] " msg "\n\n", _starpu_debug_rank, __starpu_func__, ## __VA_ARGS__); \
		} \
		assert(x); \
	} while(0)

#  endif
#endif

#define _STARPU_MPI_MALLOC(ptr, size) do { ptr = malloc(size); STARPU_MPI_ASSERT_MSG(ptr != NULL, "Cannot allocate %ld bytes\n", (long) (size)); } while (0)
#define _STARPU_MPI_CALLOC(ptr, nmemb, size) do { ptr = calloc(nmemb, size); STARPU_MPI_ASSERT_MSG(ptr != NULL, "Cannot allocate %ld bytes\n", (long) (nmemb*size)); } while (0)
#define _STARPU_MPI_REALLOC(ptr, size) do { void *_new_ptr = realloc(ptr, size); STARPU_MPI_ASSERT_MSG(_new_ptr != NULL, "Cannot reallocate %ld bytes\n", (long) (size)); ptr = _new_ptr; } while (0)

#ifdef STARPU_MPI_VERBOSE
#  define _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, node, tag, utag, comm, way) \
	do \
	{ \
	     	if (_starpu_mpi_comm_debug) \
		{ \
     			int __size; \
			char _comm_name[128]; \
			int _comm_name_len; \
			int _rank; \
			starpu_mpi_comm_rank(comm, &_rank); \
			MPI_Type_size(datatype, &__size); \
			MPI_Comm_get_name(comm, _comm_name, &_comm_name_len); \
			fprintf(stderr, "[%d][starpu_mpi] :%d:%s:%d:%d:%ld:%s:%p:%ld:%d:%s:%d\n", _rank, _rank, way, node, tag, utag, _comm_name, ptr, count, __size, __starpu_func__ , __LINE__); \
			fflush(stderr);	\
		} \
	} while(0);
#  define _STARPU_MPI_COMM_TO_DEBUG(ptr, count, datatype, dest, tag, utag, comm) _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, dest, tag, utag, comm, "-->")
#  define _STARPU_MPI_COMM_FROM_DEBUG(ptr, count, datatype, source, tag, utag, comm)  _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, source, tag, utag, comm, "<--")
#  define _STARPU_MPI_DEBUG(level, fmt, ...) \
	do \
	{								\
		if (!_starpu_silent && _starpu_debug_level_min <= level && level <= _starpu_debug_level_max)	\
		{ \
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] " fmt , (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__,## __VA_ARGS__); \
			fflush(stderr); \
		} \
	} while(0);
#else
#  define _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, node, tag, utag, comm, way)  do { } while(0)
#  define _STARPU_MPI_COMM_TO_DEBUG(ptr, count, datatype, dest, tag, utag, comm)     do { } while(0)
#  define _STARPU_MPI_COMM_FROM_DEBUG(ptr, count, datatype, source, tag, utag, comm) do { } while(0)
#  define _STARPU_MPI_DEBUG(level, fmt, ...)		do { } while(0)
#endif

#define _STARPU_MPI_DISP(fmt, ...) do { if (!_starpu_silent) { \
	       				     if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
                                             fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] " fmt , (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__ ,## __VA_ARGS__); \
                                             fflush(stderr); }} while(0);
#define _STARPU_MPI_MSG(fmt, ...) do { if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
                                             fprintf(stderr, "[%d][starpu_mpi][%s:%d] " fmt , _starpu_debug_rank, __starpu_func__ , __LINE__ ,## __VA_ARGS__); \
                                             fflush(stderr); } while(0);

#ifdef STARPU_MPI_EXTRA_VERBOSE
#  define _STARPU_MPI_LOG_IN()             do { if (!_starpu_silent) { \
                                               if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank);                        \
                                               fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] -->\n", (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__); \
                                               fflush(stderr); }} while(0)
#  define _STARPU_MPI_LOG_OUT()            do { if (!_starpu_silent) { \
                                               if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank);                        \
                                               fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] <--\n", (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__, __LINE__ ); \
                                               fflush(stderr); }} while(0)
#else
#  define _STARPU_MPI_LOG_IN()
#  define _STARPU_MPI_LOG_OUT()
#endif

enum _starpu_mpi_request_type
{
	SEND_REQ=0,
	RECV_REQ=1,
	WAIT_REQ=2,
	TEST_REQ=3,
	BARRIER_REQ=4,
	PROBE_REQ=5,
	UNKNOWN_REQ=6,
};

struct _starpu_mpi_node
{
	MPI_Comm comm;
	int rank;
};

struct _starpu_mpi_node_tag
{
	struct _starpu_mpi_node node;
	starpu_mpi_tag_t data_tag;
};

MULTILIST_CREATE_TYPE(_starpu_mpi_req, coop_sends)
/* One bag of cooperative sends */
struct _starpu_mpi_coop_sends
{
	/* List of send requests */
	struct _starpu_mpi_req_multilist_coop_sends reqs;
	struct _starpu_mpi_data *mpi_data;

	/* Array of send requests, after sorting out */
	struct _starpu_spinlock lock;
	struct _starpu_mpi_req **reqs_array;
	unsigned n;
	unsigned redirects_sent;
};

/* Initialized in starpu_mpi_data_register_comm */
struct _starpu_mpi_data
{
	int magic;
	struct _starpu_mpi_node_tag node_tag;
	char *cache_sent;
	int cache_received;

	/* Rendez-vous data for opportunistic cooperative sends */
	struct _starpu_spinlock coop_lock; /* Needed to synchronize between submit thread and workers */
	struct _starpu_mpi_coop_sends *coop_sends; /* Current cooperative send bag */
};

struct _starpu_mpi_data *_starpu_mpi_data_get(starpu_data_handle_t data_handle);

struct _starpu_mpi_req_backend;
struct _starpu_mpi_req;
LIST_TYPE(_starpu_mpi_req,
	/* description of the data at StarPU level */
	starpu_data_handle_t data_handle;

	int prio;

	/* description of the data to be sent/received */
	MPI_Datatype datatype;
	char *datatype_name;
	void *ptr;
	starpu_ssize_t count;
	int registered_datatype;

	struct _starpu_mpi_req_backend *backend;

	/* who are we talking to ? */
	struct _starpu_mpi_node_tag node_tag;
	void (*func)(struct _starpu_mpi_req *);

	MPI_Status *status;
	struct _starpu_mpi_req_multilist_coop_sends coop_sends;
	struct _starpu_mpi_coop_sends *coop_sends_head;

	int *flag;
	unsigned sync;

	int ret;

	/** 0 send, 1 recv */
	enum _starpu_mpi_request_type request_type;

	unsigned submitted;
	unsigned completed;
	unsigned posted;

	/* in the case of detached requests */
	int detached;
	void *callback_arg;
	void (*callback)(void *);

        /* in the case of user-defined datatypes, we need to send the size of the data */

	int sequential_consistency;

	long pre_sync_jobid;
	long post_sync_jobid;

#ifdef STARPU_SIMGRID
        MPI_Status status_store;
	starpu_pthread_queue_t queue;
	unsigned done;
#endif
);
PRIO_LIST_TYPE(_starpu_mpi_req, prio)

MULTILIST_CREATE_INLINES(struct _starpu_mpi_req, _starpu_mpi_req, coop_sends)

/* To be called before actually queueing a request, so the communication layer knows it has something to look at */
void _starpu_mpi_req_willpost(struct _starpu_mpi_req *req);
/* To be called to actually submit the request */
void _starpu_mpi_submit_ready_request(void *arg);
/* To be called when request is completed */
void _starpu_mpi_release_req_data(struct _starpu_mpi_req *req);

/* Build a communication tree. Called before _starpu_mpi_coop_send is ever called. coop_sends->lock is held. */
void _starpu_mpi_coop_sends_build_tree(struct _starpu_mpi_coop_sends *coop_sends);
/* Try to merge with send request with other send requests */
void _starpu_mpi_coop_send(starpu_data_handle_t data_handle, struct _starpu_mpi_req *req, enum starpu_data_access_mode mode, int sequential_consistency);

/* Actually submit the coop_sends bag to MPI.
 * At least one of submit_control or submit_data is true.
 * _starpu_mpi_submit_coop_sends may be called either
 * - just once with both parameters being true,
 * - or once with submit_control being true (data is not available yet, but we
 * can send control messages), and a second time with submit_data being true. Or
 * the converse, possibly on different threads, etc.
 */
void _starpu_mpi_submit_coop_sends(struct _starpu_mpi_coop_sends *coop_sends, int submit_control, int submit_data);

void _starpu_mpi_submit_ready_request_inc(struct _starpu_mpi_req *req);
void _starpu_mpi_request_init(struct _starpu_mpi_req **req);
struct _starpu_mpi_req * _starpu_mpi_request_fill(starpu_data_handle_t data_handle,
						  int srcdst, starpu_mpi_tag_t data_tag, MPI_Comm comm,
						  unsigned detached, unsigned sync, int prio, void (*callback)(void *), void *arg,
						  enum _starpu_mpi_request_type request_type, void (*func)(struct _starpu_mpi_req *),
						  int sequential_consistency,
						  int is_internal_req,
						  starpu_ssize_t count);

void _starpu_mpi_request_destroy(struct _starpu_mpi_req *req);
void _starpu_mpi_isend_size_func(struct _starpu_mpi_req *req);
void _starpu_mpi_irecv_size_func(struct _starpu_mpi_req *req);
int _starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status);
int _starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status);
int _starpu_mpi_barrier(MPI_Comm comm);

struct _starpu_mpi_argc_argv
{
	int initialize_mpi;
	int *argc;
	char ***argv;
	MPI_Comm comm;
	/** Fortran argc */
	int fargc;
	/** Fortran argv */
	char **fargv;
	int rank;
	int world_size;
};

void _starpu_mpi_progress_shutdown(void **value);
int _starpu_mpi_progress_init(struct _starpu_mpi_argc_argv *argc_argv);
#ifdef STARPU_SIMGRID
void _starpu_mpi_wait_for_initialization();
#endif
void _starpu_mpi_data_flush(starpu_data_handle_t data_handle);

/*
 * Specific functions to backend implementation
 */
struct _starpu_mpi_backend
{
	void (*_starpu_mpi_backend_init)(struct starpu_conf *conf);
	void (*_starpu_mpi_backend_shutdown)(void);
	int (*_starpu_mpi_backend_reserve_core)(void);
	void (*_starpu_mpi_backend_request_init)(struct _starpu_mpi_req *req);
	void (*_starpu_mpi_backend_request_fill)(struct _starpu_mpi_req *req, MPI_Comm comm, int is_internal_req);
	void (*_starpu_mpi_backend_request_destroy)(struct _starpu_mpi_req *req);
	void (*_starpu_mpi_backend_data_clear)(starpu_data_handle_t data_handle);
	void (*_starpu_mpi_backend_data_register)(starpu_data_handle_t data_handle, starpu_mpi_tag_t data_tag);
	void (*_starpu_mpi_backend_comm_register)(MPI_Comm comm);
};

extern struct _starpu_mpi_backend _mpi_backend;
#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_PRIVATE_H__
