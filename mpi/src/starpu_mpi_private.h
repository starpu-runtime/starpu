/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013,2016,2017                           Inria
 * Copyright (C) 2010-2017, 2019                          CNRS
 * Copyright (C) 2010-2019                                Université de Bordeaux
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
#if defined(STARPU_USE_MPI_NMAD)
#include <pioman.h>
#include <nm_sendrecv_interface.h>
#include <nm_session_interface.h>
#endif

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
void _starpu_mpi_simgrid_wait_req(MPI_Request *request, 	MPI_Status *status, starpu_pthread_queue_t *queue, unsigned *done);
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
extern int _starpu_mpi_thread_cpuid;
extern int _starpu_mpi_use_coop_sends;
void _starpu_mpi_env_init(void);

#ifdef STARPU_NO_ASSERT
#  define STARPU_MPI_ASSERT_MSG(x, msg, ...)	do { if (0) { (void) (x); }} while(0)
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
	do								\
	{							\
	     	if (_starpu_mpi_comm_debug)			\
		{					\
     			int __size;			\
			char _comm_name[128];		\
			int _comm_name_len;		\
			int _rank;			    \
			starpu_mpi_comm_rank(comm, &_rank); \
			MPI_Type_size(datatype, &__size);		\
			MPI_Comm_get_name(comm, _comm_name, &_comm_name_len); \
			fprintf(stderr, "[%d][starpu_mpi] :%d:%s:%d:%d:%ld:%s:%p:%ld:%d:%s:%d\n", _rank, _rank, way, node, tag, utag, _comm_name, ptr, count, __size, __starpu_func__ , __LINE__); \
			fflush(stderr);					\
		}							\
	} while(0);
#  define _STARPU_MPI_COMM_TO_DEBUG(ptr, count, datatype, dest, tag, utag, comm) 	    _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, dest, tag, utag, comm, "-->")
#  define _STARPU_MPI_COMM_FROM_DEBUG(ptr, count, datatype, source, tag, utag, comm)  _STARPU_MPI_COMM_DEBUG(ptr, count, datatype, source, tag, utag, comm, "<--")
#  define _STARPU_MPI_DEBUG(level, fmt, ...) \
	do \
	{								\
		if (!_starpu_silent && _starpu_debug_level_min <= level && level <= _starpu_debug_level_max)	\
		{							\
			if (_starpu_debug_rank == -1) starpu_mpi_comm_rank(MPI_COMM_WORLD, &_starpu_debug_rank); \
			fprintf(stderr, "%*s[%d][starpu_mpi][%s:%d] " fmt , (_starpu_debug_rank+1)*4, "", _starpu_debug_rank, __starpu_func__ , __LINE__,## __VA_ARGS__); \
			fflush(stderr); \
		}			\
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

#if defined(STARPU_USE_MPI_MPI)
extern int _starpu_mpi_tag;
#define _STARPU_MPI_TAG_ENVELOPE  _starpu_mpi_tag
#define _STARPU_MPI_TAG_DATA      _starpu_mpi_tag+1
#define _STARPU_MPI_TAG_SYNC_DATA _starpu_mpi_tag+2

enum _starpu_envelope_mode
{
	_STARPU_MPI_ENVELOPE_DATA=0,
	_STARPU_MPI_ENVELOPE_SYNC_READY=1
};

struct _starpu_mpi_envelope
{
	enum _starpu_envelope_mode mode;
	starpu_ssize_t size;
	starpu_mpi_tag_t data_tag;
	unsigned sync;
};
#endif /* STARPU_USE_MPI_MPI */

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

struct _starpu_mpi_node_tag
{
	MPI_Comm comm;
	int rank;
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
	int *cache_sent;
	int cache_received;

	/* Rendez-vous data for opportunistic cooperative sends */
	struct _starpu_spinlock coop_lock; /* Needed to synchronize between submit thread and workers */
	struct _starpu_mpi_coop_sends *coop_sends; /* Current cooperative send bag */
};

struct _starpu_mpi_data *_starpu_mpi_data_get(starpu_data_handle_t data_handle);

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

	/* who are we talking to ? */
	struct _starpu_mpi_node_tag node_tag;
#if defined(STARPU_USE_MPI_NMAD)
	nm_gate_t gate;
	nm_session_t session;
#endif

	void (*func)(struct _starpu_mpi_req *);

	MPI_Status *status;
#if defined(STARPU_USE_MPI_NMAD)
	nm_sr_request_t data_request;
	int waited;
#elif defined(STARPU_USE_MPI_MPI)
	MPI_Request data_request;
#endif
	struct _starpu_mpi_req_multilist_coop_sends coop_sends;
	struct _starpu_mpi_coop_sends *coop_sends_head;

	int *flag;
	unsigned sync;

	int ret;
#if defined(STARPU_USE_MPI_NMAD)
	piom_cond_t req_cond;
#elif defined(STARPU_USE_MPI_MPI)
	starpu_pthread_mutex_t req_mutex;
	starpu_pthread_cond_t req_cond;
	starpu_pthread_mutex_t posted_mutex;
	starpu_pthread_cond_t posted_cond;
	/* In the case of a Wait/Test request, we are going to post a request
	 * to test the completion of another request */
	struct _starpu_mpi_req *other_request;
#endif

	enum _starpu_mpi_request_type request_type; /* 0 send, 1 recv */

	unsigned submitted;
	unsigned completed;
	unsigned posted;

	/* in the case of detached requests */
	int detached;
	void *callback_arg;
	void (*callback)(void *);

        /* in the case of user-defined datatypes, we need to send the size of the data */
#if defined(STARPU_USE_MPI_NMAD)
	nm_sr_request_t size_req;
#elif defined(STARPU_USE_MPI_MPI)
	MPI_Request size_req;
#endif

#if defined(STARPU_USE_MPI_MPI)
	struct _starpu_mpi_envelope* envelope;

	unsigned is_internal_req:1;
	unsigned to_destroy:1;
	struct _starpu_mpi_req *internal_req;
	struct _starpu_mpi_early_data_handle *early_data_handle;
     	UT_hash_handle hh;
#endif

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
	int fargc;	// Fortran argc
	char **fargv;	// Fortran argv
	int rank;
	int world_size;
};

void _starpu_mpi_progress_shutdown(void **value);
int _starpu_mpi_progress_init(struct _starpu_mpi_argc_argv *argc_argv);
#ifdef STARPU_SIMGRID
void _starpu_mpi_wait_for_initialization();
#endif
void _starpu_mpi_data_flush(starpu_data_handle_t data_handle);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_MPI_PRIVATE_H__
