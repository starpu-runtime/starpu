/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2012  INRIA
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

#ifndef SCHED_CTX_HYPERVISOR_H
#define SCHED_CTX_HYPERVISOR_H

#include <starpu.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef STARPU_DEVEL
#  warning rename all objects to start with sched_ctx_hypervisor
#endif

/* ioctl properties*/
#define HYPERVISOR_MAX_IDLE -1
#define HYPERVISOR_MIN_WORKING -2
#define HYPERVISOR_PRIORITY -3
#define HYPERVISOR_MIN_WORKERS -4
#define HYPERVISOR_MAX_WORKERS -5
#define HYPERVISOR_GRANULARITY -6
#define HYPERVISOR_FIXED_WORKERS -7
#define HYPERVISOR_MIN_TASKS -8
#define HYPERVISOR_NEW_WORKERS_MAX_IDLE -9
#define HYPERVISOR_TIME_TO_APPLY -10
#define HYPERVISOR_EMPTY_CTX_MAX_IDLE -11
#define HYPERVISOR_NULL -12
#define	HYPERVISOR_ISPEED_W_SAMPLE -13
#define HYPERVISOR_ISPEED_CTX_SAMPLE -14

pthread_mutex_t act_hypervisor_mutex;

#define MAX_IDLE_TIME 5000000000
#define MIN_WORKING_TIME 500

struct sched_ctx_hypervisor_policy_config
{
	/* underneath this limit we cannot resize */
	int min_nworkers;

	/* above this limit we cannot resize */
	int max_nworkers;

	/*resize granularity */
	int granularity;

	/* priority for a worker to stay in this context */
	/* the smaller the priority the faster it will be moved */
	/* to another context */
	int priority[STARPU_NMAXWORKERS];

	/* above this limit the priority of the worker is reduced */
	double max_idle[STARPU_NMAXWORKERS];

	/* underneath this limit the priority of the worker is reduced */
	double min_working[STARPU_NMAXWORKERS];

	/* workers that will not move */
	int fixed_workers[STARPU_NMAXWORKERS];

	/* max idle for the workers that will be added during the resizing process*/
	double new_workers_max_idle;

	/* above this context we allow removing all workers */
	double empty_ctx_max_idle[STARPU_NMAXWORKERS];

	/* sample used to compute the instant speed per worker*/
	double ispeed_w_sample[STARPU_NMAXWORKERS];

	/* sample used to compute the instant speed per ctx*/
	double ispeed_ctx_sample;

};

struct sched_ctx_hypervisor_resize_ack
{
	int receiver_sched_ctx;
	int *moved_workers;
	int nmoved_workers;
	int *acked_workers;
};

/* wrapper attached to a sched_ctx storing monitoring information */
struct sched_ctx_hypervisor_wrapper
{
	/* the sched_ctx it monitors */
	unsigned sched_ctx;

	/* user configuration meant to limit resizing */
	struct sched_ctx_hypervisor_policy_config *config;

	/* idle time of workers in this context */
	double current_idle_time[STARPU_NMAXWORKERS];
	
	/* list of workers that will leave this contexts (lazy resizing process) */
	int worker_to_be_removed[STARPU_NMAXWORKERS];

	/* number of tasks pushed on each worker in this ctx */
	int pushed_tasks[STARPU_NMAXWORKERS];

	/* number of tasks poped from each worker in this ctx */
	int poped_tasks[STARPU_NMAXWORKERS];

	/* number of flops the context has to execute */
	double total_flops;

	/* number of flops executed since the biginning until now */
	double total_elapsed_flops[STARPU_NMAXWORKERS];

	/* number of flops executed since last resizing */
	double elapsed_flops[STARPU_NMAXWORKERS];

	/* data quantity executed on each worker in this ctx */
	size_t elapsed_data[STARPU_NMAXWORKERS];

	/* nr of tasks executed on each worker in this ctx */
	int elapsed_tasks[STARPU_NMAXWORKERS];

	/* the average speed of workers when they belonged to this context */
	double ref_velocity[STARPU_NMAXWORKERS];

	/* number of flops submitted to this ctx */
	double submitted_flops;

	/* number of flops that still have to be executed in this ctx */
	double remaining_flops;
	
	/* the start time of the resizing sample of this context*/
	double start_time;

	/* the workers don't leave the current ctx until the receiver ctx 
	   doesn't ack the receive of these workers */
	struct sched_ctx_hypervisor_resize_ack resize_ack;

	/* mutex to protect the ack of workers */
	pthread_mutex_t mutex;
};

/* Forward declaration of an internal data structure
 * FIXME: Remove when no longer exposed.  */
struct resize_request_entry;

struct sched_ctx_hypervisor_policy
{
	const char* name;
	unsigned custom;
	void (*size_ctxs)(int *sched_ctxs, int nsched_ctxs , int *workers, int nworkers);
	void (*handle_idle_cycle)(unsigned sched_ctx, int worker);
	void (*handle_pushed_task)(unsigned sched_ctx, int worker);
	void (*handle_poped_task)(unsigned sched_ctx, int worker);
	void (*handle_idle_end)(unsigned sched_ctx, int worker);

	void (*handle_post_exec_hook)(unsigned sched_ctx, int task_tag);

	void (*handle_submitted_job)(struct starpu_task *task, unsigned footprint);
	
	void (*end_ctx)(unsigned sched_ctx);
};

struct starpu_sched_ctx_performance_counters *sched_ctx_hypervisor_init(struct sched_ctx_hypervisor_policy *policy);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_register_ctx(unsigned sched_ctx, double total_flops);

void sched_ctx_hypervisor_unregister_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag);

void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int *workers_to_move, unsigned nworkers_to_move, unsigned now);

void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_start_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_ioctl(unsigned sched_ctx, ...);

void sched_ctx_hypervisor_set_config(unsigned sched_ctx, void *config);

struct sched_ctx_hypervisor_policy_config *sched_ctx_hypervisor_get_config(unsigned sched_ctx);

int *sched_ctx_hypervisor_get_sched_ctxs();

int sched_ctx_hypervisor_get_nsched_ctxs();

int sched_ctx_hypervisor_get_nworkers_ctx(unsigned sched_ctx, enum starpu_archtype arch);

struct sched_ctx_hypervisor_wrapper *sched_ctx_hypervisor_get_wrapper(unsigned sched_ctx);

double sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(struct sched_ctx_hypervisor_wrapper *sc_w);

double sched_ctx_hypervisor_get_total_elapsed_flops_per_sched_ctx(struct sched_ctx_hypervisor_wrapper* sc_w);

const char *sched_ctx_hypervisor_get_policy();

void sched_ctx_hypervisor_add_workers_to_sched_ctx(int* workers_to_add, unsigned nworkers_to_add, unsigned sched_ctx);

void sched_ctx_hypervisor_remove_workers_from_sched_ctx(int* workers_to_remove, unsigned nworkers_to_remove, unsigned sched_ctx, unsigned now);

void sched_ctx_hypervisor_size_ctxs(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

unsigned sched_ctx_hypervisor_get_size_req(int **sched_ctxs, int* nsched_ctxs, int **workers, int *nworkers);

void sched_ctx_hypervisor_save_size_req(int *sched_ctxs, int nsched_ctxs, int *workers, int nworkers);

void sched_ctx_hypervisor_free_size_req(void);

unsigned sched_ctx_hypervisor_can_resize(unsigned sched_ctx);

#ifdef __cplusplus
}
#endif

#endif
