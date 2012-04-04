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

#include <starpu.h>
#include <pthread.h>

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

pthread_mutex_t act_hypervisor_mutex;

#define MAX_IDLE_TIME 5000000000
#define MIN_WORKING_TIME 500

struct policy_config {
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
};


struct resize_ack{
	int receiver_sched_ctx;
	int *moved_workers;
	int nmoved_workers;
	int *acked_workers;
};

struct sched_ctx_wrapper {
	unsigned sched_ctx;
	struct policy_config *config;
	double current_idle_time[STARPU_NMAXWORKERS];
	int pushed_tasks[STARPU_NMAXWORKERS];
	int poped_tasks[STARPU_NMAXWORKERS];
	double total_flops;
	double total_elapsed_flops[STARPU_NMAXWORKERS];
	double elapsed_flops[STARPU_NMAXWORKERS];
	double remaining_flops;
	double start_time;
	struct resize_ack resize_ack;
};

struct hypervisor_policy {
	const char* name;
	unsigned custom;
	void (*handle_idle_cycle)(unsigned sched_ctx, int worker);
	void (*handle_pushed_task)(unsigned sched_ctx, int worker);
	void (*handle_poped_task)(unsigned sched_ctx, int worker);
	void (*handle_idle_end)(unsigned sched_ctx, int worker);
	void (*handle_post_exec_hook)(unsigned sched_ctx, struct starpu_htbl32_node* resize_requests, int task_tag);
};


struct starpu_performance_counters* sched_ctx_hypervisor_init(struct hypervisor_policy* policy);

void sched_ctx_hypervisor_shutdown(void);

void sched_ctx_hypervisor_register_ctx(unsigned sched_ctx, double total_flops);

void sched_ctx_hypervisor_unregister_ctx(unsigned sched_ctx);

void sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag);

void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receier_sched_ctx, int *workers_to_move, unsigned nworkers_to_move);

void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_start_resize(unsigned sched_ctx);

void sched_ctx_hypervisor_ioctl(unsigned sched_ctx, ...);

void sched_ctx_hypervisor_set_config(unsigned sched_ctx, void *config);

struct policy_config* sched_ctx_hypervisor_get_config(unsigned sched_ctx);

int* sched_ctx_hypervisor_get_sched_ctxs();

int sched_ctx_hypervisor_get_nsched_ctxs();

struct sched_ctx_wrapper* sched_ctx_hypervisor_get_wrapper(unsigned sched_ctx);

double sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(struct sched_ctx_wrapper* sc_w);

char* sched_ctx_hypervisor_get_policy();
