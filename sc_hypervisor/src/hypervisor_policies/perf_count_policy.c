/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "sc_hypervisor_policy.h"
#include <stdio.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <linux/kernel.h>
#include <omp.h>
#include <sys/ioctl.h>

struct perf_event_attr pe_instr[STARPU_NMAXWORKERS];
/* struct perf_event_attr pe_cycles[STARPU_NMAXWORKERS]; */
/* struct perf_event_attr pe_cache_misses[STARPU_NMAXWORKERS]; */
/* struct perf_event_attr pe_cache_refs[STARPU_NMAXWORKERS]; */
/* struct perf_event_attr pe_branch_instr[STARPU_NMAXWORKERS]; */
struct perf_event_attr pe_fps[STARPU_NMAXWORKERS];

int fd_instr[STARPU_NMAXWORKERS];
/* int fd_cycles[STARPU_NMAXWORKERS]; */
/* int fd_cache_misses[STARPU_NMAXWORKERS]; */
/* int fd_cache_refs[STARPU_NMAXWORKERS]; */
/* int fd_branch_instr[STARPU_NMAXWORKERS]; */
int fd_fps[STARPU_NMAXWORKERS];
unsigned perf_event_opened[STARPU_NMAXWORKERS];

long long total_instr[STARPU_NMAX_SCHED_CTXS];
/* long long total_cycles[STARPU_NMAX_SCHED_CTXS]; */
/* long long total_time[STARPU_NMAX_SCHED_CTXS]; */

/* long long total_cache_misses[STARPU_NMAX_SCHED_CTXS]; */
/* long long total_cache_refs[STARPU_NMAX_SCHED_CTXS]; */

/* long long total_branch_instr[STARPU_NMAX_SCHED_CTXS]; */
long long total_fps[STARPU_NMAX_SCHED_CTXS];

struct read_format
{
	uint64_t value;         /* The value of the event */
	uint64_t time_enabled;  /* if PERF_FORMAT_TOTAL_TIME_ENABLED */
	uint64_t time_running;  /* if PERF_FORMAT_TOTAL_TIME_RUNNING */
	uint64_t id;            /* if PERF_FORMAT_ID */
};

static long perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu,
                            int group_fd, unsigned long flags)
{
        int ret = syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
        return ret;
}

void print_results_for_worker(int workerid, unsigned sched_ctx, struct starpu_task *task)
{
	ssize_t rread;
	long long  instr, /*cycles, cache_misses, cache_refs, branch_instr,*/ fps;
	rread = read(fd_instr[workerid], &instr, sizeof(instr));
	assert(rread == sizeof(instr));
	/* read(fd_cycles[workerid], &cycles, sizeof(long long)); */
	/* read(fd_cache_misses[workerid], &cache_misses, sizeof(long long)); */
	/* read(fd_cache_refs[workerid], &cache_refs, sizeof(long long)); */
	/* read(fd_branch_instr[workerid], &branch_instr, sizeof(long long)); */
	rread = read(fd_fps[workerid], &fps, sizeof(long long));
	assert(rread == sizeof(long long));

	total_instr[sched_ctx] += instr;
	/* total_cycles[sched_ctx] += cycles; */
	/* total_cache_misses[sched_ctx] += cache_misses; */
	/* total_cache_refs[sched_ctx] += cache_refs; */
	/* total_branch_instr[sched_ctx] += branch_instr; */
	total_fps[sched_ctx] += fps;

        printf("Instrs %lf M instr of worker %lf M\n", (double)total_instr[sched_ctx]/1000000,
               (double)instr/1000000);
        printf("Fps %lf M curr fps %lf M \n",  (double)total_fps[sched_ctx]/1000000,
	       (double)fps/1000000);

	printf("Task Flops %lf k %s \n", task->flops/1000, (task->cl && task->cl->model) ? task->cl->model->symbol : "task null");
        printf("-------------------------------------------\n");

}

void print_results_for_ctx(unsigned sched_ctx, struct starpu_task *task)
{
        long long curr_total_instr = 0;
        /* long long curr_total_cycles = 0; */
        /* long long curr_total_cache_misses = 0; */
        /* long long curr_total_cache_refs = 0; */
        /* long long curr_total_branch_instr = 0; */
        long long curr_total_fps = 0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);

	struct starpu_sched_ctx_iterator it;

	int workerid;
	workers->init_iterator(workers, &it);
        while(workers->has_next(workers, &it))
	{
		ssize_t rread;

		workerid = workers->get_next(workers, &it);
                // Read event counter value
                struct read_format instr, /*cycles, cache_misses, cache_refs, branch_instr,*/ fps;
                rread = read(fd_instr[workerid], &instr, sizeof(struct read_format));
		assert(rread==sizeof(struct read_format));
		/* read(fd_cycles[workerid], &cycles, sizeof(long long));  */
                /* read(fd_cache_misses[workerid], &cache_misses, sizeof(long long)); */
		/* read(fd_cache_refs[workerid], &cache_refs, sizeof(long long));   */
		/* read(fd_branch_instr[workerid], &branch_instr, sizeof(long long));  */
		rread = read(fd_fps[workerid], &fps, sizeof(struct read_format));
		assert(rread == sizeof(struct read_format));

		curr_total_instr += (instr.time_enabled != 0 && instr.time_running !=0) ? instr.value * instr.time_enabled/instr.time_running : instr.value;
		printf("w%d instr time enabled %"PRIu64" time running %"PRIu64" \n", workerid, instr.time_enabled, instr.time_running);

		/* curr_total_cycles += cycles; */
		/* curr_total_cache_misses += cache_misses; */
		/* curr_total_cache_refs += cache_refs; */
		/* curr_total_branch_instr += branch_instr; */
		curr_total_fps += (fps.time_enabled != 0 && fps.time_running !=0) ? fps.value * fps.time_enabled/fps.time_running : fps.value;
		printf("w%d fps time enabled %lu time running %lu \n", workerid, fps.time_enabled, fps.time_running);
        }

        total_instr[sched_ctx] += curr_total_instr;
	/* total_cycles[sched_ctx] += curr_total_cycles; */
        /* total_cache_misses[sched_ctx] += curr_total_cache_misses; */
        /* total_cache_refs[sched_ctx] += curr_total_cache_refs; */
        /* total_branch_instr[sched_ctx] += curr_total_branch_instr; */
        total_fps[sched_ctx] += curr_total_fps;

        printf("%u: Instrs %lf k curr instr %lf k\n", sched_ctx, (double)total_instr[sched_ctx]/1000,
               (double)curr_total_instr/1000);
        printf("%u: Fps %lf k curr fps %lf k\n",  sched_ctx,
	       (double)total_fps[sched_ctx]/1000,
	       (double)curr_total_fps/1000);

	printf("%u: Task Flops %lf k %s \n", sched_ctx, task->flops/1000, (task->cl && task->cl->model) ? task->cl->model->symbol : "task null");
        printf("-------------------------------------------\n");
}

void config_event(struct perf_event_attr *event, unsigned with_time, uint64_t event_type, uint64_t config_type)
{
	memset(event, 0, sizeof(struct perf_event_attr));
	event->type = event_type;
	event->size = sizeof(struct perf_event_attr);
	event->config = config_type;
	event->disabled = 1;        // Event is initially disabled
	event->exclude_kernel = 1;  // excluding events that happen in the kernel space
	if(with_time)
	{
	     /* if the PMU is multiplexing several events we measure the time spent to actually measure this event (time_running)
		and compare it to the one expected is did, thus we compute the precision of the counter*/
		event->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
	}

}

void open_event(int *file_desc, struct perf_event_attr *event, int group_fd)
{
	*file_desc = perf_event_open(event, 0, -1, group_fd, 0);
	if (*file_desc == -1) {
		fprintf(stderr, "Error opening leader %llx\n", event->config);
		perror("perf_event_open");
		exit(0);
	}

}
void config_all_events_for_worker(int workerid)
{
	config_event(&pe_instr[workerid], 1, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
	/* config_event(&pe_cycles[workerid], 0, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES); */
	/* config_event(&pe_cache_misses[workerid], 0, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES); */
	/* config_event(&pe_cache_refs[workerid], 0, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES); */
	/* config_event(&pe_branch_instr[workerid], 0, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS); */
	config_event(&pe_fps[workerid], 1, PERF_TYPE_RAW, 0x1010);
}

void open_all_events_for_worker(int curr_workerid)
{
	open_event(&fd_instr[curr_workerid], &pe_instr[curr_workerid], -1);
	/* open_event(&fd_cycles[curr_workerid], &pe_cycles[curr_workerid], fd_instr[curr_workerid]); */
	/* open_event(&fd_cache_misses[curr_workerid], &pe_cache_misses[curr_workerid], fd_instr[curr_workerid]); */
	/* open_event(&fd_cache_refs[curr_workerid], &pe_cache_refs[curr_workerid], fd_instr[curr_workerid]); */
	/* open_event(&fd_branch_instr[curr_workerid], &pe_branch_instr[curr_workerid], fd_instr[curr_workerid]); */
	open_event(&fd_fps[curr_workerid], &pe_fps[curr_workerid], fd_instr[curr_workerid]);
}

void close_all_events_for_worker(int curr_workerid)
{
	close(fd_instr[curr_workerid]);
	/* close(fd_cycles[curr_workerid]); */
	/* close(fd_cache_misses[curr_workerid]); */
	/* close(fd_cache_refs[curr_workerid]); */
	/* close(fd_branch_instr[curr_workerid]); */
	close(fd_fps[curr_workerid]);
}

void start_monitoring_all_events_for_worker(int workerid)
{
	ioctl(fd_instr[workerid], PERF_EVENT_IOC_RESET, 0);
	ioctl(fd_instr[workerid], PERF_EVENT_IOC_ENABLE, 0);

	/* ioctl(fd_cycles[workerid], PERF_EVENT_IOC_RESET, 0); */
	/* ioctl(fd_cycles[workerid], PERF_EVENT_IOC_ENABLE, 0); */

	/* ioctl(fd_cache_misses[workerid], PERF_EVENT_IOC_RESET, 0); */
	/* ioctl(fd_cache_misses[workerid], PERF_EVENT_IOC_ENABLE, 0); */

	/* ioctl(fd_cache_refs[workerid], PERF_EVENT_IOC_RESET, 0); */
	/* ioctl(fd_cache_refs[workerid], PERF_EVENT_IOC_ENABLE, 0); */

	/* ioctl(fd_branch_instr[workerid], PERF_EVENT_IOC_RESET, 0); */
	/* ioctl(fd_branch_instr[workerid], PERF_EVENT_IOC_ENABLE, 0); */

	ioctl(fd_fps[workerid], PERF_EVENT_IOC_RESET, 0);
	ioctl(fd_fps[workerid], PERF_EVENT_IOC_ENABLE, 0);
}

void stop_monitoring_all_events_for_worker(int workerid)
{
	ioctl(fd_instr[workerid], PERF_EVENT_IOC_DISABLE, 0);
	/* ioctl(fd_cycles[workerid], PERF_EVENT_IOC_DISABLE, 0); */
	/* ioctl(fd_cache_misses[workerid], PERF_EVENT_IOC_DISABLE, 0); */
	/* ioctl(fd_cache_refs[workerid], PERF_EVENT_IOC_DISABLE, 0); */
	/* ioctl(fd_branch_instr[workerid], PERF_EVENT_IOC_DISABLE, 0); */
	ioctl(fd_fps[workerid], PERF_EVENT_IOC_DISABLE, 0);
}

void perf_count_handle_idle_end(unsigned sched_ctx, int worker)
{
	unsigned has_starpu_scheduler;
	unsigned has_awake_workers;
	has_starpu_scheduler = starpu_sched_ctx_has_starpu_scheduler(sched_ctx, &has_awake_workers);

	if(!has_starpu_scheduler && !has_awake_workers)
	{
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);

		struct starpu_sched_ctx_iterator it;

		int workerid;
		workers->init_iterator(workers, &it);
		while(workers->has_next(workers, &it))
		{
			workerid = workers->get_next(workers, &it);
			if(perf_event_opened[workerid])
				start_monitoring_all_events_for_worker(workerid);
		}
	}
	else
	{
		if(!perf_event_opened[worker])
		{
			config_all_events_for_worker(worker);
			open_all_events_for_worker(worker);
			perf_event_opened[worker] = 1;
		}
		start_monitoring_all_events_for_worker(worker);
	}
}

void perf_count_handle_poped_task(unsigned sched_ctx, int worker,
				  struct starpu_task *task,
				  __attribute__((unused))uint32_t footprint)
{
	unsigned has_starpu_scheduler;
	unsigned has_awake_workers;
	has_starpu_scheduler = starpu_sched_ctx_has_starpu_scheduler(sched_ctx, &has_awake_workers);

	if(!has_starpu_scheduler && !has_awake_workers)
	{
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);

		struct starpu_sched_ctx_iterator it;

		int workerid;
		workers->init_iterator(workers, &it);
		while(workers->has_next(workers, &it))
		{
			workerid = workers->get_next(workers, &it);
			if(perf_event_opened[workerid])
				stop_monitoring_all_events_for_worker(workerid);
		}
//		printf("worker requesting %d in ctx %d \n", starpu_worker_get_id(), sched_ctx);
		print_results_for_ctx(sched_ctx, task);
	}
	else
	{
		if(perf_event_opened[worker])
			stop_monitoring_all_events_for_worker(worker);
		print_results_for_worker(worker, sched_ctx, task);
	}
}

void perf_count_init_worker(int workerid, unsigned sched_ctx)
{
	if(!perf_event_opened[workerid])
	{
		open_all_events_for_worker(workerid);
		perf_event_opened[workerid] = 1;
	}
	else
	{
		close_all_events_for_worker(workerid);
		open_all_events_for_worker(workerid);
	}
}

void perf_count_start_ctx(unsigned sched_ctx)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);

	struct starpu_sched_ctx_iterator it;

	int workerid;
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		workerid = workers->get_next(workers, &it);
		config_all_events_for_worker(workerid);
	}
}

void perf_count_end_ctx(unsigned sched_ctx)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx);

	struct starpu_sched_ctx_iterator it;

	int workerid;
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		workerid = workers->get_next(workers, &it);
		close_all_events_for_worker(workerid);
	}
}

struct sc_hypervisor_policy perf_count_policy =
{
	.size_ctxs = NULL,
	.handle_poped_task = perf_count_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = perf_count_handle_idle_end,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = perf_count_end_ctx,
	.start_ctx = perf_count_start_ctx,
	.init_worker = perf_count_init_worker,
	.custom = 0,
	.name = "perf_count"
};
