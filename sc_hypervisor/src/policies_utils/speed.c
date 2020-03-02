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

#include "sc_hypervisor_policy.h"
#include "sc_hypervisor_intern.h"
#include <math.h>


double sc_hypervisor_get_ctx_speed(struct sc_hypervisor_wrapper* sc_w)
{
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
        double elapsed_flops = sc_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);
	double sample = config->ispeed_ctx_sample;


	double total_elapsed_flops = sc_hypervisor_get_total_elapsed_flops_per_sched_ctx(sc_w);
	double total_flops = sc_w->total_flops;

	char *start_sample_prc_char = getenv("SC_HYPERVISOR_START_RESIZE");
	double start_sample_prc = start_sample_prc_char ? atof(start_sample_prc_char) : 0.0;
	double start_sample = start_sample_prc > 0.0 ? (start_sample_prc / 100) * total_flops : sample;
	double redim_sample = elapsed_flops == total_elapsed_flops ? (start_sample > 0.0 ? start_sample : sample) : sample;

	double curr_time = starpu_timing_now();
	double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */

	unsigned can_compute_speed = 0;
	char *speed_sample_criteria = getenv("SC_HYPERVISOR_SAMPLE_CRITERIA");
	if(speed_sample_criteria && (strcmp(speed_sample_criteria, "time") == 0))
		can_compute_speed = elapsed_time > config->time_sample;
	else
		can_compute_speed = elapsed_flops >= redim_sample;

	if(can_compute_speed)
        {
                return (elapsed_flops/1000000000.0)/elapsed_time;/* in Gflops/s */
        }
	return -1.0;
}

double sc_hypervisor_get_speed_per_worker(struct sc_hypervisor_wrapper *sc_w, unsigned worker)
{
	if(!starpu_sched_ctx_contains_worker(worker, sc_w->sched_ctx))
		return -1.0;

        double elapsed_flops = sc_w->elapsed_flops[worker] / 1000000000.0; /*in gflops */

	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);
	double sample = config->ispeed_w_sample[worker] / 1000000000.0; /*in gflops */

	double ctx_elapsed_flops = sc_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);
	double ctx_sample = config->ispeed_ctx_sample;
	if(ctx_elapsed_flops > ctx_sample && elapsed_flops == 0.0)
		return 0.00000000000001;


        if( elapsed_flops > sample)
        {
                double curr_time = starpu_timing_now();
                double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */
		elapsed_time -= sc_w->idle_time[worker];


/* 		size_t elapsed_data_used = sc_w->elapsed_data[worker]; */
/*  		enum starpu_worker_archtype arch = starpu_worker_get_type(worker); */
/* 		if(arch == STARPU_CUDA_WORKER) */
/* 		{ */
/* /\* 			unsigned worker_in_ctx = starpu_sched_ctx_contains_worker(worker, sc_w->sched_ctx); *\/ */
/* /\* 			if(!worker_in_ctx) *\/ */
/* /\* 			{ *\/ */

/* /\* 				double transfer_speed = starpu_transfer_bandwidth(STARPU_MAIN_RAM, starpu_worker_get_memory_node(worker)); *\/ */
/* /\* 				elapsed_time +=  (elapsed_data_used / transfer_speed) / 1000000 ; *\/ */
/* /\* 			} *\/ */
/* 			double latency = starpu_transfer_latency(STARPU_MAIN_RAM, starpu_worker_get_memory_node(worker)); */
/* //			printf("%d/%d: latency %lf elapsed_time before %lf ntasks %d\n", worker, sc_w->sched_ctx, latency, elapsed_time, elapsed_tasks); */
/* 			elapsed_time += (elapsed_tasks * latency)/1000000; */
/* //			printf("elapsed time after %lf \n", elapsed_time); */
/* 		} */

                double vel  = (elapsed_flops/elapsed_time);/* in Gflops/s */
                return vel;
        }

        return -1.0;


}


/* compute an average value of the cpu/cuda speed */
double sc_hypervisor_get_speed_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch)
{
	struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sc_w->sched_ctx);

	double ctx_elapsed_flops = sc_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);
	double ctx_sample = config->ispeed_ctx_sample;

	double curr_time = starpu_timing_now();
	double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */

	unsigned can_compute_speed = 0;
	char *speed_sample_criteria = getenv("SC_HYPERVISOR_SAMPLE_CRITERIA");
	if(speed_sample_criteria && (strcmp(speed_sample_criteria, "time") == 0))
		can_compute_speed = elapsed_time > config->time_sample;
	else
		can_compute_speed = ctx_elapsed_flops > ctx_sample;

	if(can_compute_speed)
        {
		if(ctx_elapsed_flops == 0.0) return -1.0;

		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
		int worker;

		struct starpu_sched_ctx_iterator it;
		workers->init_iterator(workers, &it);

		double speed = 0.0;
		unsigned nworkers = 0;
		double all_workers_flops = 0.0;
		double max_workers_idle_time = 0.0;
		while(workers->has_next(workers, &it))
		{
			worker = workers->get_next(workers, &it);
			enum starpu_worker_archtype req_arch = starpu_worker_get_type(worker);
			if(arch == req_arch && sc_w->compute_idle[worker])
			{
				if(sc_w->exec_start_time[worker] != 0.0)
				{
					double current_exec_time = 0.0;
					if(sc_w->exec_start_time[worker] < sc_w->start_time)
						current_exec_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */
					else
						current_exec_time = (curr_time - sc_w->exec_start_time[worker]) / 1000000.0; /* in seconds */
					double suppl_flops = current_exec_time * sc_hypervisor_get_ref_speed_per_worker_type(sc_w, req_arch);
					all_workers_flops += suppl_flops;
				}

				all_workers_flops += sc_w->elapsed_flops[worker] / 1000000000.0; /*in gflops */
				if(max_workers_idle_time < sc_w->idle_time[worker])
					max_workers_idle_time = sc_w->idle_time[worker]; /* in seconds */
				nworkers++;
			}
		}

		if(nworkers != 0 && all_workers_flops > 0.0001)
		{
//			elapsed_time -= max_workers_idle_time;
			speed = (all_workers_flops / elapsed_time) / nworkers;
		}
		else
			speed = -1.0;

		if(speed != -1.0)
		{

			/* if ref_speed started being corrupted bc of the old bad distribution
			   register only the last frame otherwise make the average with the speed
			   behavior of the application until now */
			if(arch == STARPU_CUDA_WORKER)
				sc_w->ref_speed[0] = (sc_w->ref_speed[0] > 0.1) ? ((sc_w->ref_speed[0] + speed ) / 2.0) : speed;
			else
				sc_w->ref_speed[1] = (sc_w->ref_speed[1] > 0.1) ? ((sc_w->ref_speed[1] + speed ) / 2.0) : speed;
		}
		return speed;
	}

	return -1.0;
}

/* compute an average value of the cpu/cuda old speed */
double sc_hypervisor_get_ref_speed_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch)
{
	if(arch == STARPU_CUDA_WORKER && sc_w->ref_speed[0] > 0.0)
		return sc_w->ref_speed[0];
	else
		if(arch == STARPU_CPU_WORKER && sc_w->ref_speed[1] > 0.0)
			return sc_w->ref_speed[1];

	return -1.0;
}

/* returns the speed necessary for the linear programs (either the monitored one either a default value) */
double sc_hypervisor_get_speed(struct sc_hypervisor_wrapper *sc_w, enum starpu_worker_archtype arch)
{
	/* monitored speed in the last frame */
	double speed = sc_hypervisor_get_speed_per_worker_type(sc_w, arch);
	if(speed == -1.0)
	{
		/* avg value of the monitored speed over the entier current execution */
		speed = sc_hypervisor_get_ref_speed_per_worker_type(sc_w, arch);
	}
	if(speed == -1.0)
	{
		/* a default value */
		speed = arch == STARPU_CPU_WORKER ? SC_HYPERVISOR_DEFAULT_CPU_SPEED : SC_HYPERVISOR_DEFAULT_CUDA_SPEED;
	}

	return speed;
}

double sc_hypervisor_get_avg_speed(enum starpu_worker_archtype arch)
{
	double total_executed_flops = 0.0;
	double total_estimated_flops = 0.0;
	struct sc_hypervisor_wrapper *sc_w;
	double max_real_start_time = 0.0;
	int s;
	unsigned nworkers =  starpu_worker_get_count_by_type(arch);

	unsigned *sched_ctxs;
	int nsched_ctxs;
	sc_hypervisor_get_ctxs_on_level(&sched_ctxs, &nsched_ctxs, 0, STARPU_NMAX_SCHED_CTXS);

	for(s = 0; s < nsched_ctxs; s++)
	{
		sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);
		struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctxs[s]);
		int worker;

		struct starpu_sched_ctx_iterator it;
		workers->init_iterator(workers, &it);

		while(workers->has_next(workers, &it))
		{
			worker = workers->get_next(workers, &it);
			enum starpu_worker_archtype req_arch = starpu_worker_get_type(worker);
			if(arch == req_arch)
			{
				total_executed_flops += sc_w->total_elapsed_flops[worker] / 1000000000.0; /*in gflops */;
			}
		}

		total_estimated_flops += sc_w->total_flops / 1000000000.0; /*in gflops */

		if(max_real_start_time < sc_w->real_start_time)
			max_real_start_time = sc_w->real_start_time;
	}
	free(sched_ctxs);

	double speed = -1.0;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
	printf("total_exec_flops %lf total_estimated_flops %lf max_real_start_time %lf nworkers %u \n", total_executed_flops, total_estimated_flops, max_real_start_time, nworkers);
#endif
	if(total_executed_flops > 0.5*total_estimated_flops)
	{
		double curr_time = starpu_timing_now();
		double time = (curr_time - max_real_start_time) / 1000000.0; /* in seconds */
#ifdef STARPU_SC_HYPERVISOR_DEBUG
		printf("time = %lf\n", time);
#endif
		speed = (total_executed_flops / time) / nworkers;
	}

	return speed;
}

void _consider_max_for_children(unsigned sched_ctx, unsigned consider_max)
{
	struct sc_hypervisor_wrapper *sc_w = sc_hypervisor_get_wrapper(sched_ctx);
	sc_w->consider_max = consider_max;
#ifdef STARPU_SC_HYPERVISOR_DEBUG
	printf("ctx %u consider max %d \n", sched_ctx, sc_w->consider_max);
#endif

	int level = starpu_sched_ctx_get_hierarchy_level(sched_ctx);
	unsigned *sched_ctxs_child;
	int nsched_ctxs_child = 0;
	sc_hypervisor_get_ctxs_on_level(&sched_ctxs_child, &nsched_ctxs_child, level+1, sched_ctx);
	int s;
	for(s = 0; s < nsched_ctxs_child; s++)
		_consider_max_for_children(sched_ctxs_child[s], consider_max);
	free(sched_ctxs_child);
}

void sc_hypervisor_check_if_consider_max(struct types_of_workers *tw)
{
	unsigned *sched_ctxs;
	int nsched_ctxs;
	sc_hypervisor_get_ctxs_on_level(&sched_ctxs, &nsched_ctxs, 0, STARPU_NMAX_SCHED_CTXS);

	int nw = tw->nw;
	double avg_speed_per_tw[nw];
	int w;
	for(w = 0; w < nw; w++)
	{
		avg_speed_per_tw[w] = sc_hypervisor_get_avg_speed(sc_hypervisor_get_arch_for_index(w, tw));
		if(avg_speed_per_tw[w] == -1.0)
		{
			free(sched_ctxs);
			return;
		}
	}

	int s;
	for(s = 0; s < nsched_ctxs; s++)
	{
		for(w = 0; w < nw; w++)
		{
			struct sc_hypervisor_wrapper *sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]);
			double speed = sc_hypervisor_get_speed(sc_w, sc_hypervisor_get_arch_for_index(w, tw));
#ifdef STARPU_SC_HYPERVISOR_DEBUG
			printf("%u: speed %lf avg_speed %lf min %lf max %lf\n", sched_ctxs[s], speed, avg_speed_per_tw[w], (avg_speed_per_tw[w]*0.5), (avg_speed_per_tw[w]*1.5));
#endif
			if(speed < avg_speed_per_tw[w]*0.5 || speed > avg_speed_per_tw[w]*1.5)
				_consider_max_for_children(sched_ctxs[s], 1);
			else
				_consider_max_for_children(sched_ctxs[s], 0);
		}
	}
	free(sched_ctxs);
}
