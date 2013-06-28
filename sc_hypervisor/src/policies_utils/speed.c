/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  INRIA
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


double sc_hypervisor_get_ctx_velocity(struct sc_hypervisor_wrapper* sc_w)
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

	if(elapsed_flops >= redim_sample)
        {
                double curr_time = starpu_timing_now();
                double elapsed_time = (curr_time - sc_w->start_time) / 1000000.0; /* in seconds */
                return (elapsed_flops/1000000000.0)/elapsed_time;/* in Gflops/s */
        }
	return -1.0;
}

double sc_hypervisor_get_velocity_per_worker(struct sc_hypervisor_wrapper *sc_w, unsigned worker)
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

/* /\* 				double transfer_velocity = starpu_get_bandwidth_RAM_CUDA(worker); *\/ */
/* /\* 				elapsed_time +=  (elapsed_data_used / transfer_velocity) / 1000000 ; *\/ */
/* /\* 			} *\/ */
/* 			double latency = starpu_get_latency_RAM_CUDA(worker); */
/* //			printf("%d/%d: latency %lf elapsed_time before %lf ntasks %d\n", worker, sc_w->sched_ctx, latency, elapsed_time, elapsed_tasks); */
/* 			elapsed_time += (elapsed_tasks * latency)/1000000; */
/* //			printf("elapsed time after %lf \n", elapsed_time); */
/* 		} */
			
                double vel  = (elapsed_flops/elapsed_time);/* in Gflops/s */
		sc_w->ref_velocity[worker] = sc_w->ref_velocity[worker] > 1.0 ? (sc_w->ref_velocity[worker] + vel) / 2 : vel; 
                return vel;
        }

        return -1.0;


}


/* compute an average value of the cpu/cuda velocity */
double sc_hypervisor_get_velocity_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch)
{
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
        int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
                workers->init_iterator(workers, &it);

	double velocity = 0.0;
	unsigned nworkers = 0;
        while(workers->has_next(workers, &it))
	{
                worker = workers->get_next(workers, &it);
                enum starpu_worker_archtype req_arch = starpu_worker_get_type(worker);
                if(arch == req_arch)
                {
			double _vel = sc_hypervisor_get_velocity_per_worker(sc_w, worker);
			if(_vel == -1.0) return -1.0;
			velocity += _vel;
			nworkers++;
		}
	}
			

        return (nworkers != 0 ? velocity / nworkers : -1.0);
}

/* compute an average value of the cpu/cuda old velocity */
double sc_hypervisor_get_ref_velocity_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_worker_archtype arch)
{
	double ref_velocity = 0.0;
	unsigned nw = 0;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sc_w->sched_ctx);
	int worker;

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
                enum starpu_worker_archtype req_arch = starpu_worker_get_type(worker);
                if(arch == req_arch)
                {
			if(sc_w->ref_velocity[worker] < 1.0) return -1.0;
			ref_velocity += sc_w->ref_velocity[worker];
			nw++;
		}
	}
	
	return (nw != 0 ? ref_velocity / nw : -1.0);
}

double sc_hypervisor_get_velocity(struct sc_hypervisor_wrapper *sc_w, enum starpu_worker_archtype arch)
{

	double velocity = sc_hypervisor_get_velocity_per_worker_type(sc_w, arch);
	printf("arch %d vel %lf\n", arch, velocity);
	if(velocity == -1.0)
	{
		velocity = sc_hypervisor_get_ref_velocity_per_worker_type(sc_w, arch);
		printf("arch %d ref_vel %lf\n", arch, velocity);
	}
	if(velocity == -1.0)
	{
		velocity = arch == STARPU_CPU_WORKER ? 5.0 : 100.0;
		printf("arch %d default_vel %lf\n", arch, velocity);
	}
       
	return velocity;
}
