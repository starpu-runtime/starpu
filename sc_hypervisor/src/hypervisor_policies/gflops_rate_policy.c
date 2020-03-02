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

static double _get_total_elapsed_flops_per_sched_ctx(unsigned sched_ctx)
{
	struct sc_hypervisor_wrapper* sc_w = sc_hypervisor_get_wrapper(sched_ctx);
	double ret_val = 0.0;
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		ret_val += sc_w->total_elapsed_flops[i];
	return ret_val;
}

double _get_exp_end(unsigned sched_ctx)
{
	struct sc_hypervisor_wrapper *sc_w = sc_hypervisor_get_wrapper(sched_ctx);
	double elapsed_flops = sc_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);

	if( elapsed_flops >= 1.0)
	{
		double curr_time = starpu_timing_now();
		double elapsed_time = curr_time - sc_w->start_time;
		double exp_end = (elapsed_time * sc_w->remaining_flops /  elapsed_flops) + curr_time;
		return exp_end;
	}
	return -1.0;
}

/* computes the instructions left to be executed out of the total instructions to execute */
double _get_flops_left_pct(unsigned sched_ctx)
{
	struct sc_hypervisor_wrapper *wrapper = sc_hypervisor_get_wrapper(sched_ctx);
	double total_elapsed_flops = _get_total_elapsed_flops_per_sched_ctx(sched_ctx);
	if(wrapper->total_flops == total_elapsed_flops || total_elapsed_flops > wrapper->total_flops)
		return 0.0;

	return (wrapper->total_flops - total_elapsed_flops)/wrapper->total_flops;
}

/* select the workers needed to be moved in order to force the sender and the receiver context to finish simultaneously */
static int* _get_workers_to_move(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int *nworkers)
{
	struct sc_hypervisor_wrapper* sender_sc_w = sc_hypervisor_get_wrapper(sender_sched_ctx);
	struct sc_hypervisor_wrapper* receiver_sc_w = sc_hypervisor_get_wrapper(receiver_sched_ctx);
        int *workers = NULL;
        double v_receiver = sc_hypervisor_get_ctx_speed(receiver_sc_w);
        double receiver_remainig_flops = receiver_sc_w->remaining_flops;
        double sender_exp_end = _get_exp_end(sender_sched_ctx);
        double sender_v_cpu = sc_hypervisor_get_speed_per_worker_type(sender_sc_w, STARPU_CPU_WORKER);
        double v_for_rctx = (receiver_remainig_flops/(sender_exp_end - starpu_timing_now())) - v_receiver;

        int nworkers_needed = v_for_rctx/sender_v_cpu;
/*      printf("%d->%d: v_rec %lf v %lf v_cpu %lf w_needed %d \n", sender_sched_ctx, receiver_sched_ctx, */
/*             v_receiver, v_for_rctx, sender_v_cpu, nworkers_needed); */
        if(nworkers_needed > 0)
        {
                struct sc_hypervisor_policy_config *sender_config = sc_hypervisor_get_config(sender_sched_ctx);
                int potential_moving_cpus = sc_hypervisor_get_movable_nworkers(sender_config, sender_sched_ctx, STARPU_CPU_WORKER);
                int potential_moving_gpus = sc_hypervisor_get_movable_nworkers(sender_config, sender_sched_ctx, STARPU_CUDA_WORKER);
                int sender_nworkers = (int)starpu_sched_ctx_get_nworkers(sender_sched_ctx);
                struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(receiver_sched_ctx);
                int nworkers_ctx = (int)starpu_sched_ctx_get_nworkers(receiver_sched_ctx);

                if(nworkers_needed < (potential_moving_cpus + 5 * potential_moving_gpus))
                {
                        if((sender_nworkers - nworkers_needed) >= sender_config->min_nworkers)
                        {
                                if((nworkers_ctx + nworkers_needed) > config->max_nworkers)
                                        nworkers_needed = nworkers_ctx > config->max_nworkers ? 0 : (config->max_nworkers - nworkers_ctx);

                                if(nworkers_needed > 0)
                                {
                                        int ngpus = nworkers_needed / 5;
                                        int *gpus;
                                        gpus = sc_hypervisor_get_idlest_workers(sender_sched_ctx, &ngpus, STARPU_CUDA_WORKER);
                                        int ncpus = nworkers_needed - ngpus;
                                        int *cpus;
                                        cpus = sc_hypervisor_get_idlest_workers(sender_sched_ctx, &ncpus, STARPU_CPU_WORKER);
                                        workers = (int*)malloc(nworkers_needed*sizeof(int));
                                        int i;
					printf("%d: gpus: ", nworkers_needed);
                                        for(i = 0; i < ngpus; i++)
					{
                                                workers[(*nworkers)++] = gpus[i];
						printf("%d ", gpus[i]);
					}
					printf(" cpus:");
                                        for(i = 0; i < ncpus; i++)
					{
                                                workers[(*nworkers)++] = cpus[i];
						printf("%d ", cpus[i]);
					}
					printf("\n");
                                        free(gpus);
                                        free(cpus);
                                }
                        }
                }
		else
                {
			/*if the needed number of workers is to big we only move the number of workers
			  corresponding to the granularity set by the user */
                        int nworkers_to_move = sc_hypervisor_compute_nworkers_to_move(sender_sched_ctx);

                        if(sender_nworkers - nworkers_to_move >= sender_config->min_nworkers)
                        {
                                int nshared_workers = (int)starpu_sched_ctx_get_nshared_workers(sender_sched_ctx, receiver_sched_ctx);
                                if((nworkers_ctx + nworkers_to_move - nshared_workers) > config->max_nworkers)
                                        nworkers_to_move = nworkers_ctx > config->max_nworkers ? 0 : (config->max_nworkers - nworkers_ctx + nshared_workers);

                                if(nworkers_to_move > 0)
                                {
                                        workers = sc_hypervisor_get_idlest_workers(sender_sched_ctx, &nworkers_to_move, STARPU_ANY_WORKER);
                                        *nworkers = nworkers_to_move;
                                }
                        }
                }
        }
        return workers;
}

static unsigned _gflops_rate_resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize)
{
        int ret = 1;
        if(force_resize)
                STARPU_PTHREAD_MUTEX_LOCK(&act_hypervisor_mutex);
        else
                ret = starpu_pthread_mutex_trylock(&act_hypervisor_mutex);
        if(ret != EBUSY)
        {
                int nworkers_to_move = 0;
                int *workers_to_move =  _get_workers_to_move(sender_sched_ctx, receiver_sched_ctx, &nworkers_to_move);
		if(nworkers_to_move > 0)
                {
                        sc_hypervisor_move_workers(sender_sched_ctx, receiver_sched_ctx, workers_to_move, nworkers_to_move, 0);

                        struct sc_hypervisor_policy_config *new_config = sc_hypervisor_get_config(receiver_sched_ctx);
                        int i;
                        for(i = 0; i < nworkers_to_move; i++)
                                new_config->max_idle[workers_to_move[i]] = new_config->max_idle[workers_to_move[i]] !=MAX_IDLE_TIME ? new_config->max_idle[workers_to_move[i]] :  new_config->new_workers_max_idle;

                        free(workers_to_move);
                }
                STARPU_PTHREAD_MUTEX_UNLOCK(&act_hypervisor_mutex);
                return 1;
        }
        return 0;

}

static int _find_fastest_sched_ctx()
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	double first_exp_end = _get_exp_end(sched_ctxs[0]);
	int fastest_sched_ctx = first_exp_end == -1.0  ? -1 : (int)sched_ctxs[0];
	double curr_exp_end = 0.0;
	int i;
	for(i = 1; i < nsched_ctxs; i++)
	{
		curr_exp_end = _get_exp_end(sched_ctxs[i]);
		if((curr_exp_end < first_exp_end || first_exp_end == -1.0) && curr_exp_end != -1.0)
		{
			first_exp_end = curr_exp_end;
			fastest_sched_ctx = sched_ctxs[i];
		}
	}

	return fastest_sched_ctx;

}

static int _find_slowest_sched_ctx()
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	int slowest_sched_ctx = -1;
	double curr_exp_end = 0.0;
	double last_exp_end = -1.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		curr_exp_end = _get_exp_end(sched_ctxs[i]);
		/*if it hasn't started bc of no ressources give it priority */
		if(curr_exp_end == -1.0)
			return sched_ctxs[i];
		if( curr_exp_end > last_exp_end)
		{
			slowest_sched_ctx = sched_ctxs[i];
			last_exp_end = curr_exp_end;
		}
	}

	return slowest_sched_ctx;

}

static int _find_slowest_available_sched_ctx(unsigned sched_ctx)
{
	unsigned *sched_ctxs = sc_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sc_hypervisor_get_nsched_ctxs();

	int slowest_sched_ctx = -1;
	double curr_exp_end = 0.0;
	double last_exp_end = -1.0;
	int i;
	for(i = 0; i < nsched_ctxs; i++)
	{
		if(sched_ctxs[i] != sched_ctx)
		{
			curr_exp_end = _get_exp_end(sched_ctxs[i]);
			/*if it hasn't started bc of no ressources give it priority */
			if(curr_exp_end == -1.0)
				return sched_ctxs[i];
			if(last_exp_end < curr_exp_end)
			{
				slowest_sched_ctx = sched_ctxs[i];
				last_exp_end = curr_exp_end;
			}
		}
	}

	return slowest_sched_ctx;

}

static void gflops_rate_resize(unsigned sched_ctx)
{
	_get_exp_end(sched_ctx);
	double flops_left_pct = _get_flops_left_pct(sched_ctx);

	/* if the context finished all the instructions it had to execute
	 we move all the resources to the slowest context */
	if(flops_left_pct == 0.0f)
	{
		int slowest_sched_ctx = _find_slowest_available_sched_ctx(sched_ctx);
		if(slowest_sched_ctx != -1)
		{
			double slowest_flops_left_pct = _get_flops_left_pct(slowest_sched_ctx);
			if(slowest_flops_left_pct != 0.0f)
			{
				struct sc_hypervisor_policy_config* config = sc_hypervisor_get_config(sched_ctx);
				config->min_nworkers = 0;
				config->max_nworkers = 0;
				printf("ctx %u finished & gives away the res to %d; slow_left %lf\n", sched_ctx, slowest_sched_ctx, slowest_flops_left_pct);
				sc_hypervisor_policy_resize(sched_ctx, slowest_sched_ctx, 1, 1);
				sc_hypervisor_stop_resize(slowest_sched_ctx);
			}
		}
	}

	int fastest_sched_ctx = _find_fastest_sched_ctx();
	int slowest_sched_ctx = _find_slowest_sched_ctx();

	if(fastest_sched_ctx != -1 && slowest_sched_ctx != -1 && fastest_sched_ctx != slowest_sched_ctx)
	{
		double fastest_exp_end = _get_exp_end(fastest_sched_ctx);
		double slowest_exp_end = _get_exp_end(slowest_sched_ctx);

		if((slowest_exp_end == -1.0 && fastest_exp_end != -1.0) || ((fastest_exp_end + (fastest_exp_end*0.5)) < slowest_exp_end ))
		{
			double fast_flops_left_pct = _get_flops_left_pct(fastest_sched_ctx);
			if(fast_flops_left_pct < 0.8)
			{

				struct sc_hypervisor_wrapper *sc_w = sc_hypervisor_get_wrapper(slowest_sched_ctx);
				double elapsed_flops = sc_hypervisor_get_elapsed_flops_per_sched_ctx(sc_w);
				if((elapsed_flops/sc_w->total_flops) > 0.1)
					_gflops_rate_resize(fastest_sched_ctx, slowest_sched_ctx, 0);
			}
		}
	}
}

static void gflops_rate_handle_poped_task(unsigned sched_ctx, __attribute__((unused)) int worker, 
					  __attribute__((unused))struct starpu_task *task, __attribute__((unused))uint32_t footprint)
{
	gflops_rate_resize(sched_ctx);
}

struct sc_hypervisor_policy gflops_rate_policy = {
	.size_ctxs = NULL,
	.resize_ctxs = NULL,
	.handle_poped_task = gflops_rate_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.handle_submitted_job = NULL,
	.end_ctx = NULL,
	.init_worker = NULL,
	.custom = 0,
	.name = "gflops_rate"
};
