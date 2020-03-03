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

#ifndef SC_HYPERVISOR_LP_H
#define SC_HYPERVISOR_LP_H

#include <sc_hypervisor.h>
#include <starpu_config.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_SC_Hypervisor_LP Scheduling Context Hypervisor - Linear Programming
   @{
*/

#ifdef STARPU_HAVE_GLPK_H
#include <glpk.h>
#endif //STARPU_HAVE_GLPK_H

struct sc_hypervisor_policy_task_pool;
struct types_of_workers;
/**
   return tmax, and compute in table res the nr of workers needed by each context st the system ends up in the smallest tma
*/
double sc_hypervisor_lp_get_nworkers_per_ctx(int nsched_ctxs, int ntypes_of_workers, double res[nsched_ctxs][ntypes_of_workers],
					     int total_nw[ntypes_of_workers], struct types_of_workers *tw, unsigned *in_sched_ctxs);

/**
   return tmax of the system
*/
double sc_hypervisor_lp_get_tmax(int nw, int *workers);

/**
   the linear programme determins a rational number of ressources for each ctx, we round them depending on the type of ressource
*/
void sc_hypervisor_lp_round_double_to_int(int ns, int nw, double res[ns][nw], int res_rounded[ns][nw]);

/**
   redistribute the ressource in contexts by assigning the first x available ressources to each one
*/
void sc_hypervisor_lp_redistribute_resources_in_ctxs(int ns, int nw, int res_rounded[ns][nw], double res[ns][nw], unsigned *sched_ctxs, struct types_of_workers *tw);

/**
   make the first distribution of ressource in contexts by assigning the first x available ressources to each one
*/
void sc_hypervisor_lp_distribute_resources_in_ctxs(unsigned* sched_ctxs, int ns, int nw, int res_rounded[ns][nw], double res[ns][nw], int *workers, int nworkers, struct types_of_workers *tw);

/**
   make the first distribution of ressource in contexts by assigning the first x available ressources to each one, share not integer no of workers
*/
void sc_hypervisor_lp_distribute_floating_no_resources_in_ctxs(unsigned* sched_ctxs, int ns, int nw, double res[ns][nw], int *workers, int nworkers, struct types_of_workers *tw);

/**
   place resources in contexts dependig on whether they already have workers or not
*/
void sc_hypervisor_lp_place_resources_in_ctx(int ns, int nw, double w_in_s[ns][nw], unsigned *sched_ctxs, int *workers, unsigned do_size, struct types_of_workers *tw);

/**
   not used resources are shared between all contexts
*/
void sc_hypervisor_lp_share_remaining_resources(int ns, unsigned *sched_ctxs,  int nworkers, int *workers);

/**
   dichotomy btw t1 & t2
*/
double sc_hypervisor_lp_find_tmax(double t1, double t2);

/**
   execute the lp trough dichotomy
*/
unsigned sc_hypervisor_lp_execute_dichotomy(int ns, int nw, double w_in_s[ns][nw], unsigned solve_lp_integer, void *specific_data,
					    double tmin, double tmax, double smallest_tmax,
					    double (*lp_estimated_distrib_func)(int ns, int nw, double draft_w_in_s[ns][nw],
									     unsigned is_integer, double tmax, void *specifc_data));

#ifdef STARPU_HAVE_GLPK_H
/**
   linear program that returns 1/tmax, and computes in table res the
   nr of workers needed by each context st the system ends up in the
   smallest tmax
*/
double sc_hypervisor_lp_simulate_distrib_flops(int nsched_ctxs, int ntypes_of_workers, double speed[nsched_ctxs][ntypes_of_workers],
					       double flops[nsched_ctxs], double res[nsched_ctxs][ntypes_of_workers], int total_nw[ntypes_of_workers],
					       unsigned sched_ctxs[nsched_ctxs], double vmax);

/**
   linear program that simulates a distribution of tasks that
   minimises the execution time of the tasks in the pool
*/
double sc_hypervisor_lp_simulate_distrib_tasks(int ns, int nw, int nt, double w_in_s[ns][nw], double tasks[nw][nt],
					       double times[nw][nt], unsigned is_integer, double tmax, unsigned *in_sched_ctxs,
					       struct sc_hypervisor_policy_task_pool *tmp_task_pools);

/**
   linear program that simulates a distribution of flops over the
   workers on particular sample of the execution of the application
   such that the entire sample would finish in a minimum amount of
   time
*/
double sc_hypervisor_lp_simulate_distrib_flops_on_sample(int ns, int nw, double final_w_in_s[ns][nw], unsigned is_integer, double tmax,
							 double **speed, double flops[ns], double **final_flops_on_w);
#endif // STARPU_HAVE_GLPK_H


/** @} */

#ifdef __cplusplus
}
#endif

#endif
