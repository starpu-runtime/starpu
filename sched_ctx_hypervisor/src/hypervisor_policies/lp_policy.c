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

#include "policy_utils.h"


/*                                                                                                                                                                                                                  
 * GNU Linear Programming Kit backend                                                                                                                                                                               
 */
#ifdef HAVE_GLPK_H
#include <glpk.h>
static void _glp_resolve(int ns, int nw, double v[ns][nw], double flops[ns])
{
	int s, w;
	glp_prob *lp;
	
	int ne =
		ns * (nw+1)     /* worker execution time */
		+ ns * nw
		+ 1; /* glp dumbness */
	int n = 1;
	int ia[ne], ja[ne];
	double ar[ne];

	lp = glp_create_prob();

	glp_set_prob_name(lp, "sample");
	glp_set_obj_dir(lp, GLP_MAX);
        glp_set_obj_name(lp, "max speed");


#define colnum(s, w) ((w)*ns+(s)+1)
	/* we add nw*ns columns one for each type of worker in each context 
	   and another column corresponding to the 1/tmax bound (bc 1/tmax is a variable too)*/
	glp_add_cols(lp, nw*ns+1);
	/* Z = 1/tmax -> 1/vmax structural variable, nCPUs & nGPUs in ctx are auxiliar variables */
	glp_set_obj_coef(lp, nw*ns+1, 1.);

	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			char name[32];
			snprintf(name, sizeof(name), "worker%dctx%d", w, s);
			glp_set_col_name(lp, colnum(s,w), name);
			glp_set_col_bnds(lp, colnum(s,w), GLP_LO, 0.0, 0.0);
		}
	}
	glp_set_col_bnds(lp, nw*ns+1, GLP_DB, 0.0, 1.0);


	/* one row corresponds to one ctx*/
	glp_add_rows(lp, ns);

	for(s = 0; s < ns; s++)
	{
		char name[32];
		snprintf(name, sizeof(name), "ctx%d", s);
		glp_set_row_name(lp, s+1, name);
		glp_set_row_bnds(lp, s+1, GLP_LO, 0., 0.);
		for(w = 0; w < nw; w++)
		{
			ia[n] = s+1;
			ja[n] = colnum(s, w);
			ar[n] = v[s][w];
			printf("v[%d][%d] = %lf\n", s, w, v[s][w]);
			n++;
		}
		/* 1/tmax */
		ia[n] = s+1;
		ja[n] = ns*nw+1;
		ar[n] = (-1) * flops[s];
		printf("%d: flops %lf\n", s, flops[s]);
		n++;
	}
	
	/*we add another linear constraint : sum(all cpus) = 3 and sum(all gpus) = 9 */
	glp_add_rows(lp, nw);

	for(w = 0; w < nw; w++)
	{
		char name[32];
		snprintf(name, sizeof(name), "w%d", w);
		glp_set_row_name(lp, ns+w+1, name);
		for(s = 0; s < ns; s++)
		{
			ia[n] = ns+w+1;
			ja[n] = colnum(s,w);
			ar[n] = 1;
			n++;
		}
		if(w == 0) glp_set_row_bnds(lp, ns+w+1, GLP_FX, 3., 3.);
		if(w == 1) glp_set_row_bnds(lp, ns+w+1, GLP_FX, 9., 9.);
	}


	STARPU_ASSERT(n == ne);

	glp_load_matrix(lp, ne-1, ia, ja, ar);

	glp_simplex(lp, NULL);
	double vmax1 = glp_get_obj_val(lp);
	printf("vmax1 = %lf \n", vmax1);
	double res[ne];
	n = 1;
	for(w = 0; w < nw; w++)
	{
		for(s = 0; s < ns; s++)
		{
			res[n] = glp_get_col_prim(lp, colnum(s,w));
			printf("ctx %d/worker type %d: n = %lf \n", s, w, res[n]);
			n++;
		}
	}
//	res[n] = glp_get_col_prim(lp, ns*nw+1);
//	printf("vmax = %lf \n", res[n]);

	glp_delete_prob(lp);
	return;
}

/* check if there is a big velocity gap between the contexts */
int _velocity_gap_btw_ctxs()
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();
	int i = 0, j = 0;
	struct sched_ctx_wrapper* sc_w;
	struct sched_ctx_wrapper* other_sc_w;
	
	for(i = 0; i < nsched_ctxs; i++)
	{
		sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctxs[i]);
		double ctx_v = _get_ctx_velocity(sc_w);
		for(j = 0; j < nsched_ctxs; j++)
		{
			if(sched_ctxs[i] != sched_ctxs[j])
			{
				other_sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctxs[j]);
				double other_ctx_v = _get_ctx_velocity(other_sc_w);
				double gap = ctx_v < other_ctx_v ? ctx_v / other_ctx_v : other_ctx_v / ctx_v;
				if(gap > 0.5)
					return 1;
			}
		}

	}
	return 0;
}

void lp_handle_poped_task(unsigned sched_ctx, int worker)
{
	if(_velocity_gap_btw_ctxs())
	{
		int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
		int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();
		
		double v[nsched_ctxs][2];
		double flops[nsched_ctxs];
		int i = 0;
		struct sched_ctx_wrapper* sc_w;
		for(i = 0; i < nsched_ctxs; i++)
		{
			sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctxs[i]);
			v[i][0] = 200.0;//_get_velocity_per_worker_type(sc_w, STARPU_CUDA_WORKER);
			v[i][1] = 20.0;//_get_velocity_per_worker_type(sc_w, STARPU_CPU_WORKER);
			flops[i] = sc_w->total_flops;
		}
                
		int ret = pthread_mutex_trylock(&act_hypervisor_mutex);
		if(ret != EBUSY)
		{
			_glp_resolve(nsched_ctxs, 2, v, flops);
			pthread_mutex_unlock(&act_hypervisor_mutex);
		}
	}		
}

struct hypervisor_policy lp_policy = {
	.handle_poped_task = lp_handle_poped_task,
	.handle_pushed_task = NULL,
	.handle_idle_cycle = NULL,
	.handle_idle_end = NULL,
	.handle_post_exec_hook = NULL,
	.custom = 0,
	.name = "lp"
};
	
#endif /* HAVE_GLPK_H */

