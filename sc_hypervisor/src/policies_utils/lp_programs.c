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
/*
 * GNU Linear Programming Kit backend
 */

#include "sc_hypervisor_policy.h"
#include "sc_hypervisor_lp.h"

#ifdef STARPU_HAVE_GLPK_H

double sc_hypervisor_lp_simulate_distrib_tasks(int ns, int nw, int nt, double w_in_s[ns][nw], double tasks[nw][nt],
					       double times[nw][nt], unsigned is_integer, double tmax, unsigned *in_sched_ctxs,
					       struct sc_hypervisor_policy_task_pool *tmp_task_pools)
{
	struct sc_hypervisor_policy_task_pool * tp;
	int t, w, s;
	glp_prob *lp;


	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MAX);
	glp_set_obj_name(lp, "total execution time");

	{
		int ne = nt * nw /* worker execution time */
			+ nw * ns
			+ nw * (nt + ns)
			+ 1; /* glp dumbness */
		int n = 1;
		int ia[ne], ja[ne];
		double ar[ne];

		/* Variables: number of tasks i assigned to worker j, and tmax */
		glp_add_cols(lp, nw*nt+ns*nw);
#define colnum(w, t) ((t)*nw+(w)+1)
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
				glp_set_obj_coef(lp, nw*nt+s*nw+w+1, 1.);

		for (w = 0; w < nw; w++)
			for (t = 0; t < nt; t++)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%dt%dn", w, t);
				glp_set_col_name(lp, colnum(w, t), name);
				if (is_integer)
                                {
                                        glp_set_col_kind(lp, colnum(w, t), GLP_IV);
					glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0, 0);
                                }
				else
					glp_set_col_bnds(lp, colnum(w, t), GLP_LO, 0.0, 0.0);
			}
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
			{
				char name[32];
				snprintf(name, sizeof(name), "w%ds%dn", w, s);
				glp_set_col_name(lp, nw*nt+s*nw+w+1, name);
				if (is_integer)
                                {
                                        glp_set_col_kind(lp, nw*nt+s*nw+w+1, GLP_IV);
                                        glp_set_col_bnds(lp, nw*nt+s*nw+w+1, GLP_DB, 0, 1);
                                }
                                else
					glp_set_col_bnds(lp, nw*nt+s*nw+w+1, GLP_DB, 0.0, 1.0);
			}

		unsigned *sched_ctxs = in_sched_ctxs == NULL ? sc_hypervisor_get_sched_ctxs() : in_sched_ctxs;

		int curr_row_idx = 0;
		/* Total worker execution time */
		glp_add_rows(lp, nw*ns);
		for (t = 0; t < nt; t++)
		{
			int someone = 0;
			for (w = 0; w < nw; w++)
				if (!isnan(times[w][t]))
					someone = 1;
			if (!someone)
			{
				/* This task does not have any performance model at all, abort */
				printf("NO PERF MODELS\n");
				glp_delete_prob(lp);
				return 0.0;
			}
		}
		/*sum(t[t][w]*n[t][w]) < x[s][w]*tmax */
		for(s = 0; s < ns; s++)
		{
			for (w = 0; w < nw; w++)
			{
				char name[32], title[64];
				starpu_worker_get_name(w, name, sizeof(name));
				snprintf(title, sizeof(title), "worker %s", name);
				glp_set_row_name(lp, curr_row_idx+s*nw+w+1, title);
				for (t = 0, tp = tmp_task_pools; tp; t++, tp = tp->next)
				{
					if(tp->sched_ctx_id == sched_ctxs[s])
					{
						ia[n] = curr_row_idx+s*nw+w+1;
						ja[n] = colnum(w, t);
						if (isnan(times[w][t]))
						{
							printf("had to insert huge val \n");
							ar[n] = 1000000000.;
						}
						else
							ar[n] = times[w][t];
						n++;
					}
				}
				/* x[s][w] = 1 | 0 */
				ia[n] = curr_row_idx+s*nw+w+1;
				ja[n] = nw*nt+s*nw+w+1;
				ar[n] = (-1) * tmax;
				n++;
				if (is_integer)
                                {
					glp_set_row_bnds(lp, curr_row_idx+s*nw+w+1, GLP_UP, 0, 0);
                                }
                                else
					glp_set_row_bnds(lp, curr_row_idx+s*nw+w+1, GLP_UP, 0.0, 0.0);
			}
		}

		curr_row_idx += nw*ns;

		/* Total task completion */
		glp_add_rows(lp, nt);
		for (t = 0, tp = tmp_task_pools; tp; t++, tp = tp->next)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "task %s key %x", tp->cl->name, (unsigned) tp->footprint);
			glp_set_row_name(lp, curr_row_idx+t+1, title);
			for (w = 0; w < nw; w++)
			{
				ia[n] = curr_row_idx+t+1;
				ja[n] = colnum(w, t);
				ar[n] = 1;
				n++;
			}
			glp_set_row_bnds(lp, curr_row_idx+t+1, GLP_FX, tp->n, tp->n);
		}

		curr_row_idx += nt;

		/* sum(x[s][i]) = 1 */
		glp_add_rows(lp, nw);
		for (w = 0; w < nw; w++)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "w%x", w);
			glp_set_row_name(lp, curr_row_idx+w+1, title);
			for(s = 0; s < ns; s++)
			{
				ia[n] = curr_row_idx+w+1;
				ja[n] = nw*nt+s*nw+w+1;
				ar[n] = 1;
				n++;
			}
			if(is_integer)
                                glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1, 1);
			else
				glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1.0, 1.0);
		}
		if(n != ne)
			printf("ns= %d nw = %d nt = %d n = %d ne = %d\n", ns, nw, nt, n, ne);
		STARPU_ASSERT(n == ne);

		glp_load_matrix(lp, ne-1, ia, ja, ar);
	}

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	int ret = glp_simplex(lp, &parm);

	/* char str[50]; */
	/* sprintf(str, "outpu_lp_%g", tmax); */

	/* glp_print_sol(lp, str); */

	if (ret)
	{
		printf("error in simplex\n");
		glp_delete_prob(lp);
		lp = NULL;
		return 0.0;
	}

	int stat = glp_get_prim_stat(lp);
	/* if we don't have a solution return */
	if(stat == GLP_NOFEAS)
	{
		glp_delete_prob(lp);
//		printf("no_sol in tmax = %lf\n", tmax);
		lp = NULL;
		return 0.0;
	}


	if (is_integer)
        {
                glp_iocp iocp;
                glp_init_iocp(&iocp);
                iocp.msg_lev = GLP_MSG_OFF;
//		iocp.tm_lim = 1000;
		glp_intopt(lp, &iocp);
		int stat = glp_mip_status(lp);
		/* if we don't have a solution return */
		if(stat == GLP_NOFEAS || stat == GLP_ETMLIM || stat == GLP_UNDEF)
		{
//			printf("no int sol in tmax = %lf\n", tmax);
			if(stat == GLP_ETMLIM || stat == GLP_UNDEF)
				printf("timeout \n");
			glp_delete_prob(lp);
			lp = NULL;
			return 0.0;
		}
	}

	double res = glp_get_obj_val(lp);
	for (w = 0; w < nw; w++)
		for (t = 0; t < nt; t++)
			if (is_integer)
				tasks[w][t] = (double)glp_mip_col_val(lp, colnum(w, t));
                        else
				tasks[w][t] = glp_get_col_prim(lp, colnum(w, t));

	/* printf("**********************************************\n"); */
	/* printf("for tmax %lf\n", tmax); */
	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			if (is_integer)
				w_in_s[s][w] = (double)glp_mip_col_val(lp, nw*nt+s*nw+w+1);
                        else
				w_in_s[s][w] = glp_get_col_prim(lp, nw*nt+s*nw+w+1);
//			printf("w %d in ctx %d = %lf\n", w, s, w_in_s[s][w]);
		}
	/* printf("\n"); */
	/* printf("**********************************************\n"); */
	glp_delete_prob(lp);
	return res;
}

double sc_hypervisor_lp_simulate_distrib_flops(int ns, int nw, double v[ns][nw], double flops[ns], double res[ns][nw],
					       int  total_nw[nw], unsigned sched_ctxs[ns], double last_vmax)
{
	int integer = 1;
	int s, w;
	glp_prob *lp;

	int ne = (ns*nw+1)*(ns+nw)
		+ 1; /* glp dumbness */
	int n = 1;
	int ia[ne], ja[ne];
	double ar[ne];

	lp = glp_create_prob();

	glp_set_prob_name(lp, "sample");
	glp_set_obj_dir(lp, GLP_MAX);
        glp_set_obj_name(lp, "max speed");

	/* we add nw*ns columns one for each type of worker in each context
	   and another column corresponding to the 1/tmax bound (bc 1/tmax is a variable too)*/
	glp_add_cols(lp, nw*ns+1);

	/* struct sc_hypervisor_wrapper *sc_w = NULL; */
	for(s = 0; s < ns; s++)
	{
		/* sc_w = sc_hypervisor_get_wrapper(sched_ctxs[s]); */
		struct sc_hypervisor_policy_config *config = sc_hypervisor_get_config(sched_ctxs[s]);
		for(w = 0; w < nw; w++)
		{
			char name[32];
			snprintf(name, sizeof(name), "worker%dctx%d", w, s);
			glp_set_col_name(lp, n, name);

			if (integer)
			{
				glp_set_col_kind(lp, n, GLP_IV);
				/* if(sc_w->consider_max) */
				/* { */
				/* 	if(config->max_nworkers == 0) */
				/* 		glp_set_col_bnds(lp, n, GLP_FX, config->min_nworkers, config->max_nworkers); */
				/* 	else */
				/* 		glp_set_col_bnds(lp, n, GLP_DB, config->min_nworkers, config->max_nworkers); */
				/* } */
				/* else */
				{
					if(total_nw[w] == 0)
						glp_set_col_bnds(lp, n, GLP_FX, config->min_nworkers, total_nw[w]);
					else
						glp_set_col_bnds(lp, n, GLP_DB, config->min_nworkers, total_nw[w]);
				}
			}
			else
			{
/* 				if(sc_w->consider_max) */
/* 				{ */
/* 					if(config->max_nworkers == 0) */
/* 						glp_set_col_bnds(lp, n, GLP_FX, config->min_nworkers*1.0, config->max_nworkers*1.0); */
/* 					else */
/* 						glp_set_col_bnds(lp, n, GLP_DB, config->min_nworkers*1.0, config->max_nworkers*1.0); */
/* #ifdef STARPU_SC_HYPERVISOR_DEBUG */
/* 					printf("%d****************consider max %lf in lp\n", sched_ctxs[s], config->max_nworkers*1.0); */
/* #endif */
/* 				} */
/* 				else */
				{
					if(total_nw[w] == 0)
						glp_set_col_bnds(lp, n, GLP_FX, config->min_nworkers*1.0, total_nw[w]*1.0);
					else
						glp_set_col_bnds(lp, n, GLP_DB, config->min_nworkers*1.0, total_nw[w]*1.0);
#ifdef STARPU_SC_HYPERVISOR_DEBUG
					printf("%u****************don't consider max %d but total %d in lp\n", sched_ctxs[s], config->max_nworkers, total_nw[w]);
#endif
				}
			}
			n++;
		}
	}
#ifdef STARPU_SC_HYPERVISOR_DEBUG
	printf("ns = %d nw = %d\n", ns, nw);
#endif
	/*1/tmax should belong to the interval [0.0;1.0]*/
	glp_set_col_name(lp, n, "vmax");
//	glp_set_col_bnds(lp, n, GLP_DB, 0.0, 1.0);
	if(last_vmax != -1.0)
		glp_set_col_bnds(lp, n, GLP_LO, last_vmax, last_vmax);
	else
		glp_set_col_bnds(lp, n, GLP_LO, 0.0, 0.0);
	/* Z = 1/tmax -> 1/tmax structural variable, nCPUs & nGPUs in ctx are auxiliar variables */
	glp_set_obj_coef(lp, n, 1.0);

	n = 1;
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
			int s2;
			for(s2 = 0; s2 < ns; s2++)
			{
				if(s2 == s)
				{
					ia[n] = s+1;
					ja[n] = w + nw*s2 + 1;
					ar[n] = v[s][w];
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				else
				{
					ia[n] = s+1;
					ja[n] = w + nw*s2 + 1;
					ar[n] = 0.0;
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				n++;
			}
		}
		/* 1/tmax */
		ia[n] = s+1;
		ja[n] = ns*nw+1;
		ar[n] = (-1) * flops[s];
//		printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
		n++;
	}

	/*we add another linear constraint : sum(all cpus) = 9 and sum(all gpus) = 3 */
	glp_add_rows(lp, nw);

	for(w = 0; w < nw; w++)
	{
		char name[32];
		snprintf(name, sizeof(name), "w%d", w);
		glp_set_row_name(lp, ns+w+1, name);
		for(s = 0; s < ns; s++)
		{
			int w2;
			for(w2 = 0; w2 < nw; w2++)
			{
				if(w2 == w)
				{
					ia[n] = ns+w+1;
					ja[n] = w2+s*nw + 1;
					ar[n] = 1.0;
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				else
				{
					ia[n] = ns+w+1;
					ja[n] = w2+s*nw + 1;
					ar[n] = 0.0;
//					printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
				}
				n++;
			}
		}
		/* 1/tmax */
		ia[n] = ns+w+1;
		ja[n] = ns*nw+1;
		ar[n] = 0.0;
//		printf("ia[%d]=%d ja[%d]=%d ar[%d]=%lf\n", n, ia[n], n, ja[n], n, ar[n]);
		n++;

		/*sum(all gpus) = 3*/
		if(w == 0)
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, total_nw[0], total_nw[0]);

		/*sum(all cpus) = 9*/
		if(w == 1)
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, total_nw[1], total_nw[1]);
	}

	STARPU_ASSERT(n == ne);

	glp_load_matrix(lp, ne-1, ia, ja, ar);

	glp_smcp parm;
	glp_init_smcp(&parm);
  	parm.msg_lev = GLP_MSG_OFF;
	int ret = glp_simplex(lp, &parm);
	if (ret)
        {
                printf("error in simplex\n");
		glp_delete_prob(lp);
                lp = NULL;
                return 0.0;
        }

	int stat = glp_get_prim_stat(lp);
        /* if we don't have a solution return */
        if(stat == GLP_NOFEAS)
        {
                glp_delete_prob(lp);
		printf("no_sol\n");
                lp = NULL;
                return 0.0;
        }


	if (integer)
        {
                glp_iocp iocp;
                glp_init_iocp(&iocp);
                iocp.msg_lev = GLP_MSG_OFF;
                glp_intopt(lp, &iocp);
                int stat = glp_mip_status(lp);
                /* if we don't have a solution return */
                if(stat == GLP_NOFEAS)
                {
			printf("no int sol\n");
                        glp_delete_prob(lp);
                        lp = NULL;
                        return 0.0;
                }
        }

	double vmax = glp_get_obj_val(lp);
#ifdef STARPU_SC_HYPERVISOR_DEBUG
	printf("vmax = %lf \n", vmax);
#endif
	n = 1;
	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			if (integer)
                                res[s][w] = (double)glp_mip_col_val(lp, n);
			else
				res[s][w] = glp_get_col_prim(lp, n);
#ifdef STARPU_SC_HYPERVISOR_DEBUG
  			printf("%d/%d: res %lf flops = %lf v = %lf\n", w,s, res[s][w], flops[s], v[s][w]);
#endif
			n++;
		}
	}

	glp_delete_prob(lp);
	return vmax;
}

double sc_hypervisor_lp_simulate_distrib_flops_on_sample(int ns, int nw, double final_w_in_s[ns][nw], unsigned is_integer, double tmax,
							 double **speed, double flops[ns], double **final_flops_on_w)
{
	double w_in_s[ns][nw];
	double flops_on_w[ns][nw];

	int w, s;
	glp_prob *lp;

//	printf("try with tmax %lf\n", tmax);
	lp = glp_create_prob();
	glp_set_prob_name(lp, "StarPU theoretical bound");
	glp_set_obj_dir(lp, GLP_MAX);
	glp_set_obj_name(lp, "total execution time");

	{
		int ne = 5 * ns * nw /* worker execution time */
			+ 1; /* glp dumbness */
		int n = 1;
		int ia[ne], ja[ne];
		double ar[ne];


		/* Variables: number of flops assigned to worker w in context s, and
		 the acknwoledgment that the worker w belongs to the context s */
		glp_add_cols(lp, 2*nw*ns);
#define colnum_sample(w, s) ((s)*nw+(w)+1)
		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
				glp_set_obj_coef(lp, nw*ns+colnum_sample(w,s), 1.);

		for(s = 0; s < ns; s++)
			for(w = 0; w < nw; w++)
			{
				char name[32];
				snprintf(name, sizeof(name), "flopsw%ds%dn", w, s);
				glp_set_col_name(lp, colnum_sample(w,s), name);
				glp_set_col_bnds(lp, colnum_sample(w,s), GLP_LO, 0., 0.);

				snprintf(name, sizeof(name), "w%ds%dn", w, s);
				glp_set_col_name(lp, nw*ns+colnum_sample(w,s), name);
				if (is_integer)
				{
                                        glp_set_col_kind(lp, nw*ns+colnum_sample(w, s), GLP_IV);
					glp_set_col_bnds(lp, nw*ns+colnum_sample(w,s), GLP_DB, 0, 1);
				}
				else
					glp_set_col_bnds(lp, nw*ns+colnum_sample(w,s), GLP_DB, 0.0, 1.0);

			}


		int curr_row_idx = 0;
		/* Total worker execution time */
		glp_add_rows(lp, nw*ns);

		/*nflops[s][w]/v[s][w] < x[s][w]*tmax */
		for(s = 0; s < ns; s++)
		{
			for (w = 0; w < nw; w++)
			{
				char name[32], title[64];
				starpu_worker_get_name(w, name, sizeof(name));
				snprintf(title, sizeof(title), "worker %s", name);
				glp_set_row_name(lp, curr_row_idx+s*nw+w+1, title);

				/* nflosp[s][w] */
				ia[n] = curr_row_idx+s*nw+w+1;
				ja[n] = colnum_sample(w, s);
				ar[n] = 1 / speed[s][w];

				n++;

				/* x[s][w] = 1 | 0 */
				ia[n] = curr_row_idx+s*nw+w+1;
				ja[n] = nw*ns+colnum_sample(w,s);
				ar[n] = (-1) * tmax;
				n++;
				glp_set_row_bnds(lp, curr_row_idx+s*nw+w+1, GLP_UP, 0.0, 0.0);
			}
		}

		curr_row_idx += nw*ns;

		/* sum(flops[s][w]) = flops[s] */
		glp_add_rows(lp, ns);
		for (s = 0; s < ns; s++)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "flops %lf ctx%d", flops[s], s);
			glp_set_row_name(lp, curr_row_idx+s+1, title);
			for (w = 0; w < nw; w++)
			{
				ia[n] = curr_row_idx+s+1;
				ja[n] = colnum_sample(w, s);
				ar[n] = 1;
				n++;
			}
			glp_set_row_bnds(lp, curr_row_idx+s+1, GLP_FX, flops[s], flops[s]);
		}

		curr_row_idx += ns;

		/* sum(x[s][w]) = 1 */
		glp_add_rows(lp, nw);
		for (w = 0; w < nw; w++)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "w%x", w);
			glp_set_row_name(lp, curr_row_idx+w+1, title);
			for(s = 0; s < ns; s++)
			{
				ia[n] = curr_row_idx+w+1;
				ja[n] = nw*ns+colnum_sample(w,s);
				ar[n] = 1;
				n++;
			}
			if(is_integer)
				glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1, 1);
			else
				glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_FX, 1.0, 1.0);
		}

		curr_row_idx += nw;

		/* sum(nflops[s][w]) > 0*/
		glp_add_rows(lp, nw);
		for (w = 0; w < nw; w++)
		{
			char name[32], title[64];
			starpu_worker_get_name(w, name, sizeof(name));
			snprintf(title, sizeof(title), "flopsw%x", w);
			glp_set_row_name(lp, curr_row_idx+w+1, title);
			for(s = 0; s < ns; s++)
			{
				ia[n] = curr_row_idx+w+1;
				ja[n] = colnum_sample(w,s);
				ar[n] = 1;
				n++;
			}

			glp_set_row_bnds(lp, curr_row_idx+w+1, GLP_LO, 0.1, 0.);
		}

		if(n != ne)
			printf("ns= %d nw = %d n = %d ne = %d\n", ns, nw, n, ne);
		STARPU_ASSERT(n == ne);

		glp_load_matrix(lp, ne-1, ia, ja, ar);
	}

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	int ret = glp_simplex(lp, &parm);
	if (ret)
	{
		glp_delete_prob(lp);
		lp = NULL;
		return 0.0;
	}

        if (is_integer)
        {
                glp_iocp iocp;
                glp_init_iocp(&iocp);
                iocp.msg_lev = GLP_MSG_OFF;
                glp_intopt(lp, &iocp);
		int stat = glp_mip_status(lp);
		/* if we don't have a solution return */
		if(stat == GLP_NOFEAS)
		{
			glp_delete_prob(lp);
			lp = NULL;
			return 0.0;
		}
        }

	int stat = glp_get_prim_stat(lp);
	/* if we don't have a solution return */
	if(stat == GLP_NOFEAS)
	{
		glp_delete_prob(lp);
		lp = NULL;
		return 0.0;
	}

	double res = glp_get_obj_val(lp);

	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			flops_on_w[s][w] = glp_get_col_prim(lp, colnum_sample(w, s));
			if (is_integer)
				w_in_s[s][w] = (double)glp_mip_col_val(lp, nw*ns+colnum_sample(w, s));
			else
				w_in_s[s][w] = glp_get_col_prim(lp, nw*ns+colnum_sample(w,s));
//			printf("w_in_s[s%d][w%d] = %lf flops[s%d][w%d] = %lf \n", s, w, w_in_s[s][w], s, w, flops_on_w[s][w]);
		}

	glp_delete_prob(lp);
	for(s = 0; s < ns; s++)
		for(w = 0; w < nw; w++)
		{
			final_w_in_s[s][w] = w_in_s[s][w];
			final_flops_on_w[s][w] = flops_on_w[s][w];
		}

	return res;

}
#endif // STARPU_HAVE_GLPK_H
