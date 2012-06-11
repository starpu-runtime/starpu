#include "lp_tools.h"

#ifdef HAVE_GLPK_H

static double _glp_get_nworkers_per_ctx(int ns, int nw, double v[ns][nw], double flops[ns], double res[ns][nw])
{
	int s, w;
	glp_prob *lp;
	
	int ne =
		(ns*nw+1)*(ns+nw)
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

	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			char name[32];
			snprintf(name, sizeof(name), "worker%dctx%d", w, s);
			glp_set_col_name(lp, n, name);
			glp_set_col_bnds(lp, n, GLP_LO, 0.3, 0.0);
			n++;
		}
	}

	/*1/tmax should belong to the interval [0.0;1.0]*/
	glp_set_col_name(lp, n, "vmax");
	glp_set_col_bnds(lp, n, GLP_DB, 0.0, 1.0);
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
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, 3., 3.);

		/*sum(all cpus) = 9*/
		if(w == 1) 
			glp_set_row_bnds(lp, ns+w+1, GLP_FX, 9., 9.);
	}

	STARPU_ASSERT(n == ne);

	glp_load_matrix(lp, ne-1, ia, ja, ar);

	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.msg_lev = GLP_MSG_OFF;
	glp_simplex(lp, &parm);
	
	double vmax = glp_get_obj_val(lp);

	n = 1;
	for(s = 0; s < ns; s++)
	{
		for(w = 0; w < nw; w++)
		{
			res[s][w] = glp_get_col_prim(lp, n);
			n++;
		}
	}

	glp_delete_prob(lp);
	return vmax;
}

#endif //HAVE_GLPK_H

double _lp_get_nworkers_per_ctx(int nsched_ctxs, int ntypes_of_workers, double res[nsched_ctxs][ntypes_of_workers])
{
	int *sched_ctxs = sched_ctx_hypervisor_get_sched_ctxs();
	double v[nsched_ctxs][ntypes_of_workers];
	double flops[nsched_ctxs];
	int i = 0;
	struct sched_ctx_wrapper* sc_w;
	for(i = 0; i < nsched_ctxs; i++)
	{
		sc_w = sched_ctx_hypervisor_get_wrapper(sched_ctxs[i]);
		v[i][0] = 200.0;//_get_velocity_per_worker_type(sc_w, STARPU_CUDA_WORKER);
		v[i][1] = 20.0;//_get_velocity_per_worker_type(sc_w, STARPU_CPU_WORKER);
		flops[i] = sc_w->remaining_flops/1000000000; //sc_w->total_flops/1000000000; /* in gflops*/
//			printf("%d: flops %lf\n", sched_ctxs[i], flops[i]);
	}

#ifdef HAVE_GLPK_H	
	return 1/_glp_get_nworkers_per_ctx(nsched_ctxs, ntypes_of_workers, v, flops, res);
#else
	return 0.0;
#endif
}

double _lp_get_tmax()
{
	int nsched_ctxs = sched_ctx_hypervisor_get_nsched_ctxs();
	
	double res[nsched_ctxs][2];
	return _lp_get_nworkers_per_ctx(nsched_ctxs, 2, res) * 1000;
}
