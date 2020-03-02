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

#include "sc_hypervisor_lp.h"
#include "sc_hypervisor_policy.h"
#include <math.h>
#include <sys/time.h>

/* executes the function lp_estimated_distrib_func over the interval [tmin, tmax] until it finds the lowest value that
   still has solutions */
unsigned sc_hypervisor_lp_execute_dichotomy(int ns, int nw, double w_in_s[ns][nw], unsigned solve_lp_integer, void *specific_data,
					    double tmin, double tmax, double smallest_tmax,
					    double (*lp_estimated_distrib_func)(int ns, int nw, double draft_w_in_s[ns][nw],
									     unsigned is_integer, double tmax, void *specifc_data))
{
	double res = 1.0;
	unsigned has_sol = 0;
	double tmid = tmax;
	unsigned found_sol = 0;
	struct timeval start_time;
	struct timeval end_time;
	int nd = 0;
	double found_tmid = tmax;
	double potential_tmid = tmid;
	double threashold = tmax*0.1;
	gettimeofday(&start_time, NULL);

	/* we fix tmax and we do not treat it as an unknown
	   we just vary by dichotomy its values*/
	while(1)
	{
		/* find solution and save the values in draft tables
		   only if there is a solution for the system we save them
		   in the proper table */
		printf("solving for tmid %lf \n", tmid);
		res = lp_estimated_distrib_func(ns, nw, w_in_s, solve_lp_integer, tmid, specific_data);
		if(res < 0.0)
		{
			printf("timeouted no point in continuing\n");
			found_sol = 0;
			break;
		}
		else if(res != 0.0)
		{
			has_sol = 1;
			found_sol = 1;
			found_tmid = tmid;
			printf("found sol for tmid %lf \n", tmid);
		}
		else
		{
			printf("failed for tmid %lf \n", tmid);
			if(tmid == tmax)
			{
				printf("failed for tmid %lf from the first time\n", tmid);
				break;
			}
			has_sol = 0;
		}

		/* if we have a solution with this tmid try a smaller value
		   bigger than the old one */
		if(has_sol)
		{
			/* if the difference between tmax and tmid is smaller than
			   a given threashold there is no point in searching more
			   precision */
			tmax = tmid;
			potential_tmid = tmin + ((tmax-tmin)/2.0);
			if((tmax - potential_tmid) < threashold)
			{
				printf("had_sol but stop doing it for tmin %lf tmax %lf and potential tmid %lf \n", tmin, tmax, potential_tmid);
				break;
			}
			printf("try for smaller potential tmid %lf \n", potential_tmid);
		}
		else /*else try a bigger one */
		{
			/* if we previously found a good sol and we keep failing
			   we stop searching for a better sol */
			tmin = tmid;
			potential_tmid = tmin + ((tmax-tmin)/2.0);
			if((tmax - potential_tmid) < threashold)
			{
				printf("didn't have sol but stop doing it for tmin %lf tmax %lf and potential tmid %lf \n", tmin, tmax, potential_tmid);
				break;
			}
			printf("try for bigger potential tmid %lf \n", potential_tmid);
		}

		tmid = potential_tmid;

		nd++;
	}
	printf("solve againd for tmid %lf \n", found_tmid);
	if(found_sol)
	{
		res = lp_estimated_distrib_func(ns, nw, w_in_s, solve_lp_integer, found_tmid, specific_data);
		found_sol = (res != 0.0);
	}
	printf("found sol %u for tmid %lf\n", found_sol, found_tmid);
	gettimeofday(&end_time, NULL);

	long diff_s = end_time.tv_sec  - start_time.tv_sec;
	long diff_us = end_time.tv_usec  - start_time.tv_usec;

	__attribute__((unused)) float timing = (float)(diff_s*1000000 + diff_us)/1000;

	return found_sol;
}
