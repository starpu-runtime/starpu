/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011 - 2013  INRIA 
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
	double old_tmax = 0.0;
	unsigned found_sol = 0;

	struct timeval start_time;
	struct timeval end_time;
	int nd = 0;
	gettimeofday(&start_time, NULL);

	/* we fix tmax and we do not treat it as an unknown
	   we just vary by dichotomy its values*/
	while(tmax > 1.0)
	{
		/* find solution and save the values in draft tables
		   only if there is a solution for the system we save them
		   in the proper table */
		res = lp_estimated_distrib_func(ns, nw, w_in_s, solve_lp_integer, tmax, specific_data);
		if(res != 0.0)
		{
			has_sol = 1;
			found_sol = 1;
		}
		else
			has_sol = 0;

		/* if we have a solution with this tmax try a smaller value
		   bigger than the old min */
		if(has_sol)
		{
			if(old_tmax != 0.0 && (old_tmax - tmax) < 0.5)
				break;
			old_tmax = tmax;
		}
		else /*else try a bigger one but smaller than the old tmax */
		{
			tmin = tmax;
			if(old_tmax != 0.0)
				tmax = old_tmax;
		}
		if(tmin == tmax) break;
		tmax = sc_hypervisor_lp_find_tmax(tmin, tmax);

		if(tmax < smallest_tmax)
		{
			tmax = old_tmax;
			tmin = smallest_tmax;
			tmax = sc_hypervisor_lp_find_tmax(tmin, tmax);
		}
		nd++;
	}
	gettimeofday(&end_time, NULL);

	long diff_s = end_time.tv_sec  - start_time.tv_sec;
	long diff_us = end_time.tv_usec  - start_time.tv_usec;

	float timing = (float)(diff_s*1000000 + diff_us)/1000;

	return found_sol;
}

