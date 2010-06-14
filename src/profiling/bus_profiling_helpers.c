/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <starpu_profiling.h>
#include <profiling/profiling.h>

void starpu_bus_profiling_helper_display_summary(void)
{
	int long long sum_transferred = 0;

	fprintf(stderr, "Data transfer statistics:\n");

	int busid;
	int bus_cnt = starpu_bus_get_count();
	for (busid = 0; busid < bus_cnt; busid++)
	{
		int src, dst;
	
		src = starpu_bus_get_src(busid);
		dst = starpu_bus_get_dst(busid);

		struct starpu_bus_profiling_info bus_info;
		starpu_bus_get_profiling_info(busid, &bus_info);

		int long long transferred = bus_info.transferred_bytes;
		int transfer_cnt =  bus_info.transfer_count;
		double elapsed_time = starpu_timing_timespec_to_us(&bus_info.total_time);

		fprintf(stderr, "\t%d -> %d\t%.2lf MB\t%.2lfMB/s\t(transfers : %d - avg %.2lf MB)\n", src, dst, (1.0*transferred)/(1024*1024), transferred/elapsed_time, transfer_cnt, (1.0*transferred)/(transfer_cnt*1024*1024));

		sum_transferred += transferred;
	}

	fprintf(stderr, "Total transfers: %.2lf MB\n", (1.0*sum_transferred)/(1024*1024));
}
