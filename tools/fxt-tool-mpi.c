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

#include "fxt-tool.h"

/* Returns 0 if a barrier is found, -1 otherwise. In case of success, offset is
 * filled with the timestamp of the barrier */
int find_sync_point(char *filename_in, uint64_t *offset, int *key)
{
	STARPU_ASSERT(offset);

	/* Open the trace file */
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0) {
	        perror("open failed :");
	        exit(-1);
	}

	static fxt_t fut;
	fut = fxt_fdopen(fd_in);
	if (!fut) {
	        perror("fxt_fdopen :");
	        exit(-1);
	}
	
	fxt_blockev_t block;
	block = fxt_blockev_enter(fut);

	struct fxt_ev_64 ev;

	int func_ret = -1;
	unsigned found = 0;
	while(!found) {
		int ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev);
		if (ret != FXT_EV_OK) {
			fprintf(stderr, "no more block ...\n");
			break;
		}

		if (ev.code == FUT_MPI_BARRIER)
		{
			/* We found the sync point */
			*offset = ev.time;
			*key = ev.param[2];
			found = 1;
			func_ret = 0;
		}
	}

	/* Close the trace file */
	if (close(fd_in))
	{
	        perror("close failed :");
	        exit(-1);
	}

	return func_ret;
}


