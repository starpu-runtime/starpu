/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2017,2019  Federal University of Rio Grande do Sul (UFRGS)
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

#include <starpu.h>
#include <common/config.h>

#ifdef STARPU_USE_FXT

#include "starpu_fxt.h"
#ifdef STARPU_HAVE_POTI
#include <poti.h>
#define STARPU_POTI_STR_LEN 200
#endif

LIST_TYPE(mpi_transfer,
	unsigned matched;
	int src;
	int dst;
	long mpi_tag;
	size_t size;
	float date;
	long jobid;
	double bandwidth;
);

/* Returns 0 if a barrier is found, -1 otherwise. In case of success, offset is
 * filled with the timestamp of the barrier */
int _starpu_fxt_mpi_find_sync_point(char *filename_in, uint64_t *offset, int *key, int *rank)
{
	STARPU_ASSERT(offset);

	/* Open the trace file */
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0)
	{
	        perror("open failed :");
	        exit(-1);
	}

	static fxt_t fut;
	fut = fxt_fdopen(fd_in);
	if (!fut)
	{
	        perror("fxt_fdopen :");
	        exit(-1);
	}

	fxt_blockev_t block;
	block = fxt_blockev_enter(fut);

	struct fxt_ev_64 ev;

	int func_ret = -1;
	unsigned found = 0;
	while(!found)
	{
		int ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev);
		if (ret != FXT_EV_OK)
		{
			_STARPU_MSG("no more block ...\n");
			break;
		}

		if (ev.code == _STARPU_MPI_FUT_BARRIER)
		{
			/* We found the sync point */
			*offset = ev.time;
			*rank = ev.param[0];
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

/*
 *	Deal with the actual MPI transfers performed with the MPI lib
 */

/* the list of MPI transfers found in the different traces */
static struct mpi_transfer *mpi_sends[MAX_MPI_NODES] = {NULL};
static struct mpi_transfer *mpi_recvs[MAX_MPI_NODES] = {NULL};

/* number of available slots in the lists  */
unsigned mpi_sends_list_size[MAX_MPI_NODES] = {0};
unsigned mpi_recvs_list_size[MAX_MPI_NODES] = {0};

/* number of slots actually used in the list  */
unsigned mpi_sends_used[MAX_MPI_NODES] = {0};
unsigned mpi_recvs_used[MAX_MPI_NODES] = {0};

/* number of slots already matched at the beginning of the list. This permits
 * going through the lists from the beginning to match each and every
 * transfer, thus avoiding a quadratic complexity. */
unsigned mpi_recvs_matched[MAX_MPI_NODES][MAX_MPI_NODES] = { {0} };
unsigned mpi_sends_matched[MAX_MPI_NODES][MAX_MPI_NODES] = { {0} };

void _starpu_fxt_mpi_add_send_transfer(int src, int dst STARPU_ATTRIBUTE_UNUSED, long mpi_tag, size_t size, float date, long jobid)
{
	STARPU_ASSERT(src >= 0);
	if (src >= MAX_MPI_NODES)
		return;
	unsigned slot = mpi_sends_used[src]++;

	if (mpi_sends_used[src] > mpi_sends_list_size[src])
	{
		if (mpi_sends_list_size[src] > 0)
		{
			mpi_sends_list_size[src] *= 2;
		}
		else
		{
			mpi_sends_list_size[src] = 1;
		}

		_STARPU_REALLOC(mpi_sends[src], mpi_sends_list_size[src]*sizeof(struct mpi_transfer));
	}

	mpi_sends[src][slot].matched = 0;
	mpi_sends[src][slot].src = src;
	mpi_sends[src][slot].dst = dst;
	mpi_sends[src][slot].mpi_tag = mpi_tag;
	mpi_sends[src][slot].size = size;
	mpi_sends[src][slot].date = date;
	mpi_sends[src][slot].jobid = jobid;
}

void _starpu_fxt_mpi_add_recv_transfer(int src STARPU_ATTRIBUTE_UNUSED, int dst, long mpi_tag, float date, long jobid)
{
	if (dst >= MAX_MPI_NODES)
		return;
	unsigned slot = mpi_recvs_used[dst]++;

	if (mpi_recvs_used[dst] > mpi_recvs_list_size[dst])
	{
		if (mpi_recvs_list_size[dst] > 0)
		{
			mpi_recvs_list_size[dst] *= 2;
		}
		else
		{
			mpi_recvs_list_size[dst] = 1;
		}

		_STARPU_REALLOC(mpi_recvs[dst], mpi_recvs_list_size[dst]*sizeof(struct mpi_transfer));
	}

	mpi_recvs[dst][slot].matched = 0;
	mpi_recvs[dst][slot].src = src;
	mpi_recvs[dst][slot].dst = dst;
	mpi_recvs[dst][slot].mpi_tag = mpi_tag;
	mpi_recvs[dst][slot].date = date;
	mpi_recvs[dst][slot].jobid = jobid;
}

static
struct mpi_transfer *try_to_match_send_transfer(int src STARPU_ATTRIBUTE_UNUSED, int dst, long mpi_tag)
{
	unsigned slot;
	unsigned firstslot = mpi_recvs_matched[src][dst];

	unsigned all_previous_were_matched = 1;

	for (slot = firstslot; slot < mpi_recvs_used[dst]; slot++)
	{
		if (!mpi_recvs[dst][slot].matched)
		{
			if (mpi_recvs[dst][slot].mpi_tag == mpi_tag)
			{
				/* we found a match ! */
				mpi_recvs[dst][slot].matched = 1;
				return &mpi_recvs[dst][slot];
			}

			all_previous_were_matched = 0;
		}
		else
		{
			if (all_previous_were_matched)
			{
				/* All previous transfers are already matched,
				 * we need not consider them anymore */
				mpi_recvs_matched[src][dst] = slot;
			}
		}
	}

	/* If we reached that point, we could not find a match */
	return NULL;
}

static unsigned long mpi_com_id = 0;

static void display_all_transfers_from_trace(FILE *out_paje_file, unsigned n)
{
	unsigned slot[MAX_MPI_NODES] = { 0 }, node;
	struct mpi_transfer_list pending_receives; /* Sorted list of matches which have not happened yet */
	double current_out_bandwidth[MAX_MPI_NODES] = { 0. };
	double current_in_bandwidth[MAX_MPI_NODES] = { 0. };
#ifdef STARPU_HAVE_POTI
	char mpi_container[STARPU_POTI_STR_LEN];
#endif

	//bwi_mpi and bwo_mpi are set to zero when MPI thread containers are created

	mpi_transfer_list_init(&pending_receives);

	while (1)
	{
		float start_date;
		struct mpi_transfer *cur, *match;
		int src;

		/* Find out which event comes first: a pending receive, or a new send */

		if (mpi_transfer_list_empty(&pending_receives))
			start_date = INFINITY;
		else
			start_date = mpi_transfer_list_front(&pending_receives)->date;

		src = MAX_MPI_NODES;
		for (node = 0; node < n; node++)
		{
			if (slot[node] < mpi_sends_used[node] && mpi_sends[node][slot[node]].date < start_date)
			{
				/* next send for node is earlier than others */
				src = node;
				start_date = mpi_sends[src][slot[src]].date;
			}
		}
		if (start_date == INFINITY)
			/* No event any more, we're finished! */
			break;

		if (src == MAX_MPI_NODES)
		{
			/* Pending match is earlier than all new sends, finish its communication */
			match = mpi_transfer_list_pop_front(&pending_receives);
			current_out_bandwidth[match->src] -= match->bandwidth;
			current_in_bandwidth[match->dst] -= match->bandwidth;
#ifdef STARPU_HAVE_POTI
			snprintf(mpi_container, sizeof(mpi_container), "%d_mpict", match->src);
			poti_SetVariable(match->date, mpi_container, "bwo_mpi", current_out_bandwidth[match->src]);
			snprintf(mpi_container, sizeof(mpi_container), "%d_mpict", match->dst);
			poti_SetVariable(match->date, mpi_container, "bwi_mpi", current_in_bandwidth[match->dst]);
#else
			fprintf(out_paje_file, "13	%.9f	%d_mpict	bwo_mpi	%f\n", match->date, match->src, current_out_bandwidth[match->src]);
			fprintf(out_paje_file, "13	%.9f	%d_mpict	bwi_mpi	%f\n", match->date, match->dst, current_in_bandwidth[match->dst]);
#endif
			continue;
		}

		cur = &mpi_sends[src][slot[src]];
		int dst = cur->dst;
		long mpi_tag = cur->mpi_tag;
		size_t size = cur->size;

		if (dst < MAX_MPI_NODES)
			match = try_to_match_send_transfer(src, dst, mpi_tag);
		else
			match = NULL;

		if (match)
		{
			float end_date = match->date;
			struct mpi_transfer *prev;

			match->bandwidth = (0.001*size)/(end_date - start_date);
			current_out_bandwidth[src] += match->bandwidth;
			current_in_bandwidth[dst] += match->bandwidth;

			/* Insert in sorted list, most probably at the end so let's use a mere insertion sort */
			for (prev = mpi_transfer_list_last(&pending_receives);
				prev != mpi_transfer_list_alpha(&pending_receives);
				prev = mpi_transfer_list_prev(prev))
				if (prev->date <= end_date)
				{
					/* Found its place */
					mpi_transfer_list_insert_after(&pending_receives, match, prev);
					break;
				}
			if (prev == mpi_transfer_list_alpha(&pending_receives))
			{
				/* No element earlier than this one, put it at the head */
				mpi_transfer_list_push_front(&pending_receives, match);
			}

			unsigned long id = mpi_com_id++;
			if (cur->jobid != -1)
				_starpu_fxt_dag_add_send(src, cur->jobid, mpi_tag, id);
			if (match->jobid != -1)
				_starpu_fxt_dag_add_receive(dst, match->jobid, mpi_tag, id);
#ifdef STARPU_HAVE_POTI
			char paje_value[STARPU_POTI_STR_LEN], paje_key[STARPU_POTI_STR_LEN];
			snprintf(paje_value, sizeof(paje_value), "%lu", (long unsigned) size);
			snprintf(paje_key, sizeof(paje_key), "mpicom_%lu", id);
			snprintf(mpi_container, sizeof(mpi_container), "%d_mpict", src);

			char str_mpi_tag[STARPU_POTI_STR_LEN];
			snprintf(str_mpi_tag, sizeof(str_mpi_tag), "%ld", mpi_tag);
			poti_user_StartLink(_starpu_poti_MpiLinkStart, start_date, "MPIroot", "MPIL", mpi_container, paje_value, paje_key, 1, str_mpi_tag);

			poti_SetVariable(start_date, mpi_container, "bwo_mpi", current_out_bandwidth[src]);
			snprintf(mpi_container, sizeof(mpi_container), "%d_mpict", dst);
			poti_EndLink(end_date, "MPIroot", "MPIL", mpi_container, paje_value, paje_key);
			poti_SetVariable(start_date, mpi_container, "bwo_mpi", current_in_bandwidth[dst]);
#else
			fprintf(out_paje_file, "13	%.9f	%d_mpict	bwo_mpi	%f\n", start_date, src, current_out_bandwidth[src]);
			fprintf(out_paje_file, "13	%.9f	%d_mpict	bwi_mpi	%f\n", start_date, dst, current_in_bandwidth[dst]);
			fprintf(out_paje_file, "23	%.9f	MPIL	MPIroot	%lu	%d_mpict	mpicom_%lu	%ld\n", start_date, (unsigned long)size, src, id, mpi_tag);
			fprintf(out_paje_file, "19	%.9f	MPIL	MPIroot	%lu	%d_mpict	mpicom_%lu\n", end_date, (unsigned long)size, dst, id);
#endif
		}
		else
		{
			_STARPU_DISP("Warning, could not match MPI transfer from %d to %d (tag %lx) starting at %f\n", src, dst, mpi_tag, start_date);
		}

		slot[src]++;
	}
}

void _starpu_fxt_display_mpi_transfers(struct starpu_fxt_options *options, int *ranks STARPU_ATTRIBUTE_UNUSED, FILE *out_paje_file)
{
	if (options->ninputfiles > MAX_MPI_NODES)
	{
		_STARPU_DISP("Warning: %u files given, maximum %u supported, truncating to %u\n", options->ninputfiles, MAX_MPI_NODES, MAX_MPI_NODES);
		options->ninputfiles = MAX_MPI_NODES;
	}

	/* display the MPI transfers if possible */
	if (out_paje_file)
		display_all_transfers_from_trace(out_paje_file, options->ninputfiles);
}

#endif // STARPU_USE_FXT
