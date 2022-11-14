/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2017-2020  Federal University of Rio Grande do Sul (UFRGS)
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

LIST_TYPE(mpi_transfer,
	unsigned matched;
	int src;
	int dst;
	long mpi_tag;
	size_t size;
	float date;
	long jobid;
	double bandwidth;
	unsigned long handle;
	char *name;
	unsigned X;
	unsigned Y;
	unsigned type;
	int prio;
	long numa_nodes_bitmap;
);

struct starpu_fxt_mpi_offset _starpu_fxt_mpi_find_sync_points(char *filename_in, int *key, int *rank)
{
	struct starpu_fxt_mpi_offset offset;
	offset.nb_barriers = 0;
	offset.local_time_start = 0;
	offset.local_time_end = 0;
	offset.offset_start = 0;
	offset.offset_end = 0;

	/* Open the trace file */
	int fd_in;
	fd_in = open(filename_in, O_RDONLY);
	if (fd_in < 0)
	{
		perror("open failed :");
		_exit(EXIT_FAILURE);
	}

	static fxt_t fut;
	fut = fxt_fdopen(fd_in);
	if (!fut)
	{
		perror("fxt_fdopen :");
		_exit(EXIT_FAILURE);
	}

	fxt_blockev_t block;
	block = fxt_blockev_enter(fut);

	struct fxt_ev_64 ev;
	int ret;
	uint64_t local_sync_time;

	while (offset.nb_barriers < 2 && (ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev)) == FXT_EV_OK)
	{
		if (ev.code == _STARPU_MPI_FUT_BARRIER)
		{
			/* We found a sync point */
			*rank = ev.param[0];
			*key = ev.param[2];
			local_sync_time = (uint64_t) ((double) ev.param[3]); // It is stored as a double in the trace

			if (local_sync_time == 0)
			{
				/* This clock synchronization was made with an
				 * MPI_Barrier, consider the event timestamp as
				 * a local synchronized barrier time: */
				local_sync_time = ev.time;
			}

			if (offset.nb_barriers == 0)
			{
				offset.local_time_start = local_sync_time;
			}
			else
			{
				offset.local_time_end = local_sync_time;
			}

			offset.nb_barriers++;
		}
	}

	/* Close the trace file */
	if (close(fd_in))
	{
		perror("close failed :");
		_exit(EXIT_FAILURE);
	}

	return offset;
}

/*
 *	Deal with the actual MPI transfers performed with the MPI lib
 */

/* the list of MPI transfers found in the different traces */
static struct mpi_transfer *mpi_sends[STARPU_FXT_MAX_FILES] = {NULL};
static struct mpi_transfer *mpi_recvs[STARPU_FXT_MAX_FILES] = {NULL};

/* number of available slots in the lists  */
static unsigned mpi_sends_list_size[STARPU_FXT_MAX_FILES] = {0};
static unsigned mpi_recvs_list_size[STARPU_FXT_MAX_FILES] = {0};

/* number of slots actually used in the list  */
static unsigned mpi_sends_used[STARPU_FXT_MAX_FILES] = {0};
static unsigned mpi_recvs_used[STARPU_FXT_MAX_FILES] = {0};

/* number of slots already matched at the beginning of the list. This permits
 * going through the lists from the beginning to match each and every
 * transfer, thus avoiding a quadratic complexity. */
static unsigned mpi_recvs_matched[STARPU_FXT_MAX_FILES][STARPU_FXT_MAX_FILES] = { {0} };

void _starpu_fxt_mpi_add_send_transfer(int src, int dst, long mpi_tag, size_t size, float date, long jobid, unsigned long handle, unsigned type, int prio)
{
	STARPU_ASSERT(src >= 0);
	if (src >= STARPU_FXT_MAX_FILES)
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
	mpi_sends[src][slot].handle = handle;
	mpi_sends[src][slot].X = _starpu_fxt_data_get_coord(handle, src, 0);
	mpi_sends[src][slot].Y = _starpu_fxt_data_get_coord(handle, src, 1);
	const char *name = _starpu_fxt_data_get_name(handle, src);
	if (!name)
		name = "";
	mpi_sends[src][slot].name = strdup(name);
	mpi_sends[src][slot].type = type;
	mpi_sends[src][slot].prio = prio;
	mpi_sends[src][slot].numa_nodes_bitmap = -1;
}

void _starpu_fxt_mpi_send_transfer_set_numa_node(int src, int dest, long jobid, long numa_nodes_bitmap)
{
	STARPU_ASSERT(src >= 0);
	if (src >= STARPU_FXT_MAX_FILES || jobid == -1)
		return;

	unsigned i, slot;
	for (i = 0; i < mpi_sends_used[src]; i++)
	{
		/* The probe is just after the one handled by
		* _starpu_fxt_mpi_add_send_transfer, so the send transfer should have been
		* added recently: */
		slot = mpi_sends_used[src] - i - 1;
		if (mpi_sends[src][slot].dst == dest && mpi_sends[src][slot].jobid == jobid)
		{
			mpi_sends[src][slot].numa_nodes_bitmap = numa_nodes_bitmap;
			return;
		}
	}

	_STARPU_MSG("Warning: did not find the send transfer from %d to %d with jobid %ld\n", src, dest, jobid);
}

void _starpu_fxt_mpi_add_recv_transfer(int src, int dst, long mpi_tag, float date, long jobid, unsigned long handle)
{
	if (dst >= STARPU_FXT_MAX_FILES)
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
	mpi_recvs[dst][slot].handle = handle;
	mpi_recvs[dst][slot].numa_nodes_bitmap = -1;
}

void _starpu_fxt_mpi_recv_transfer_set_numa_node(int src, int dst, long jobid, long numa_nodes_bitmap)
{
	STARPU_ASSERT(src >= 0);
	if (src >= STARPU_FXT_MAX_FILES || jobid == -1)
		return;

	unsigned i, slot;
	for (i = 0; i < mpi_recvs_used[dst]; i++)
	{
		/* The probe is just after the one handled by
		* _starpu_fxt_mpi_add_send_transfer, so the send transfer should have been
		* added recently: */
		slot = mpi_recvs_used[dst] - i - 1;
		if (mpi_recvs[dst][slot].src == src && mpi_recvs[dst][slot].jobid == jobid)
		{
			mpi_recvs[dst][slot].numa_nodes_bitmap = numa_nodes_bitmap;
			return;
		}
	}

	_STARPU_MSG("Warning: did not find the recv transfer from %d to %d with jobid %ld\n", src, dst, jobid);
}


static
struct mpi_transfer *try_to_match_send_transfer(int src, int dst, long mpi_tag)
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

static const char* get_mpi_type_str(unsigned mpi_type)
{
	switch (mpi_type)
	{
		case _STARPU_MPI_FUT_POINT_TO_POINT_SEND:
			return "PointToPoint";
		case _STARPU_MPI_FUT_COLLECTIVE_SEND:
			return "Collective";
		default:
			return "Unknown";
	}
}

static void display_all_transfers_from_trace(FILE *out_paje_file, FILE *out_comms_file, unsigned n)
{
	unsigned slot[STARPU_FXT_MAX_FILES] = { 0 }, node;
	unsigned nb_wrong_comm_timing = 0;
	struct mpi_transfer_list pending_receives; /* Sorted list of matches which have not happened yet */
	double current_out_bandwidth[STARPU_FXT_MAX_FILES] = { 0. };
	double current_in_bandwidth[STARPU_FXT_MAX_FILES] = { 0. };
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

		src = STARPU_FXT_MAX_FILES;
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

		if (src == STARPU_FXT_MAX_FILES)
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
		unsigned long send_handle = cur->handle;
		long send_numa_nodes_bitmap = cur->numa_nodes_bitmap;

		if (dst < STARPU_FXT_MAX_FILES)
			match = try_to_match_send_transfer(src, dst, mpi_tag);
		else
			match = NULL;

		if (match)
		{
			float end_date = match->date;
			unsigned long recv_handle = match->handle;
			long recv_numa_nodes_bitmap = match->numa_nodes_bitmap;
			struct mpi_transfer *prev;

			if (end_date <= start_date)
				nb_wrong_comm_timing++;

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
			char str_priority[STARPU_POTI_STR_LEN];
			snprintf(str_priority, sizeof(str_priority), "%d", cur->prio);
			char str_handle[STARPU_POTI_STR_LEN];
			snprintf(str_handle, sizeof(str_handle), "%lx", send_handle);
			char X_str[STARPU_POTI_STR_LEN];
			snprintf(X_str, sizeof(X_str), "%u", cur->X);
			char Y_str[STARPU_POTI_STR_LEN];
			snprintf(Y_str, sizeof(Y_str), "%u", cur->Y);

			poti_user_StartLink(_starpu_poti_MpiLinkStart, start_date, "MPIroot", "MPIL", mpi_container, paje_value, paje_key, 7, str_mpi_tag, get_mpi_type_str(cur->type), str_priority, str_handle, name, X_str, Y_str);

			poti_SetVariable(start_date, mpi_container, "bwo_mpi", current_out_bandwidth[src]);
			snprintf(mpi_container, sizeof(mpi_container), "%d_mpict", dst);
			poti_EndLink(end_date, "MPIroot", "MPIL", mpi_container, paje_value, paje_key);
			poti_SetVariable(start_date, mpi_container, "bwo_mpi", current_in_bandwidth[dst]);
#else
			fprintf(out_paje_file, "13	%.9f	%d_mpict	bwo_mpi	%f\n", start_date, src, current_out_bandwidth[src]);
			fprintf(out_paje_file, "13	%.9f	%d_mpict	bwi_mpi	%f\n", start_date, dst, current_in_bandwidth[dst]);
			fprintf(out_paje_file, "23	%.9f	MPIL	MPIroot	%lu	%d_mpict	mpicom_%lu	%ld	%s	%d	%lx	\"%s\"	%u	%u\n", start_date, (unsigned long)size, src, id, mpi_tag, get_mpi_type_str(cur->type), cur->prio, send_handle, cur->name, cur->X, cur->Y);
			fprintf(out_paje_file, "19	%.9f	MPIL	MPIroot	%lu	%d_mpict	mpicom_%lu\n", end_date, (unsigned long)size, dst, id);
#endif

			if (out_comms_file != NULL)
			{
				fprintf(out_comms_file, "Src: %d\n", src);
				fprintf(out_comms_file, "Dst: %d\n", dst);
				fprintf(out_comms_file, "Tag: %ld\n", mpi_tag);
				fprintf(out_comms_file, "SendTime: %.9f\n", start_date);
				fprintf(out_comms_file, "RecvTime: %.9f\n", end_date);
				fprintf(out_comms_file, "SendHandle: %lx\n", send_handle);
				fprintf(out_comms_file, "RecvHandle: %lx\n", recv_handle);
				if (cur->jobid != -1)
					fprintf(out_comms_file, "SendJobId: %d_%ld\n", src, cur->jobid);
				if (match->jobid != -1)
					fprintf(out_comms_file, "RecvJobId: %d_%ld\n", dst, match->jobid);
				fprintf(out_comms_file, "Size: %lu\n", (unsigned long)size);
				fprintf(out_comms_file, "Priority: %d\n", cur->prio);
				fprintf(out_comms_file, "Type: %s\n", get_mpi_type_str(cur->type));
				char str[STARPU_TRACE_STR_LEN] = "";
				_starpu_convert_numa_nodes_bitmap_to_str(send_numa_nodes_bitmap, str);
				fprintf(out_comms_file, "SendNumaNodes: %s\n", str);
				_starpu_convert_numa_nodes_bitmap_to_str(recv_numa_nodes_bitmap, str);
				fprintf(out_comms_file, "RecvNumaNodes: %s\n", str);
				fprintf(out_comms_file, "\n");
			}
			free(cur->name);
		}
		else
		{
			_STARPU_DISP("Warning, could not match MPI transfer from %d to %d (tag %lx) starting at %f\n", src, dst, mpi_tag, start_date);
		}

		slot[src]++;
	}

	if (nb_wrong_comm_timing == 1)
		_STARPU_MSG("Warning: a communication finished before it started !\n");
	else if (nb_wrong_comm_timing > 1)
		_STARPU_MSG("Warning: %u communications finished before they started !\n", nb_wrong_comm_timing);
}

void _starpu_fxt_display_mpi_transfers(struct starpu_fxt_options *options, int *ranks STARPU_ATTRIBUTE_UNUSED, FILE *out_paje_file, FILE* out_comms_file)
{
	if (options->ninputfiles > STARPU_FXT_MAX_FILES)
	{
		_STARPU_DISP("Warning: %u files given, maximum %u supported, truncating to %u\n", options->ninputfiles, STARPU_FXT_MAX_FILES, STARPU_FXT_MAX_FILES);
		options->ninputfiles = STARPU_FXT_MAX_FILES;
	}

	/* display the MPI transfers if possible */
	if (out_paje_file)
		display_all_transfers_from_trace(out_paje_file, out_comms_file, options->ninputfiles);
}

#endif // STARPU_USE_FXT
