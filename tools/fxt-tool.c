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

/*
 *	Default user options
 */

static unsigned per_task_colour = 0;
static unsigned generate_distrib = 0;
static unsigned no_counter = 0;
static unsigned no_bus = 0;

/* TODO don't make that global ? */
struct fxt_ev_64 ev;
/* In case we are going to gather multiple traces (eg in the case of MPI
 * processes), we may need to prefix the name of the containers. */
char *prefix = "";
uint64_t offset = 0;
int rank = -1;

static uint64_t start_time = 0;
static uint64_t end_time = 0;

static int nworkers = 0;

//static char *filename = NULL;
/* XXX remove the 64 ... */
unsigned ninputfiles = 0;
static char *filenames[64];

LIST_TYPE(symbol_name,
	char *name;
);

static symbol_name_list_t symbol_list;

LIST_TYPE(communication,
	unsigned comid;
	float comm_start;	
	float bandwith;
	unsigned node;
);

static communication_list_t communication_list;

/*
 * Paje trace file tools
 */

static char *out_paje_path = "paje.trace";
static FILE *out_paje_file;

static char *distrib_time_path = "distrib.data";
static FILE *distrib_time;

static void paje_output_file_init(void)
{
	/* create a new file */
	out_paje_file = fopen(out_paje_path, "w+");

	write_paje_header(out_paje_file);

	fprintf(out_paje_file, "                                        \n \
	1       MPIP      0       \"MPI Program\"                      	\n \
	1       P      MPIP       \"Program\"                      	\n \
	1       Mn      P       \"Memory Node\"                         \n \
	1       T      Mn       \"Worker\"                               \n \
	1       Sc       P       \"Scheduler State\"                        \n \
	2       event   T       \"event type\"				\n \
	3       S       T       \"Thread State\"                        \n \
	3       MS       Mn       \"Memory Node State\"                        \n \
	4       ntask    Sc       \"Number of tasks\"                        \n \
	4       bw      Mn       \"Bandwith\"                        \n \
	6       I       S      Initializing       \"0.0 .7 1.0\"            \n \
	6       D       S      Deinitializing       \"0.0 .1 .7\"            \n \
	6       Fi       S      FetchingInput       \"1.0 .1 1.0\"            \n \
	6       Po       S      PushingOutput       \"0.1 1.0 1.0\"            \n \
	6       E       S       Executing       \".0 .6 .4\"            \n \
	6       C       S       Callback       \".0 .3 .8\"            \n \
	6       B       S       Blocked         \".9 .1 .0\"		\n \
	6       P       S       Progressing         \".4 .1 .6\"		\n \
	6       A       MS      Allocating         \".4 .1 .0\"		\n \
	6       Ar       MS      AllocatingReuse       \".1 .1 .8\"		\n \
	6       R       MS      Reclaiming         \".0 .1 .4\"		\n \
	6       Co       MS     DriverCopy         \".3 .5 .1\"		\n \
	6       No       MS     Nothing         \".0 .0 .0\"		\n \
	5       MPIL     MPIP	P	P      MPIL\n \
	5       L       P	Mn	Mn      L\n");

	fprintf(out_paje_file, "7      0.0 MPIroot      MPIP      0       root\n");
}

/*
 * Generic tools
 */

static float get_event_time_stamp(void)
{
	return (float)((ev.time-offset)/1000000.0);
}

static int register_worker_id(unsigned long tid)
{
	int workerid = nworkers++;

	/* create a new key in the htable */
	char *tidstr = malloc(16*sizeof(char));
	sprintf(tidstr, "%ld", tid);

	ENTRY item;
		item.key = tidstr;
		item.data = (void *)(uintptr_t)workerid;

	ENTRY *res;
	res = hsearch(item, FIND);

	/* only register a thread once */
	STARPU_ASSERT(res == NULL);

	res = hsearch(item, ENTER);
	STARPU_ASSERT(res);

	return workerid;
}

static int find_worker_id(unsigned long tid)
{
	char tidstr[16];
	sprintf(tidstr, "%ld", tid);

	ENTRY item;
		item.key = tidstr;
		item.data = NULL;
	ENTRY *res;
	res = hsearch(item, FIND);
	if (!res)
		return -1;

	int id = (uintptr_t)(res->data);

	return id;
}

/*
 *	Initialization
 */

static void handle_new_mem_node(void)
{
	fprintf(out_paje_file, "7       %f	%ld      Mn      %sp	%sMEMNODE%ld\n", get_event_time_stamp(), ev.param[0], prefix, prefix, ev.param[0]);

	if (!no_bus)
		fprintf(out_paje_file, "13       %f bw %sMEMNODE%d 0.0\n", 0.0f, prefix, ev.param[0]);
}

static void handle_worker_init_start(void)
{
	/* 
	   arg0 : type of worker (cuda, core ..)
	   arg1 : memory node
	   arg2 : thread id 
	*/
	fprintf(out_paje_file, "7       %f	%s%ld      T      %sMEMNODE%ld       %s%ld\n",
		get_event_time_stamp(), prefix, ev.param[2], prefix, ev.param[1], prefix, ev.param[2]);

	int workerid = register_worker_id(ev.param[2]);

	switch (ev.param[0]) {
		case FUT_APPS_KEY:
			set_next_other_worker_color(workerid);
			break;
		case FUT_CORE_KEY:
			set_next_cpu_worker_color(workerid);
			break;
		case FUT_CUDA_KEY:
			set_next_cuda_worker_color(workerid);
			break;
		default:
			STARPU_ABORT();
	}

	/* start initialization */
	fprintf(out_paje_file, "10       %f     S      %s%ld      I\n",
			get_event_time_stamp(), prefix, ev.param[2]);
}

static void handle_worker_init_end(void)
{
	fprintf(out_paje_file, "10       %f     S      %s%ld      B\n",
			get_event_time_stamp(), prefix, ev.param[0]);
}

static void handle_worker_deinit_start(void)
{
	fprintf(out_paje_file, "10       %f     S      %s%ld      D\n",
			get_event_time_stamp(), prefix, ev.param[0]);
}

static void handle_worker_deinit_end(void)
{
	fprintf(out_paje_file, "8       %f	%s%ld	T\n",
			get_event_time_stamp(), prefix, ev.param[1]);
}

static void create_paje_state_if_not_found(char *name)
{
	symbol_name_itor_t itor;
	for (itor = symbol_name_list_begin(symbol_list);
		itor != symbol_name_list_end(symbol_list);
		itor = symbol_name_list_next(itor))
	{
		if (!strcmp(name, itor->name))
		{
			/* we found an entry */
			return;
		}
	}

	/* it's the first time ... */
	symbol_name_t entry = symbol_name_new();
		entry->name = malloc(strlen(name));
		strcpy(entry->name, name);

	symbol_name_list_push_front(symbol_list, entry);
	
	/* choose some colour ... that's disguting yes */
	unsigned hash_symbol_red = get_colour_symbol_red(name);
	unsigned hash_symbol_green = get_colour_symbol_green(name);
	unsigned hash_symbol_blue = get_colour_symbol_blue(name);

	fprintf(stderr, "name %s hash red %d green %d blue %d \n", name, hash_symbol_red, hash_symbol_green, hash_symbol_blue);

	uint32_t hash_sum = hash_symbol_red + hash_symbol_green + hash_symbol_blue;

	float red = (1.0f * hash_symbol_red) / hash_sum;
	float green = (1.0f * hash_symbol_green) / hash_sum;
	float blue = (1.0f * hash_symbol_blue) / hash_sum;

	/* create the Paje state */
	fprintf(out_paje_file, "6       %s       S       %s \"%f %f %f\" \n", name, red, green, blue, name);
}

static double last_codelet_start[MAXWORKERS];
static uint64_t last_codelet_hash[MAXWORKERS];
static char last_codelet_symbol[128][MAXWORKERS];

static void handle_start_codelet_body(void)
{
	int worker;
	worker = find_worker_id(ev.param[1]);

	if (worker < 0) return;

	unsigned long has_name = ev.param[2];
	char *name = has_name?(char *)&ev.param[3]:"unknown";

	snprintf(last_codelet_symbol[worker], 128, "%s", name);

	/* TODO */
	last_codelet_hash[worker] = 0;

	float start_codelet_time = get_event_time_stamp();
	last_codelet_start[worker] = start_codelet_time;

	if (per_task_colour)
	{
		create_paje_state_if_not_found(name);

		fprintf(out_paje_file, "101       %f	S      %s%ld      E	%s\n", start_codelet_time, prefix, ev.param[1], name);
	}
	else {
		fprintf(out_paje_file, "10       %f	S      %s%ld      E\n", start_codelet_time, prefix, ev.param[1]);
	}

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_end_codelet_body(void)
{
	int worker;
	worker = find_worker_id(ev.param[1]);
	if (worker < 0) return;

	float end_codelet_time = get_event_time_stamp();

	fprintf(out_paje_file, "10       %f	S      %s%ld      B\n", end_codelet_time, prefix, ev.param[1]);

	float codelet_length = (end_codelet_time - last_codelet_start[worker]);
	
	if (generate_distrib)
	fprintf(distrib_time, "%s\t%s%d\t%lx\t%f\n", last_codelet_symbol[worker],
				prefix, worker, last_codelet_hash[worker], codelet_length);

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_user_event(void)
{
	int worker;
	worker = find_worker_id(ev.param[1]);
	if (worker < 0) return;

	unsigned code;
	code = ev.param[2];	

	fprintf(out_paje_file, "9       %f     event      %s%ld      %d\n", get_event_time_stamp(), prefix, ev.param[1], code);
}

static void handle_start_callback(void)
{
	int worker;
	worker = find_worker_id(ev.param[1]);
	if (worker < 0) return;
	fprintf(out_paje_file, "10       %f	S      %s%ld      C\n", get_event_time_stamp(), prefix, ev.param[1] );
}

static void handle_end_callback(void)
{
	int worker;
	worker = find_worker_id(ev.param[1]);
	if (worker < 0) return;
	fprintf(out_paje_file, "10       %f	S      %s%ld      B\n", get_event_time_stamp(), prefix, ev.param[1] );
}

static void handle_worker_status(const char *newstatus)
{
	int worker;
	worker = find_worker_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %s%ld      %s\n",
				get_event_time_stamp(), prefix, ev.param[1], newstatus);

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_data_copy(void)
{
}

static void handle_start_driver_copy(void)
{
	unsigned src = ev.param[0];
	unsigned dst = ev.param[1];
	unsigned size = ev.param[2];
	unsigned comid = ev.param[3];

	if (!no_bus)
	{
		fprintf(out_paje_file, "10       %f     MS      %sMEMNODE%d      Co\n", get_event_time_stamp(), prefix, dst);
		fprintf(out_paje_file, "18       %f	L      %sp	%d	%sMEMNODE%d	com_%d\n", get_event_time_stamp(), prefix, size, prefix, src, comid);

		/* create a structure to store the start of the communication, this will be matched later */
		communication_t com = communication_new();
		com->comid = comid;
		com->comm_start = get_event_time_stamp();

		/* that's a hack: either src or dst is non null */
		com->node = (src + dst);

		communication_list_push_back(communication_list, com);
	}

}

static void handle_end_driver_copy(void)
{
	unsigned dst = ev.param[1];
	unsigned size = ev.param[2];
	unsigned comid = ev.param[3];

	if (!no_bus)
	{
		fprintf(out_paje_file, "10       %f     MS      %sMEMNODE%d      No\n", get_event_time_stamp(), prefix, dst);
		fprintf(out_paje_file, "19       %f	L      %sp	%d	%sMEMNODE%d	com_%d\n", get_event_time_stamp(), prefix, size, prefix, dst, comid);

		/* look for a data transfer to match */
		communication_itor_t itor;
		for (itor = communication_list_begin(communication_list);
			itor != communication_list_end(communication_list);
			itor = communication_list_next(itor))
		{
			if (itor->comid == comid)
			{
				float comm_end = get_event_time_stamp();
				float bandwith = (float)((0.001*size)/(comm_end - itor->comm_start));

				itor->bandwith = bandwith;

				communication_t com = communication_new();
				com->comid = comid;
				com->comm_start = get_event_time_stamp();
				com->bandwith = -bandwith;

				com->node = itor->node;

				communication_list_push_back(communication_list, com);

				break;
			}
		}
	}
}

static void display_bandwith_evolution(void)
{
	float current_bandwith = 0.0;
	float current_bandwith_per_node[32] = {0.0};

	communication_itor_t itor;
	for (itor = communication_list_begin(communication_list);
		itor != communication_list_end(communication_list);
		itor = communication_list_next(itor))
	{
		current_bandwith += itor->bandwith;
		fprintf(out_paje_file, "13  %f bw %sMEMNODE0 %f\n",
				itor->comm_start, prefix, current_bandwith);

		current_bandwith_per_node[itor->node] +=  itor->bandwith;
		fprintf(out_paje_file, "13  %f bw %sMEMNODE%d %f\n",
				itor->comm_start, prefix, itor->node, current_bandwith_per_node[itor->node]);
	}
}

static void handle_memnode_event(const char *eventstr)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      %sMEMNODE%d      %s\n",
		get_event_time_stamp(), prefix, memnode, eventstr);
}

/*
 *	Number of task submitted to the scheduler
 */
static int curq_size = 0;

static void handle_job_push(void)
{
	curq_size++;
	fprintf(out_paje_file, "13       %f ntask %ssched %f\n", get_event_time_stamp(), prefix, (float)curq_size);
}

static void handle_job_pop(void)
{
	curq_size--;
	fprintf(out_paje_file, "13       %f ntask %ssched %f\n", get_event_time_stamp(), prefix, (float)curq_size);
}

static void handle_codelet_tag_deps(void)
{
	uint64_t child;
	uint64_t father;

	child = ev.param[0]; 
	father = ev.param[1]; 

	add_deps(child, father);
}

static void handle_task_done(void)
{
	uint64_t tag_id;
	tag_id = ev.param[0];

	unsigned long has_name = ev.param[2];
	char *name = has_name?(char *)&ev.param[3]:"unknown";

        int worker;
        worker = find_worker_id(ev.param[1]);

	char *colour;
	char buffer[32];
	if (per_task_colour) {
		snprintf(buffer, 32, "%.4f,%.4f,%.4f",
			get_colour_symbol_red(name)/1024.0,
			get_colour_symbol_green(name)/1024.0,
			get_colour_symbol_blue(name)/1024.0);
		colour = &buffer[0];
	}
	else {
		colour= (worker < 0)?"0.0,0.0,0.0":get_worker_color(worker);
	}

	dot_set_tag_done(tag_id, colour);
}

static void handle_mpi_barrier(void)
{
	rank = ev.param[0];

	/* Add an event in the trace */
	fprintf(out_paje_file, "9       %f     event      %sp      %d\n", get_event_time_stamp(), prefix, rank);
}

static void handle_mpi_isend(void)
{
	int dest = ev.param[0];
	int mpi_tag = ev.param[1];
	size_t size = ev.param[2];
	float date = get_event_time_stamp();

	add_mpi_send_transfer(rank, dest, mpi_tag, size, date);
}

static void handle_mpi_irecv_end(void)
{
	int src = ev.param[0];
	int mpi_tag = ev.param[1];
	float date = get_event_time_stamp();

	add_mpi_recv_transfer(src, rank, mpi_tag, date);
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-c") == 0) {
			per_task_colour = 1;
		}

		if (strcmp(argv[i], "-o") == 0) {
			out_paje_path = argv[++i];
		}

		if (strcmp(argv[i], "-i") == 0) {
			filenames[ninputfiles++] = argv[++i];
		}

		if (strcmp(argv[i], "-no-counter") == 0) {
			no_counter = 1;
		}

		if (strcmp(argv[i], "-no-bus") == 0) {
			no_bus = 1;
		}

		if (strcmp(argv[i], "-d") == 0) {
			generate_distrib = 1;
		}

		if (strcmp(argv[i], "-h") == 0) {
		        fprintf(stderr, "Usage : %s [-c] [-no-counter] [-no-bus] [-i input_filename] [-o output_filename]\n", argv[0]);
			fprintf(stderr, "\t-c: use a different colour for every type of task.\n");
		        exit(-1);
		}
	}
}

void parse_new_file(char *filename_in, char *file_prefix, uint64_t file_offset)
{
	prefix = file_prefix;
	offset = file_offset;

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

	/* create a htable to identify each worker(tid) */
	hcreate(MAXWORKERS);

	symbol_list = symbol_name_list_new(); 
	communication_list = communication_list_new();

	/* TODO starttime ...*/
	/* create the "program" container */
	fprintf(out_paje_file, "7      0.0 %sp      P      MPIroot       program%s \n", prefix, prefix);
	/* create a variable with the number of tasks */
	if (!no_counter)
	{
		fprintf(out_paje_file, "7     %f    %ssched   Sc    %sp     scheduler \n", 0.0, prefix, prefix);
		fprintf(out_paje_file, "13    0.0    ntask %ssched 0.0\n", prefix);
	}

	unsigned first_event = 1;

	while(1) {
		int ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev);
		if (ret != FXT_EV_OK) {
			fprintf(stderr, "no more block ...\n");
			break;
		}

		__attribute__ ((unused)) int nbparam = ev.nb_params;

		if (first_event)
		{
			first_event = 0;
			start_time = ev.time;

		}

		switch (ev.code) {
			case FUT_WORKER_INIT_START:
				handle_worker_init_start();
				break;

			case FUT_WORKER_INIT_END:
				handle_worker_init_end();
				break;

			case FUT_NEW_MEM_NODE:
				handle_new_mem_node();
				break;

			/* detect when the workers were idling or not */
			case FUT_START_CODELET_BODY:
				handle_start_codelet_body();
				break;
			case FUT_END_CODELET_BODY:
				handle_end_codelet_body();
				break;

			case FUT_START_CALLBACK:
				handle_start_callback();
				break;
			case FUT_END_CALLBACK:
				handle_end_callback();
				break;

			/* monitor stack size */
			case FUT_JOB_PUSH:
				if (!no_counter)
				handle_job_push();
				break;
			case FUT_JOB_POP:
				if (!no_counter)
				handle_job_pop();
				break;

			/* check the memory transfer overhead */
			case FUT_START_FETCH_INPUT:
				handle_worker_status("Fi");
				break;

			case FUT_START_PUSH_OUTPUT:
				handle_worker_status("Po");
				break;

			case FUT_START_PROGRESS:
				handle_worker_status("P");
				break;

			case FUT_END_FETCH_INPUT:
			case FUT_END_PROGRESS:
			case FUT_END_PUSH_OUTPUT:
				handle_worker_status("B");
				break;

			case FUT_CODELET_TAG:
				/* XXX */
				break;

			case FUT_CODELET_TAG_DEPS:
				handle_codelet_tag_deps();
				break;

			case FUT_TASK_DONE:
				handle_task_done();
				break;

			case FUT_DATA_COPY:
				if (!no_bus)
				handle_data_copy();
				break;

			case FUT_START_DRIVER_COPY:
				if (!no_bus)
				handle_start_driver_copy();
				break;

			case FUT_END_DRIVER_COPY:
				if (!no_bus)
				handle_end_driver_copy();
				break;

			case FUT_WORK_STEALING:
				/* XXX */
				break;

			case FUT_WORKER_DEINIT_START:
				handle_worker_deinit_start();
				break;

			case FUT_WORKER_DEINIT_END:
				handle_worker_deinit_end();
				break;

			case FUT_START_ALLOC:
				if (!no_bus)
				handle_memnode_event("A");
				break;

			case FUT_START_ALLOC_REUSE:
				if (!no_bus)
				handle_memnode_event("Ar");
				break;

			case FUT_START_MEMRECLAIM:
				handle_memnode_event("R");
				break;

			case FUT_END_ALLOC:
			case FUT_END_ALLOC_REUSE:
			case FUT_END_MEMRECLAIM:
				if (!no_bus)
				handle_memnode_event("No");
				break;

			case FUT_USER_EVENT:
				handle_user_event();
				break;

			case FUT_MPI_BARRIER:
				handle_mpi_barrier();
				break;

			case FUT_MPI_ISEND:
				handle_mpi_isend();
				break;

			case FUT_MPI_IRECV_END:
				handle_mpi_irecv_end();
				break;

			default:
				fprintf(stderr, "unknown event.. %x at time %llx WITH OFFSET %llx\n",
					(unsigned)ev.code, (long long unsigned)ev.time, (long long unsigned)(ev.time-offset));
				break;
		}
	}

	hdestroy();

	/* Close the trace file */
	if (close(fd_in))
	{
	        perror("close failed :");
	        exit(-1);
	}
}

/*
 * This program should be used to parse the log generated by FxT 
 */
int main(int argc, char **argv)
{
	int fd_out;

	parse_args(argc, argv);

	init_dag_dot();

	if (generate_distrib)
		distrib_time = fopen(distrib_time_path, "w+");

	paje_output_file_init();

	if (ninputfiles == 1)
	{
		/* we usually only have a single trace */
		uint64_t file_start_time = find_start_time(filenames[0]);
		parse_new_file(filenames[0], "", file_start_time);
	}
	else {
		unsigned inputfile;

		uint64_t offsets[64];
		uint64_t found_offsets[64];
		uint64_t start_times[64];

		uint64_t max = 0;

		/*
		 * Find the trace offsets:
		 *	- If there is no sync point
		 *		psi_k(x) = x - start_k
		 *	- If there is a sync point sync_k
		 *		psi_k(x) = x - sync_k + M
		 *		where M = max { sync_i - start_i | there exists sync_i}
		 * More generally:
		 *	- psi_k(x) = x - offset_k
		 */
		
		int unique_keys[64];
		int rank_k[64];
		uint64_t start_k[64];
		uint64_t sync_k[64];
		unsigned sync_k_exists[64];
		uint64_t M = 0;

		unsigned found_one_sync_point = 0;
		int key;
		unsigned display_mpi = 0; 

		/* Compute all start_k */
		for (inputfile = 0; inputfile < ninputfiles; inputfile++)
		{
			uint64_t file_start = find_start_time(filenames[inputfile]);
			start_k[inputfile] = file_start; 
		}

		/* Compute all sync_k if they exist */
		for (inputfile = 0; inputfile < ninputfiles; inputfile++)
		{
			int ret = find_sync_point(filenames[inputfile],
							&sync_k[inputfile],
							&unique_keys[inputfile],
							&rank_k[inputfile]);
			if (ret == -1)
			{
				/* There was no sync point, we assume there is no offset */
				sync_k_exists[inputfile] = 0;
			}
			else {
				if (!found_one_sync_point)
				{
					key = unique_keys[inputfile];
					display_mpi = 1;
					found_one_sync_point = 1;
				}
				else {
					if (key != unique_keys[inputfile])
					{
						fprintf(stderr, "Warning: traces are coming from different run so we will not try to display MPI communications.\n");
						display_mpi = 0;
					}
				}


				STARPU_ASSERT(sync_k[inputfile] >= start_k[inputfile]);

				sync_k_exists[inputfile] = 1;

				uint64_t diff = sync_k[inputfile] - start_k[inputfile];
				if (diff > M)
					M = diff;
			}
		}

		/* Compute the offset */
		for (inputfile = 0; inputfile < ninputfiles; inputfile++)
		{
			offsets[inputfile] = (sync_k_exists[inputfile]?start_k[inputfile]:(M-sync_k[inputfile]));
		}

		/* generate the Paje trace for the different files */
		for (inputfile = 0; inputfile < ninputfiles; inputfile++)
		{
			int filerank = rank_k[inputfile];

			char file_prefix[32];
			snprintf(file_prefix, 32, "mpi_%d_", filerank);
			parse_new_file(filenames[inputfile], file_prefix, offsets[inputfile]);
		}

		/* display the MPI transfers if possible */
		for (inputfile = 0; inputfile < ninputfiles; inputfile++)
		{
			int filerank = rank_k[inputfile];
			display_all_transfers_from_trace(out_paje_file, filerank);
		}
	}

	display_bandwith_evolution();

	/* close the different files */
	fclose(out_paje_file);

	if (generate_distrib)
		fclose(distrib_time);

	terminate_dat_dot();

	return 0;
}
