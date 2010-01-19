/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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
	1       P      0       \"Program\"                      	\n \
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
	5       L       P	Mn	Mn      L\n");
}

/*
 * Generic tools
 */

static float get_event_time_stamp(void)
{
	return (float)((ev.time-start_time)/1000000.0);
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

void parse_new_file(char *filename_in, char *file_prefix)
{
	prefix = file_prefix;

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

			/* create the "program" container */
			fprintf(out_paje_file, "7      0.0 %sp      P      0       program%s \n", prefix, prefix);
			/* create a variable with the number of tasks */
			if (!no_counter)
			{
				fprintf(out_paje_file, "7     0.0    %ssched   Sc    %sp     scheduler \n", prefix, prefix);
				fprintf(out_paje_file, "13    0.0    ntask %ssched 0.0\n", prefix);
			}

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

			default:
				fprintf(stderr, "unknown event.. %x at time %llx\n",
					(unsigned)ev.code, (long long unsigned)ev.time);
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

	int use_stdout = 1;

	parse_args(argc, argv);

	init_dag_dot();

	if (generate_distrib)
		distrib_time = fopen(distrib_time_path, "w+");

	paje_output_file_init();

	if (ninputfiles == 1)
	{
		/* we usually only have a single trace */
		parse_new_file(filenames[0], "");
	}
	else {
		unsigned inputfile;
		for (inputfile = 0; inputfile < ninputfiles; inputfile++)
		{
			char file_prefix[32];
			snprintf(file_prefix, 32, "FILE%d", inputfile);
			parse_new_file(filenames[inputfile], file_prefix);
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
