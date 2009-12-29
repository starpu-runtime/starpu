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

static char *worker_name[MAXWORKERS];

static char *cpus_worker_colors[MAXWORKERS] = {"/greens9/7", "/greens9/6", "/greens9/5", "/greens9/4",  "/greens9/9", "/greens9/3",  "/greens9/2",  "/greens9/1"  };
static char *cuda_worker_colors[MAXWORKERS] = {"/ylorrd9/9", "/ylorrd9/6", "/ylorrd9/3", "/ylorrd9/1", "/ylorrd9/8", "/ylorrd9/7", "/ylorrd9/4", "/ylorrd9/2",  "/ylorrd9/1"};
static char *other_worker_colors[MAXWORKERS] = {"/greys9/9", "/greys9/8", "/greys9/7", "/greys9/6"};
static char *worker_colors[MAXWORKERS];

static fxt_t fut;
struct fxt_ev_64 ev;

static unsigned first_event = 1;
static uint64_t start_time = 0;
static uint64_t end_time = 0;

static unsigned nworkers = 0;

static char *filename = NULL;
static unsigned per_task_colour = 0;

static unsigned generate_distrib = 0;

static unsigned no_counter = 0;
static unsigned no_bus = 0;

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

	if (generate_distrib)
		distrib_time = fopen(distrib_time_path, "w+");
	
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

static void paje_output_file_terminate(void)
{
	/* close the file */
	fclose(out_paje_file);

	if (generate_distrib)
		fclose(distrib_time);
}

/*
 * Generic tools
 */

static void handle_new_mem_node(void)
{
	char *memnodestr = malloc(16*sizeof(char));
	sprintf(memnodestr, "%ld", ev.param[0]);
	
	fprintf(out_paje_file, "7       %f	%s      Mn      p	MEMNODE%s\n", (float)((ev.time-start_time)/1000000.0), memnodestr, memnodestr);

	if (!no_bus)
		fprintf(out_paje_file, "13       %f bw MEMNODE%d 0.0\n", (float)(start_time-start_time), ev.param[0]);
}

static unsigned cuda_index = 0;
static unsigned cpus_index = 0;
static unsigned other_index = 0;

static void handle_worker_init_start(void)
{
	/* 
	   arg0 : type of worker (cuda, core ..)
	   arg1 : memory node
	   arg2 : thread id 
	*/
	char *str = malloc(20*sizeof(char));
	char *color = NULL;
	
	strcpy(str, "unknown");

	switch (ev.param[0]) {
		case FUT_APPS_KEY:
			str = "apps";
			color = other_worker_colors[other_index++];
			break;
		case FUT_CORE_KEY:
			str = "core";
			color = cpus_worker_colors[cpus_index++];
			break;
		case FUT_CUDA_KEY:
			str = "cuda";
			color = cuda_worker_colors[cuda_index++];
			break;
	}

//	fprintf(stderr, "new %s worker (tid = %d)\n", str, ev.param[1]);
	char *memnodestr = malloc(16*sizeof(char));
	sprintf(memnodestr, "%ld", ev.param[1]);
	
	char *tidstr = malloc(16*sizeof(char));
	sprintf(tidstr, "%ld", ev.param[2]);

	fprintf(out_paje_file, "7       %f	%s      T      MEMNODE%s       %s \n", (float)((ev.time-start_time)/1000000.0), tidstr, memnodestr, tidstr);

	/* create a new key in the htable */
	uint64_t workerid = nworkers++;
	ENTRY item;
		item.key = tidstr;
		item.data = (void *)workerid;

	worker_colors[workerid] = color;

	ENTRY *res;
	res = hsearch(item, FIND);

	worker_name[workerid] = str;

	/* only register a thread once */
	STARPU_ASSERT(res == NULL);

	res = hsearch(item, ENTER);
	STARPU_ASSERT(res);

	/* start initialization */
	fprintf(out_paje_file, "10       %f     S      %ld      I\n", (float)((ev.time-start_time)/1000000.0), ev.param[2]);
}

static void handle_worker_init_end(void)
{
	fprintf(out_paje_file, "10       %f     S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[0]);
}

static void handle_worker_deinit_start(void)
{
	fprintf(out_paje_file, "10       %f     S      %ld      D\n", (float)((ev.time-start_time)/1000000.0), ev.param[0]);
}

static void handle_worker_deinit_end(void)
{
	fprintf(out_paje_file, "8       %f	%ld	T\n", (float)((ev.time-start_time)/1000000.0), ev.param[1]);
}

static int find_workder_id(unsigned long tid)
{
	char tidstr[16];
	sprintf(tidstr, "%ld", tid);

	ENTRY item;
		item.key = tidstr;
		item.data = NULL;
	ENTRY *res;
	res = hsearch(item, FIND);
	//STARPU_ASSERT(res);
	if (!res)
		return -1;

	int id = (uintptr_t)(res->data);

	return id;
}

static unsigned get_colour_symbol_red(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = crc32_string(name, 0);
	return (unsigned)crc32_string("red", hash_symbol) % 1024;
}

static unsigned get_colour_symbol_green(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = crc32_string(name, 0);
	return (unsigned)crc32_string("green", hash_symbol) % 1024;
}

static unsigned get_colour_symbol_blue(char *name)
{
	/* choose some colour ... that's disguting yes */
	uint32_t hash_symbol = crc32_string(name, 0);
	return (unsigned)crc32_string("blue", hash_symbol) % 1024;
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

/* TODO  remove 32 */
static double last_codelet_start[32];
static uint64_t last_codelet_hash[32];
static char last_codelet_symbol[128][32];

static void handle_start_codelet_body(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);

	if (worker < 0) return;

	unsigned long has_name = ev.param[2];
	char *name = has_name?(char *)&ev.param[3]:"unknown";

	snprintf(last_codelet_symbol[worker], 128, "%s", name);

	/* TODO */
	last_codelet_hash[worker] = 0;

	float start_codelet_time = (float)((ev.time-start_time)/1000000.0);
	last_codelet_start[worker] = start_codelet_time;

	if (per_task_colour)
	{
		create_paje_state_if_not_found(name);

		fprintf(out_paje_file, "101       %f	S      %ld      E	%s\n", start_codelet_time, ev.param[1], name);
	}
	else {
		fprintf(out_paje_file, "10       %f	S      %ld      E\n", start_codelet_time, ev.param[1]);
	}

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_end_codelet_body(void)
{
	//fprintf(stderr, "end codelet %p on tid %d\n", (void *)ev.param[0], ev.param[1]);

	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	float end_codelet_time = (float)((ev.time-start_time)/1000000.0);

//	printf("<- worker %d\n", worker);
	fprintf(out_paje_file, "10       %f	S      %ld      B\n", end_codelet_time, ev.param[1] );

	float codelet_length = (end_codelet_time - last_codelet_start[worker]);
	
	if (generate_distrib)
	fprintf(distrib_time, "%s\t%lx\t%d\t%f\n", last_codelet_symbol[worker],
				worker, last_codelet_hash[worker], codelet_length);

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_user_event(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	unsigned code;
	code = ev.param[2];	

	fprintf(out_paje_file, "9       %f     event      %ld      %d\n", (float)((ev.time-start_time)/1000000.0), ev.param[1], code);
}

static void handle_start_callback(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
	fprintf(out_paje_file, "10       %f	S      %ld      C\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );
}

static void handle_end_callback(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );
}


static void handle_start_fetch_input(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %ld      Fi\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_end_fetch_input(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_start_push_output(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %ld      Po\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_end_push_output(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
	
	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_start_progress(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %ld      P\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	end_time = STARPU_MAX(end_time, ev.time);
}

static void handle_end_progress(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
	
	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

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
		fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      Co\n", (float)((ev.time-start_time)/1000000.0), dst);
		fprintf(out_paje_file, "18       %f	L      p	%d	MEMNODE%d	com_%d\n", (float)((ev.time-start_time)/1000000.0), size, src, comid);

		/* create a structure to store the start of the communication, this will be matched later */
		communication_t com = communication_new();
		com->comid = comid;
		com->comm_start = (float)((ev.time-start_time)/1000000.0);

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
		fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), dst);
		fprintf(out_paje_file, "19       %f	L      p	%d	MEMNODE%d	com_%d\n", (float)((ev.time-start_time)/1000000.0), size, dst, comid);

		/* look for a data transfer to match */
		communication_itor_t itor;
		for (itor = communication_list_begin(communication_list);
			itor != communication_list_end(communication_list);
			itor = communication_list_next(itor))
		{
			if (itor->comid == comid)
			{
				float comm_end = (float)((ev.time-start_time)/1000000.0);
				float bandwith = (float)((0.001*size)/(comm_end - itor->comm_start));

				itor->bandwith = bandwith;

				communication_t com = communication_new();
				com->comid = comid;
				com->comm_start = (float)((ev.time-start_time)/1000000.0);
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
		fprintf(out_paje_file, "13  %f bw MEMNODE0 %f\n",
				itor->comm_start, current_bandwith);

		current_bandwith_per_node[itor->node] +=  itor->bandwith;
		fprintf(out_paje_file, "13  %f bw MEMNODE%d %f\n",
				itor->comm_start, itor->node, current_bandwith_per_node[itor->node]);
	}
}

static void handle_start_alloc(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      A\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

static void handle_end_alloc(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

static void handle_start_alloc_reuse(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      Ar\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

static void handle_end_alloc_reuse(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

static void handle_start_memreclaim(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      R\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

static void handle_end_memreclaim(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

static int curq_size = 0;

static void handle_job_push(void)
{
	curq_size++;

	fprintf(out_paje_file, "13       %f ntask sched %f\n", (float)((ev.time-start_time)/1000000.0), (float)curq_size);
}

static void handle_job_pop(void)
{
	curq_size--;

	fprintf(out_paje_file, "13       %f ntask sched %f\n", (float)((ev.time-start_time)/1000000.0), (float)curq_size);
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
        worker = find_workder_id(ev.param[1]);

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
		colour=(worker < 0)?"0.0,0.0,0.0":worker_colors[worker];
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
			filename = argv[++i];
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

/*
 * This program should be used to parse the log generated by FxT 
 */
int main(int argc, char **argv)
{
	int ret;
	int fd_in, fd_out;

	int use_stdout = 1;

	init_dag_dot();

	parse_args(argc, argv);

	fd_in = open(filename, O_RDONLY);
	if (fd_in < 0) {
	        perror("open failed :");
	        exit(-1);
	}

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

	paje_output_file_init();

	while(1) {
		ret = fxt_next_ev(block, FXT_EV_TYPE_64, (struct fxt_ev *)&ev);
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
			fprintf(out_paje_file, "7       %f p      P      0       program \n", (float)(start_time-start_time));
			/* create a variable with the number of tasks */
			if (!no_counter)
			{
				fprintf(out_paje_file, "7       %f sched      Sc      p       scheduler \n", (float)(start_time-start_time));
				fprintf(out_paje_file, "13       %f ntask sched 0.0\n", (float)(start_time-start_time));
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
				handle_start_fetch_input();
				break;
			case FUT_END_FETCH_INPUT:
				handle_end_fetch_input();
				break;
			case FUT_START_PUSH_OUTPUT:
				handle_start_push_output();
				break;
			case FUT_END_PUSH_OUTPUT:
				handle_end_push_output();
				break;

			case FUT_START_PROGRESS:
				handle_start_progress();
				break;
			case FUT_END_PROGRESS:
				handle_end_progress();
				break;


			case FUT_CODELET_TAG:
				//handle_codelet_tag();
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
				handle_start_alloc();
				break;

			case FUT_END_ALLOC:
				if (!no_bus)
				handle_end_alloc();
				break;

			case FUT_START_ALLOC_REUSE:
				if (!no_bus)
				handle_start_alloc_reuse();
				break;

			case FUT_END_ALLOC_REUSE:
				if (!no_bus)
				handle_end_alloc_reuse();
				break;

			case FUT_START_MEMRECLAIM:
				handle_start_memreclaim();
				break;

			case FUT_END_MEMRECLAIM:
				handle_end_memreclaim();
				break;

			case FUT_USER_EVENT:
				handle_user_event();
				break;

			default:
				fprintf(stderr, "unknown event.. %x at time %llx\n", (unsigned)ev.code, (long long unsigned)ev.time);
				break;
		}
	}

	display_bandwith_evolution();

	paje_output_file_terminate();

	terminate_dat_dot();

	return 0;
}
