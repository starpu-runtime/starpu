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

event_list_t events[MAXWORKERS];
workq_list_t taskq;
char *worker_name[MAXWORKERS];


static char *cuda_worker_colors[MAXWORKERS] = {"/greens9/7", "/greens9/6", "/greens9/5", "/greens9/4"};
static char *cpus_worker_colors[MAXWORKERS] = {"/ylorrd9/9", "/ylorrd9/6", "/ylorrd9/3", "/ylorrd9/1"};
static char *other_worker_colors[MAXWORKERS] = {"/greys9/9", "/greys9/8", "/greys9/7", "/greys9/6"};
static char *worker_colors[MAXWORKERS];

static fxt_t fut;
struct fxt_ev_64 ev;

unsigned first_event = 1;
uint64_t start_time = 0;
uint64_t end_time = 0;

unsigned nworkers = 0;


/*
 * Paje trace file tools
 */

static char *out_paje_path = "paje.trace";
static FILE *out_paje_file;

void paje_output_file_init(void)
{
	/* create a new file */
	out_paje_file = fopen(out_paje_path, "w+");
	
	write_paje_header(out_paje_file);

	fprintf(out_paje_file, "                                        \n \
	1       P      0       \"Program\"                      	\n \
	1       Mn      P       \"Memory Node\"                         \n \
	1       T      Mn       \"Worker\"                               \n \
	3       S       T       \"Thread State\"                        \n \
	3       MS       Mn       \"Memory Node State\"                        \n \
	6       Fi       S      FetchingInput       \"1.0 .1 1.0\"            \n \
	6       Po       S      PushingOutput       \"0.1 1.0 1.0\"            \n \
	6       E       S       Executing       \".0 .6 .4\"            \n \
	6       C       S       Callback       \".0 .3 .8\"            \n \
	6       B       S       Blocked         \".9 .1 .0\"		\n \
	6       A       MS      Allocating         \".4 .1 .0\"		\n \
	6       Ar       MS      AllocatingReuse       \".1 .1 .8\"		\n \
	6       R       MS      Reclaiming         \".0 .1 .4\"		\n \
	6       Co       MS     DriverCopy         \".3 .5 .1\"		\n \
	6       No       MS     Nothing         \".0 .0 .0\"		\n \
	5       L       P	Mn	Mn      L\n");
}

void paje_output_file_terminate(void)
{

	/* close the file */
	fclose(out_paje_file);
}


/*
 * Generic tools
 */


void handle_new_mem_node(void)
{
	char *memnodestr = malloc(16*sizeof(char));
	sprintf(memnodestr, "%ld", ev.param[0]);
	
	fprintf(out_paje_file, "7       %f	%s      Mn      p	MEMNODE%s\n", (float)((ev.time-start_time)/1000000.0), memnodestr, memnodestr);
}

static unsigned cuda_index = 0;
static unsigned cpus_index = 0;
static unsigned other_index = 0;

void handle_new_worker(void)
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

	events[workerid] = event_list_new();
}

void handle_worker_terminated(void)
{
	char *tidstr = malloc(16*sizeof(char));
	sprintf(tidstr, "%ld", ev.param[1]);

	fprintf(out_paje_file, "8       %f	%s\n", (float)((ev.time-start_time)/1000000.0), tidstr);
}


int find_workder_id(unsigned long tid)
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

void handle_start_codelet_body(void)
{

	//fprintf(stderr, "start codelet %p on tid %d\n", (void *)ev.param[0], ev.param[1]);

	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
//	printf("-> worker %d\n", worker);
	fprintf(out_paje_file, "10       %f	S      %ld      E\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	event_t e = event_new();
	e->time =  ev.time;
	e->mode = WORKING;
	event_list_push_back(events[worker], e);

	end_time = STARPU_MAX(end_time, ev.time);
}

void handle_end_codelet_body(void)
{
	//fprintf(stderr, "end codelet %p on tid %d\n", (void *)ev.param[0], ev.param[1]);

	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
//	printf("<- worker %d\n", worker);
	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	event_t e = event_new();
	e->time =  ev.time;
	e->mode = IDLE;
	event_list_push_back(events[worker], e);

	end_time = STARPU_MAX(end_time, ev.time);
}



void handle_start_callback(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
	fprintf(out_paje_file, "10       %f	S      %ld      C\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );
}

void handle_end_callback(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );
}


void handle_start_fetch_input(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %ld      Fi\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	event_t e = event_new();
	e->time =  ev.time;
	e->mode = FETCHING;
	event_list_push_back(events[worker], e);

	end_time = STARPU_MAX(end_time, ev.time);
}

void handle_end_fetch_input(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	event_t e = event_new();
	e->time =  ev.time;
	e->mode = IDLE;
	event_list_push_back(events[worker], e);

	end_time = STARPU_MAX(end_time, ev.time);
}

void handle_start_push_output(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;

	fprintf(out_paje_file, "10       %f	S      %ld      Po\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	event_t e = event_new();
	e->time =  ev.time;
	e->mode = PUSHING;
	event_list_push_back(events[worker], e);

	end_time = STARPU_MAX(end_time, ev.time);
}

void handle_end_push_output(void)
{
	int worker;
	worker = find_workder_id(ev.param[1]);
	if (worker < 0) return;
	
	fprintf(out_paje_file, "10       %f	S      %ld      B\n", (float)((ev.time-start_time)/1000000.0), ev.param[1] );

	event_t e = event_new();
	e->time =  ev.time;
	e->mode = IDLE;
	event_list_push_back(events[worker], e);

	end_time = STARPU_MAX(end_time, ev.time);
}

void handle_data_copy(void)
{
}

void handle_start_driver_copy(void)
{
	unsigned src = ev.param[0];
	unsigned dst = ev.param[1];
	unsigned size = ev.param[2];
	unsigned comid = ev.param[3];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      Co\n", (float)((ev.time-start_time)/1000000.0), dst);
	fprintf(out_paje_file, "18       %f	L      p	%d	MEMNODE%d	com_%d\n", (float)((ev.time-start_time)/1000000.0), size, src, comid);

}

void handle_end_driver_copy(void)
{
	unsigned dst = ev.param[1];
	unsigned size = ev.param[2];
	unsigned comid = ev.param[3];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), dst);
	fprintf(out_paje_file, "19       %f	L      p	%d	MEMNODE%d	com_%d\n", (float)((ev.time-start_time)/1000000.0), size, dst, comid);
}

void handle_start_alloc(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      A\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

void handle_end_alloc(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), memnode);
}


void handle_start_alloc_reuse(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      Ar\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

void handle_end_alloc_reuse(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), memnode);
}


void handle_start_memreclaim(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      R\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

void handle_end_memreclaim(void)
{
	unsigned memnode = ev.param[0];

	fprintf(out_paje_file, "10       %f     MS      MEMNODE%d      No\n", (float)((ev.time-start_time)/1000000.0), memnode);
}

int maxq_size = 0;
int curq_size = 0;

void handle_job_push(void)
{
	curq_size++;

	maxq_size = STARPU_MAX(maxq_size, curq_size);

	workq_t e = workq_new();
	e->time =  ev.time;
	e->diff =  +1;
	e->current_size = curq_size;

	workq_list_push_back(taskq, e);
}

void handle_job_pop(void)
{
	curq_size--;

	workq_t e = workq_new();
	e->time =  ev.time;
	e->diff =  -1;
	e->current_size = curq_size;

	workq_list_push_back(taskq, e);
}

void handle_codelet_tag_deps(void)
{
	uint64_t child;
	uint64_t father;

	child = ev.param[0]; 
	father = ev.param[1]; 

	add_deps(child, father);
}

void handle_task_done(void)
{
	uint64_t tag_id;
	tag_id = ev.param[0];

        int worker;
        worker = find_workder_id(ev.param[1]);

	dot_set_tag_done(tag_id, worker_colors[worker]);
}

#ifdef FLASH_RENDER
void generate_flash_output(void)
{
	flash_engine_init();
	flash_engine_generate_output(events, taskq, worker_name, nworkers, maxq_size, start_time, end_time, "toto.swf");
}
#endif

void generate_svg_output(void)
{
	svg_engine_generate_output(events, taskq, worker_name, nworkers, maxq_size, start_time, end_time, "toto.svg");
}

void generate_gnuplot_output(void)
{
	FILE *output;
	output = fopen("data", "w+");
	STARPU_ASSERT(output);
	
	unsigned linesize;
	unsigned maxline = 0;

	unsigned worker;
	for (worker = 0; worker < nworkers; worker++)
	{
		linesize = 0;

		event_itor_t i;
		for (i = event_list_begin(events[worker]);
		     i != event_list_end(events[worker]);
		     i = event_list_next(i))
		{
			linesize++;
		}
		maxline = STARPU_MAX(maxline, linesize);
	}

	unsigned i;
	for (i = 0; i < maxline + 1; i++)
	{
		fprintf(output, "bla\t");
	}
	fprintf(output,"\n");


	for (worker = 0; worker < nworkers; worker++)
	{
		unsigned long prev = start_time;

		fprintf(output, "%d\t", 0);

		event_itor_t i;
		for (i = event_list_begin(events[worker]);
		     i != event_list_end(events[worker]);
		     i = event_list_next(i))
		{
			fprintf(output, "%lu\t", (i->time - prev)/FACTOR);
			prev = i->time;
		}
		fprintf(output, "\n");
	}

	fclose(output);
}

#ifdef USE_GTK
void gtk_viewer(int argc, char **argv)
{
	gtk_viewer_apps(argc, argv, events, taskq, worker_name, nworkers, maxq_size, start_time, end_time);
}
#endif

/*
 * This program should be used to parse the log generated by FxT 
 */
int main(int argc, char **argv)
{
	char *filename, *filenameout = NULL;
	int ret;
	int fd_in, fd_out;

	int use_stdout = 1;

	init_dag_dot();
	
	if (argc < 2) {
	        fprintf(stderr, "Usage : %s input_filename [-o output_filename]\n", argv[0]);
	        exit(-1);
	}
	
	filename = argv[1];
	

	fd_in = open(filename, O_RDONLY);
	if (fd_in < 0) {
	        perror("open failed :");
	        exit(-1);
	}

	if (argc > 2) {
		filenameout = argv[2];
		use_stdout = 0;
		fd_out = open(filenameout, O_RDWR);
		if (fd_out < 0) {
			perror("open (out) failed :");
			exit(-1);
		}
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

	paje_output_file_init();

	taskq = workq_list_new();

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

			fprintf(out_paje_file, "7       %f p      P      0       program \n", (float)(start_time-start_time));
		}

		switch (ev.code) {
			case FUT_NEW_WORKER_KEY:
				handle_new_worker();
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
				handle_job_push();
				break;
			case FUT_JOB_POP:
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
				handle_data_copy();
				break;

			case FUT_START_DRIVER_COPY:
				handle_start_driver_copy();
				break;

			case FUT_END_DRIVER_COPY:
				handle_end_driver_copy();
				break;

			case FUT_WORK_STEALING:
				/* XXX */
				break;

			case FUT_WORKER_TERMINATED:
				handle_worker_terminated();
				break;

			case FUT_START_ALLOC:
				handle_start_alloc();
				break;

			case FUT_END_ALLOC:
				handle_end_alloc();
				break;

			case FUT_START_ALLOC_REUSE:
				handle_start_alloc_reuse();
				break;

			case FUT_END_ALLOC_REUSE:
				handle_end_alloc_reuse();
				break;


			case FUT_START_MEMRECLAIM:
				handle_start_memreclaim();
				break;

			case FUT_END_MEMRECLAIM:
				handle_end_memreclaim();
				break;

			default:
				fprintf(stderr, "unknown event.. %x at time %llx\n", (unsigned)ev.code, (long long unsigned)ev.time);
				break;
		}
	}

	generate_gnuplot_output();
	//generate_flash_output();
	generate_svg_output();
	paje_output_file_terminate();

	terminate_dat_dot();

#ifdef USE_GTK
	gtk_viewer(argc, argv);
#endif

	return 0;
}
