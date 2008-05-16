#ifdef USE_CUBLAS

#include "mult_cublas.h"

#if 0
static cublasStatus status;
#endif

extern int cublascounters[MAXCUBLASDEVS];

unsigned get_cublas_device_count(void)
{
	/* XXX */
	return 1;
}

static int execute_job_on_cublas(job_t j)
{
	job_descr *jd;
	jd = j->argcb;

	switch (j->type){
		case CODELET:
			assert(j->cl);
			assert(j->cl->cublas_func);
			fetch_codelet_input(j->buffers, j->nbuffers);

			TRACE_START_CODELET_BODY(j);
			j->cl->cublas_func(j->buffers, j->cl->cl_arg);
			cuCtxSynchronize();
			TRACE_END_CODELET_BODY(j);

			push_codelet_output(j->buffers, j->nbuffers, 1<<0);
			break;
		case ABORT:
			fprintf(stderr, "CUBLAS abort\n");
			cublasShutdown();
			pthread_exit(NULL);
			break;
		default:
			break;
	}

	return OK;
}

void stress_memory(void)
{
	char *strval;
	strval = getenv("STRESS_MEM");
	if (strval) {
		/* just for performance measurement purposes ! */
		unsigned size;
		char *check;
		void *dummy;

		size = strtol(strval, &check, 10);
		ASSERT(strcmp(check, "\0") == 0);

		fprintf(stderr, "Warning : Stress CUBLAS memory, pre-allocate %d MB\n", size);
		cublasAlloc(size*1024*1024, 1, &dummy);
	}
}

void *cublas_worker(void *arg)
{
	struct cublas_worker_arg_t* args = (struct cublas_worker_arg_t*)arg;

	int devid = args->deviceid;

#ifdef USE_FXT
	fxt_register_thread(args->bindid);
#endif

	TRACE_NEW_WORKER(FUT_CUBLAS_KEY);

#ifndef DONTBIND
        /* fix the thread on the correct cpu */
        cpu_set_t aff_mask;
        CPU_ZERO(&aff_mask);
        CPU_SET(args->bindid, &aff_mask);
        sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

	set_local_memory_node_key(&(((cublas_worker_arg *)arg)->memory_node));

	cublasInit();

	/* just to test the impact of memory stress ... */
	stress_memory();

	fprintf(stderr, "cublas thread is ready to run on CPU %d !\n", args->bindid);
	/* tell the main thread that this one is ready to work */
	args->ready_flag = 1;

	int res;
	job_t j;
	do {
		j = pop_task();
		if (j == NULL) continue;

		/* can cublas do that task ? */
		if (!CUBLAS_MAY_PERFORM(j))
		{
			push_task(j);
			continue;
		}

		res = execute_job_on_cublas(j);

		if (res != OK) {
			switch (res) {
				case OK:
					assert(0);
				case FATAL:
					assert(0);
				case TRYAGAIN:
					push_task(j);
					continue;
				default:
					assert(0);
			}
		}

		if (j->cb)
			j->cb(j->argcb);
		
		cublascounters[devid]++;		

		job_delete(j);
	} while(1);

	return NULL;
}


#endif // USE_CUBLAS
