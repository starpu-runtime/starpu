#ifdef USE_SPU

#include "mult_spu.h"
#include "../spu/mult_spu_common.h"

/* this is the program that will run on the SPU */
extern spe_program_handle_t spu_worker_program;

spe_context_ptr_t speid;
spe_stop_info_t stopinfo;
int status;

/* returns the number of available SPUs */
unsigned get_spu_count(void)
{
	unsigned cnt;

	cnt = spe_cpu_info_get(SPE_COUNT_USABLE_SPES,-1);
	printf("DEBUG cnt %d \n", cnt);

	return cnt;
}

///* TODO move into the SPU code */
//static int execute_job_on_spu(job_t j)
//{
//	job_descr *jd;
//	jd = j->argcb;
//#warning TODO 
////	switch (j->type){
////		case CODELET:
////			assert(j->cl);
////			assert(j->cl->cublas_func);
////			j->cl->cublas_func(j->cl->cl_arg);
////			break;
////		case ABORT:
////			//printf("SPU abort\n");
////			//cublasShutdown();
////			pthread_exit(NULL);
////			break;
////		default:
////			break;
////	}
//
//	return OK;
//}
//
void *spu_worker(void *arg)
{
	struct spu_worker_arg_t* args = (struct spu_worker_arg_t*)arg;

	int devid = args->deviceid;

#ifndef DONTBIND
        /* fix the thread on the correct cpu */
        cpu_set_t aff_mask;
        CPU_ZERO(&aff_mask);
        CPU_SET(args->bindid, &aff_mask);
        sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

	/* each SPU has its local memory node defined by its local store */
	set_local_memory_node_key(&(((spu_worker_arg *)arg)->memory_node));

	/* here we assume that this thread will only control a single SPU 
 	* so there is no need for an extra thread since that one will handle
 	* the SPU directly */

	speid = spe_context_create(SPE_EVENTS_ENABLE, NULL);
	if (!speid) {
		fprintf(stderr, "Context #%d was not initialized\n", devid);
		ASSERT(0);
	}

	args->speid = speid;

	status = spe_program_load(speid, &spu_worker_program);
	if (status) {
		fprintf(stderr, "Could not load program on SPU %d\n", devid);
		ASSERT(0);
	}

	/* the input argument */
	spu_init_arguments worker_arg __attribute__ ((aligned(16)));
	worker_arg.ea_ready_flag = &args->ready_flag;

	/* we start at main */
	unsigned int entry = SPE_DEFAULT_ENTRY; 
	spe_context_run(speid, &entry, 0, &worker_arg, NULL, &stopinfo); 

	printf("SPU %d terminated\n", devid);

	return NULL;
}


#endif // USE_SPU
