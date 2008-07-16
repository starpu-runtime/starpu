#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <common/timing.h>
#include <common/util.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>

pthread_t tid_gpu;

void callback_core(void *argcb)
{
	printf("callback core\n");
}

void core_codelet(void *_args)
{
	printf("codelet core\n");
	sleep(1);

	/* send a signal to the GPU thread */
	pthread_kill(tid_gpu, SIGUSR1); 


}



void callback_gpu(void *argcb)
{
	printf("callback gpu\n");
}

void sigusr1_handler(int sig)
{
	printf("SIGUSR1 received\n");
}

void gpu_codelet(void *_args)
{
	printf("codelet gpu\n");
	tid_gpu = pthread_self();

	signal(SIGUSR2, sigusr1_handler);

	/* now perform a lot of work ... */
	void **d_ptr;
	unsigned sizex, sizey;

	float *h_ptr = malloc(sizex*sizey*sizeof(float));
	memset(h_ptr, 'a', sizex*sizey*sizeof(float));
	sizex = 1024*1024;
	sizey = 128;
	cublasAlloc(sizex*sizey, sizeof(float), d_ptr);
	cublasSetMatrix(sizex, sizey, sizeof(float), h_ptr, sizex, d_ptr, sizex);
}

int main(int argc, char **argv)
{
	init_machine();

	printf("ok\n");

	codelet cl;
	codelet cl2;

	job_t j = job_create();
	j->where = CUBLAS;
	j->cb = callback_gpu;
	j->cl = &cl;

	job_t j2 = job_create();
	j2->where = CORE;
	j2->cb = callback_core;
	j2->cl = &cl2;

	cl.cl_arg = NULL;
	cl.cublas_func = gpu_codelet;

	cl2.cl_arg = NULL;
	cl2.core_func = core_codelet;

	push_task(j2);
	push_task(j);

	sleep(100);
}
