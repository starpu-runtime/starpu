#include <starpu.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>

/* size of the buffer used for bandwith measurement */
#define SIZE	32*1024*1024*sizeof(char)

#define NITER	32

double cudadev_timing_htod[MAXCUDADEVS] = {0.0};
double cudadev_timing_dtoh[MAXCUDADEVS] = {0.0};

void measure_bandwith_between_host_and_dev(int dev)
{
	/* Initiliaze CUDA context on the device */
	cudaSetDevice(dev);

	/* hack to force the initialization */
	cudaFree(0);

	/* Allocate a buffer on the device */
	unsigned char *d_buffer;
	cudaMalloc((void **)&d_buffer, SIZE);
	assert(d_buffer);

	/* Allocate a buffer on the host */
	unsigned char *h_buffer;
	cudaHostAlloc((void **)&h_buffer, SIZE, 0); 
	assert(h_buffer);

	/* Fill them */
	memset(h_buffer, 0, SIZE);
	cudaMemset(d_buffer, 0, SIZE);

	unsigned iter;
	double timing;
	struct timeval start;
	struct timeval end;

	/* Measure upload bandwith */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(d_buffer, h_buffer, SIZE, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();
	}
	gettimeofday(&end, NULL);
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	cudadev_timing_htod[dev] = timing/NITER;

	/* Measure download bandwith */
	gettimeofday(&start, NULL);
	for (iter = 0; iter < NITER; iter++)
	{
		cudaMemcpy(h_buffer, d_buffer, SIZE, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
	}
	gettimeofday(&end, NULL);
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

	cudadev_timing_dtoh[dev] = timing/NITER;

	/* Free buffers */
	cudaFreeHost(h_buffer);
	cudaFree(d_buffer);

	cudaThreadExit();
}

#define MAXNODES	16

int main(int argc, char **argv)
{
        int ncuda;
        cudaGetDeviceCount(&ncuda);

	fprintf(stderr, "FOUD %d devices\n", ncuda);

	int i, j;
	for (i = 0; i < ncuda; i++)
	{
		/* measure bandwith between Host and Device i */
		measure_bandwith_between_host_and_dev(i);
	}

	fprintf(stderr, "\n\nLatency Matrix\n\n");

	fprintf(stderr, "{\n");
	for (j = 0; j < MAXNODES; j++)
	{
		fprintf(stderr, "\t{");
		for (i = 0; i < MAXNODES; i++)
		{
			double latency;

			if ((i > ncuda) || (j > ncuda))
			{
				/* convention */
				latency = -1.0;
			}
			else if (i == j)
			{
				latency = 0.0;
			}
			else {
				latency = ((i && j)?2000.0:500.0);
			}
	
			fprintf(stderr, "%.2f%s", latency, ((i != (MAXNODES -1)?", ":"")));
		}

		fprintf(stderr, "}%s\n", ((j != (MAXNODES - 1))?",":""));
	}
	fprintf(stderr, "};\n");

	fprintf(stderr, "\n\nBandwith Matrix\n\n");

	fprintf(stderr, "{\n");
	for (j = 0; j < MAXNODES; j++)
	{
		fprintf(stderr, "\t{");
		for (i = 0; i < MAXNODES; i++)
		{
			double bandwith;

			if ((i > ncuda) || (j > ncuda))
			{
				bandwith = -1.0;
			}
			else if (i != j)
			{
				/* Bandwith = (SIZE)/(time i -> ram + time ram -> j)*/
				double time_i_to_ram = (i==0)?0.0:cudadev_timing_dtoh[i-1];
				double time_ram_to_j = (j==0)?0.0:cudadev_timing_htod[j-1];
	
				double timing = time_i_to_ram + time_ram_to_j;
	
				bandwith = 1.0*SIZE/timing;
			}
			else {
				/* convention */
				bandwith = 0.0;
			}
	
			fprintf(stderr, "%.2f%s", bandwith, ((i != (MAXNODES -1)?", ":"")));
		}

		fprintf(stderr, "}%s\n", ((j != (MAXNODES - 1))?",":""));
	}

	fprintf(stderr, "};\n");

	return 0;
}
