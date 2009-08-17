#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

#include <starpu.h>
#include <fftw3.h>
#ifdef USE_CUDA
#include <cufft.h>
#endif

#define NSAMPLES	8192
#define SAMPLERATE	44100

/* This is a band filter, we want to stop everything that is not between LOWFREQ and HIGHFREQ*/
/* LOWFREQ < i * SAMPLERATE / NSAMPLE */
#define LOWFREQ	800U
#define HIFREQ	1100U
static unsigned lowfreq_index = (LOWFREQ*NSAMPLES)/SAMPLERATE;
static unsigned hifreq_index = (HIFREQ*NSAMPLES)/SAMPLERATE;

static const size_t headersize = 37+9;
static FILE *infile, *outfile;
static FILE *file, *file2, *file3;

unsigned length_data;

/* buffer containing input WAV data */
float *A;
/* working buffer for FFT algorithm */
fftwf_complex *outdata;

starpu_data_handle A_handle, outdata_handle;

/* 
 *	Functions to Manipulate WAV files 
 */
unsigned get_wav_data_bytes_length(FILE *file)
{
	/* this is clearly suboptimal !! */
	fseek(file, headersize, SEEK_SET);

	unsigned cnt = 0;
	while (fgetc(file) != EOF)
		cnt++;

	return cnt;
}

void copy_wav_header(FILE *srcfile, FILE *dstfile)
{
	unsigned char buffer[128];

	fseek(srcfile, 0, SEEK_SET);
	fseek(dstfile, 0, SEEK_SET);

	fread(buffer, 1, headersize, infile);	
	fwrite(buffer, 1, headersize, outfile);	
}

void read_16bit_wav(FILE *infile, unsigned size, float *arrayout, FILE *save_file)
{
	int v;
	unsigned currentpos = 0;

	/* we skip the header to only keep the data */
	fseek(infile, headersize, SEEK_SET);
	
	for (v=0;v<size;v++) {
		signed char val = (signed char)fgetc(infile);
		signed char val2 = (signed char)fgetc(infile);

		arrayout[v] = 256*val2 + val;

                fprintf(save_file, "%d %f\n", currentpos++, arrayout[v]);
	}
}

/* we only write the data, not the header !*/
void write_16bit_wav(FILE *outfile, unsigned size, float *arrayin, FILE *save_file)
{
	int v;
	unsigned currentpos = 0;

	/* we assume that the header is copied using copy_wav_header */
	fseek(outfile, headersize, SEEK_SET);
	
	for (v=0;v<size;v++) {
		signed char val = ((int)arrayin[v]) % 256; 
		signed char val2  = ((int)arrayin[v]) / 256;

		fputc(val, outfile);
		fputc(val2, outfile);

		if (save_file)
	                fprintf(save_file, "%d %f\n", currentpos++, arrayin[v]);
	}
}

#ifdef USE_CUDA
static void band_filter_kernel_gpu(starpu_data_interface_t *descr, __attribute__((unused)) void *arg)
{
	float *localA = (float *)descr[0].vector.ptr;
	cufftComplex *localout = (cufftComplex *)descr[1].vector.ptr;

	/* create plans */
	cufftHandle plan;
	cufftHandle inv_plan;

	/* we should reuse that ! */
	/* TODO check error values */
	cufftPlan1d(&plan, NSAMPLES, CUFFT_R2C, 1);
	cufftPlan1d(&inv_plan, NSAMPLES, CUFFT_C2R, 1);

	/* FFT */
	cufftExecR2C(plan, localA, localout);
	
	
	/* filter low freqs */
	cudaMemset(&localout[0], 0, lowfreq_index*sizeof(fftwf_complex));

	/* filter high freqs */
	cudaMemset(&localout[hifreq_index], NSAMPLES/2, (NSAMPLES/2 - hifreq_index)*sizeof(fftwf_complex));


	/* inverse FFT */
	cufftExecC2R(inv_plan, localout, localA);

	/* FFTW does not normalize its output ! */
	cublasSscal (NSAMPLES, 1.0f/NSAMPLES, localA, 1);

	cufftDestroy(plan);
	cufftDestroy(inv_plan);
}
#endif

static pthread_mutex_t fftw_mutex = PTHREAD_MUTEX_INITIALIZER;

static void band_filter_kernel_cpu(starpu_data_interface_t *descr, __attribute__((unused)) void *arg)
{
	float *localA = (float *)descr[0].vector.ptr;
	fftwf_complex *localout = (fftwf_complex *)descr[1].vector.ptr;

	/* create plans, only "fftwf_execute" is thread safe in FFTW ... */
	pthread_mutex_lock(&fftw_mutex);
	fftwf_plan rplan = fftwf_plan_dft_r2c_1d(NSAMPLES, localA, localout, FFTW_ESTIMATE);
	fftwf_plan inv_rplan = fftwf_plan_dft_c2r_1d(NSAMPLES, localout, localA, FFTW_ESTIMATE);
	pthread_mutex_unlock(&fftw_mutex);

	/* FFT */
	fftwf_execute(rplan);
	
	/* filter low freqs */
	memset(&localout[0], 0, lowfreq_index*sizeof(fftwf_complex));

	/* filter high freqs */
	memset(&localout[hifreq_index], NSAMPLES/2, (NSAMPLES/2 - hifreq_index)*sizeof(fftwf_complex));

	/* inverse FFT */
	fftwf_execute(inv_rplan);

	/* FFTW does not normalize its output ! */
	/* TODO use BLAS ?*/
	int i;
	for (i = 0; i < NSAMPLES; i++)
		localA[i] /= NSAMPLES;

	pthread_mutex_lock(&fftw_mutex);
	fftwf_destroy_plan(rplan);
	fftwf_destroy_plan(inv_rplan);
	pthread_mutex_unlock(&fftw_mutex);
}

static starpu_codelet band_filter_cl = {
	.where = CORE|CUBLAS,
#ifdef USE_CUDA
	.cublas_func = band_filter_kernel_gpu,
#endif
	.core_func = band_filter_kernel_cpu,
	.nbuffers = 2
};

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static unsigned ntasks_remaining = 0;

void callback(void *arg __attribute__((unused)))
{
	if (STARPU_ATOMIC_ADD(&ntasks_remaining, -1) == 0)
	{
		pthread_mutex_lock(&mutex);
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
	}
}

void create_starpu_task(unsigned iter)
{
	struct starpu_task *task = starpu_task_create();

	task->cl = &band_filter_cl;

	task->buffers[0].handle = get_sub_data(A_handle, 1, iter);
	task->buffers[0].mode = STARPU_RW;
	task->buffers[1].handle = get_sub_data(outdata_handle, 1, iter);
	task->buffers[1].mode = STARPU_W;

	task->callback_func = callback;
	task->callback_arg = NULL;

	starpu_submit_task(task);
}

static void init_problem(void)
{
	infile = fopen("input.wav", "r");
	outfile = fopen("output.wav", "w+");

	file = fopen("input.dat", "w");
	file2 = fopen("input.raw", "w");
	file3 = fopen("output.raw", "w");

	/* copy input's header into output WAV  */
	copy_wav_header(infile, outfile);

	/* read length of input WAV's data */
	/* each element is 2 bytes long (16bits)*/
	length_data = get_wav_data_bytes_length(infile)/2;

	/* allocate a buffer to store the content of input file */
	A = calloc(length_data, sizeof(float));

	/* allocate working buffer (this could be done online, but we'll keep it simple) */
	outdata = malloc(length_data*sizeof(fftwf_complex));

	/* read input data into buffer "A" */
	read_16bit_wav(infile, length_data, A, file2);
}

int main(int argc, char **argv)
{
	init_problem();

	unsigned niter = length_data/NSAMPLES;
	ntasks_remaining = niter;

	/* launch StarPU */
	starpu_init(NULL);

	starpu_register_vector_data(&A_handle, 0, (uintptr_t)A, niter*NSAMPLES, sizeof(float));
	starpu_register_vector_data(&outdata_handle, 0, (uintptr_t)outdata, niter*NSAMPLES, sizeof(fftwf_complex));

	starpu_filter f = 
	{
		.filter_func = starpu_block_filter_func_vector,
		.filter_arg = niter
	};

	starpu_partition_data(A_handle, &f);
	starpu_partition_data(outdata_handle, &f);

	unsigned iter;
	for (iter = 0; iter < niter; iter++)
	{
		create_starpu_task(iter);
	}

	/* wait for the termination of all tasks */
	pthread_mutex_lock(&mutex);
	while (ntasks_remaining != 0)
		pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	/* make sure that the output is in RAM before quitting StarPU */
	starpu_unpartition_data(A_handle, 0);
	starpu_delete_data(A_handle);

	/* we are done ! */
	starpu_shutdown();

	write_16bit_wav(outfile, length_data, A, file3);

	fclose(file);
	fclose(file2);
	fclose(file3);
	fclose(infile);
	fclose(outfile);

	return 0;
}
