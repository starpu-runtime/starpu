static void
bad_0(void)
{
	cudaError_t ret;
	ret = cudaMalloc(&addr, size);

	ret = foo();
}

static int
bad_1(void)
{
	cudaError_t cures;
	cures = cudaMemcpy(NULL, NULL, 0);

	return 0;
}

static void
good_0(void)
{
	cudaError_t st;
	st = cudaMemcpy(dst, src, size);
	if (st)
		do_stg_good();
}

static void
good_1(void)
{
	cudaError_t st;
	st = cudaMemcpy(dst, src, size);
	if (!st)
		report_error();
	else
		lol();
}

static void
good_2(void)
{
	cudaError_t st;
	st = cudaMemcpy(dst, src, size);
	if (STARPU_UNLIKELY(!st))
		report_error();
}

static void
good_3(void)
{
	cudaError_t st;
	st = cudaMemcpy(dst, src, size);
	if (STARPU_UNLIKELY(!st))
		report_error();
	else
		foo();
}

static void
good_4(void)
{
	cudaError_t st;
	st = cudaMemcpy(dst, src, size);
	if (st != cudaSuccess)
		error();
}

static void
good_5(void)
{
	cudaError_t st;
	st = cudaMemcpy(dst, src, size);
	if (st == cudaSuccess)
		cool();
}


static void
no_assignment_bad_0(void)
{
	cudaGetLastError();
}

static void
no_assignment_bad_1(void)
{
	cudaMemcpy(dst, src, size);
}

static void
no_assignment_good_0(void)
{
	(void) cudaGetLastError();
}
