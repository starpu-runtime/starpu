void
foo(void)
{
	if (ret != CL_SUCCESS)
		foo();

	if (ret != cudaSuccess) {
		fprintf(stderr, "Fail.\n");
		STARPU_ABORT();
	}

	if (ret == cudaSuccess)
		do_stg();

}
