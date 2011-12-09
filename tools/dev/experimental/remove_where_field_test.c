struct dummy_struct
{
	int where;
};


/* Simple example : remove the where field */
struct starpu_codelet cl = {
	.cuda_func = bar,
	.where = STARPU_CPU | STARPU_OPENCL,
	.cpu_func = foo
};


void
dummy(void)
{
	cl.where = STARPU_CUDA;
	starpu_codelet_t *clp = &cl;
	clp->where = STARPU_CPU; /* Must be removed */

	struct dummy_struct ds;
	struct dummy_struct *dsp = &ds;
	ds.where = 12;   /* Must not be removed */
	dsp->where = 12; /* Must not be removed */ 
}
