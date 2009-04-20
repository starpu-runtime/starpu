void incA (__attribute__ ((unused)) void **alloc,
                __attribute__ ((unused)) void **in,
                __attribute__ ((unused)) void **inout,
                __attribute__ ((unused)) void **out)
{
	unsigned *v = inout[0];
	v[0]++;
}

void incC (__attribute__ ((unused)) void **alloc,
                __attribute__ ((unused)) void **in,
                __attribute__ ((unused)) void **inout,
                __attribute__ ((unused)) void **out)
{
	unsigned *v = inout[0];
	v[2]++;
}
