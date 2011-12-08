#include <stdio.h>
#include <sdtdlib.h>
#include "lol.h"

/*
 * Old format
 */
struct starpu_codelet cl1 = {
	.where = STARPU_CPU,
	.cpu_func = foo
};

/*
 * New format : it must not be changed !
 */
struct starpu_codelet cl2 = {
	.cpu_funcs = {foo, NULL}
};

/*
 * Maybe we added the cpu_funcs fields, but forgot to remove the cpu_func one.
 */
struct starpu_codelet cl3 = {
	.cpu_func = foo,
	.cpu_funcs = { foo, NULL }
};

/*
 * Old multiimplementations format, but not terminated by NULL
 * XXX : NULL is not added.
 */
struct starpu_codelet cl4 = {
	.cpu_func = STARPU_MULTIPLE_CPU_IMPLEMENTATIONS,
	.cpu_funcs = { foo, bar }
};

/*
 * Old multiimplementations format, terminated by NULL
 */
struct starpu_codelet cl5 = {
	.cpu_func = STARPU_MULTIPLE_CPU_IMPLEMENTATIONS,
	.cpu_funcs = { foo, bar, NULL }
};
