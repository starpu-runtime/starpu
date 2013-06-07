/* Use the `task' attribute only when StarPU's GCC plug-in
   is available.   */
#ifdef STARPU_GCC_PLUGIN
# define __task  __attribute__ ((task))
#else
# define __task
#endif

static void matmul (const float *A, const float *B, float *C,
                    unsigned nx, unsigned ny, unsigned nz) __task;
