#define TYPE	double

#define CUBLAS_GEMM cublasDgemm
#define MAGMABLAS_GEMM magmablas_dgemm
#define CPU_GEMM	DGEMM
#define CPU_ASUM	DASUM
#define CPU_IAMAX	IDAMAX
#define STARPU_GEMM(name)	starpu_dgemm_##name

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_GEMM_STR(name)  xstr(STARPU_GEMM(name))

