#define TYPE	float

#define CUBLAS_GEMM cublasSgemm
#define MAGMABLAS_GEMM magmablas_sgemm
#define CPU_GEMM	SGEMM
#define CPU_ASUM	SASUM
#define CPU_IAMAX	ISAMAX
#define STARPU_GEMM(name)	starpu_sgemm_##name

#define str(s) #s
#define xstr(s)        str(s)
#define STARPU_GEMM_STR(name)  xstr(STARPU_GEMM(name))

