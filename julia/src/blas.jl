@enum STARPU_BLAS begin
    STARPU_SAXPY
end

cuda_blas_codelets = Dict(STARPU_SAXPY => "julia_saxpy_cuda_codelet")
cpu_blas_codelets = Dict(STARPU_SAXPY => "julia_saxpy_cpu_codelet")
