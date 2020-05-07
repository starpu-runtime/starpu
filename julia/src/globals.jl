
global starpu_wrapper_library_handle = C_NULL

global starpu_tasks_library_handle = C_NULL

global starpu_target=STARPU_CPU

global generated_cuda_kernel_file_name = "PRINT TO STDOUT"
global generated_cpu_kernel_file_name = "PRINT TO STDOUT"

export CPU_CODELETS
global CPU_CODELETS=Dict{String,String}()

export CUDA_CODELETS
global CUDA_CODELETS=Dict{String,String}()

export CODELETS_SCALARS
global CODELETS_SCALARS=Dict{String,Any}()

export CODELETS_PARAMS_STRUCT
global CODELETS_PARAMS_STRUCT=Dict{String,Any}()

global starpu_type_traduction_dict = Dict(
    Int32 => "int32_t",
    UInt32 => "uint32_t",
    Float32 => "float",
    Int64 => "int64_t",
    UInt64 => "uint64_t",
    Float64 => "double",
    Nothing => "void"
)
export starpu_type_traduction_dict

global perfmodels = Vector{starpu_perfmodel}()
