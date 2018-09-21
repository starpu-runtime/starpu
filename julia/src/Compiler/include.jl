


include("utils.jl")

include("Expressions/simple_expressions.jl")
include("Expressions/affect.jl")
include("Expressions/block.jl")
include("Expressions/call.jl")
include("Expressions/cuda_call.jl")
include("Expressions/field.jl")
include("Expressions/interval.jl")
include("Expressions/for.jl")
include("Expressions/typed.jl")
include("Expressions/function.jl")
include("Expressions/if.jl")
include("Expressions/ref.jl")
include("Expressions/return.jl")
include("Expressions/while.jl")

include("parsing.jl")

include("expression_manipulation.jl")

include("C/substitute_args.jl")
include("C/substitute_func_calls.jl")
include("C/substitute_indexing.jl")
include("C/add_for_loop_declarations.jl")
include("C/flatten_blocks.jl")
include("C/create_cpu_kernel.jl")

include("Cuda/indep_for.jl")
include("Cuda/indep_for_kernel_ids.jl")
include("Cuda/create_cuda_kernel.jl")


include("Generate_files/c_files.jl")
include("Generate_files/cuda_files.jl")
include("Generate_files/so_files.jl")
