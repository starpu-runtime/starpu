# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
using Clang
using Clang.LibClang.LLVM_jll

function starpu_translate_headers()
    debug_print("Translating StarPU headers...")

    if !isdir(joinpath(fstarpu_build_dir(), "julia/gen"))
        mkdir(joinpath(fstarpu_build_dir(), "julia/gen"))
    end

    STARPU_BUILD_INCLUDE=joinpath(fstarpu_build_dir(), "include")
    STARPU_SRC_INCLUDE=joinpath(fstarpu_src_dir(), "include")
    STARPU_HEADERS = [joinpath(STARPU_BUILD_INCLUDE, header) for header in readdir(STARPU_BUILD_INCLUDE) if endswith(header, ".h")]
    if STARPU_SRC_INCLUDE != STARPU_BUILD_INCLUDE
        for header in readdir(STARPU_SRC_INCLUDE)
            if endswith(header, ".h")
                push!(STARPU_HEADERS, joinpath(STARPU_SRC_INCLUDE, header))
            end
        end
    end

    LIBCLANG_INCLUDE = joinpath(dirname(LLVM_jll.libclang_path), "..", "include", "clang-c") |> normpath

    clang_args = ["-I", STARPU_BUILD_INCLUDE, "-I", STARPU_SRC_INCLUDE]

    for header in find_std_headers()
        push!(clang_args, "-I")
        push!(clang_args, header)
    end

    only_select_symbols = Set(["starpu_task",
                               "starpu_cublas_init",
                               "starpu_codelet",
                               "starpu_data_filter",
                               "starpu_tag_t",
                               "starpu_perfmodel",
                               "starpu_perfmodel_type",
                               "starpu_data_handle_t",
                               "starpu_init",
                               "starpu_data_acquire_on_node",
                               "starpu_data_release_on_node",
                               "starpu_data_unregister",
                               "starpu_data_partition",
                               "starpu_data_unpartition",
                               "starpu_data_get_sub_data",
                               "starpu_data_map_filters",
                               "starpu_data_get_default_sequential_consistency_flag",
                               "starpu_data_set_default_sequential_consistency_flag",
                               "starpu_data_get_sequential_consistency_flag",
                               "starpu_data_set_sequential_consistency_flag",
                               "starpu_data_wont_use",
                               "starpu_matrix_data_register",
                               "starpu_block_data_register",
                               "starpu_vector_data_register",
                               "starpu_variable_data_register",
                               "starpu_memory_pin",
                               "starpu_memory_unpin",
                               "starpu_task_end_dep_add",
                               "starpu_task_end_dep_release",
                               "starpu_task_init",
                               "starpu_task_destroy",
                               "starpu_task_submit",
                               "starpu_task_wait",
                               "starpu_task_wait_for_n_submitted",
                               "starpu_tag_remove",
                               "starpu_tag_wait",
                               "starpu_tag_declare_deps_array",
                               "starpu_tag_notify_from_apps",
                               "starpu_task_declare_end_deps_array",
                               "starpu_task_declare_deps_array",
                               "starpu_iteration_push",
                               "starpu_iteration_pop",
                               "starpu_worker_get_count",
                               "starpu_cpu_worker_get_count",
                               "starpu_cuda_worker_get_count",
                               "starpu_opencl_worker_get_count",
                               "STARPU_CPU",
                               "STARPU_CUDA",
                               "STARPU_CUDA_ASYNC",
                               "STARPU_OPENCL",
                               "STARPU_MAIN_RAM",
                               "STARPU_NMAXBUFS",
                               "STARPU_USE_CUDA"])

    wc = init(; headers = STARPU_HEADERS,
              output_file = joinpath(fstarpu_build_dir(), "julia/gen/libstarpu_api.jl"),
              common_file = joinpath(fstarpu_build_dir(), "julia/gen/libstarpu_common.jl"),
              clang_includes = vcat(LIBCLANG_INCLUDE, CLANG_INCLUDE),
              clang_args = clang_args,
              header_library = x->"starpu_wrapper_library_name",
              clang_diagnostics = false,
              rewriter = x->x,
              only_select_symbols = only_select_symbols,
              fields_align = Dict((:starpu_pthread_spinlock_t,:taken) => 16)
              )

    run(wc)
end
