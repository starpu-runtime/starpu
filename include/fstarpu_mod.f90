! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016  Inria
!
! StarPU is free software; you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation; either version 2.1 of the License, or (at
! your option) any later version.
!
! StarPU is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!
! See the GNU Lesser General Public License in COPYING.LGPL for more details.

module fstarpu_mod
        use iso_c_binding
        implicit none

        ! Note: Constants truly are intptr_t, but are declared as c_ptr to be
        ! readily usable in c_ptr arrays to mimic variadic functions.
        ! A side effect, though, is that such constants cannot be logically
        ! 'or'-ed.
        type(c_ptr), bind(C) :: FSTARPU_R
        type(c_ptr), bind(C) :: FSTARPU_W
        type(c_ptr), bind(C) :: FSTARPU_RW
        type(c_ptr), bind(C) :: FSTARPU_SCRATCH
        type(c_ptr), bind(C) :: FSTARPU_REDUX

        type(c_ptr), bind(C) :: FSTARPU_DATA
        type(c_ptr), bind(C) :: FSTARPU_VALUE
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX

        type(c_ptr), bind(C) :: FSTARPU_SZ_INT4
        type(c_ptr), bind(C) :: FSTARPU_SZ_INT8

        type(c_ptr), bind(C) :: FSTARPU_SZ_REAL4
        type(c_ptr), bind(C) :: FSTARPU_SZ_REAL8
        interface
                ! == starpu.h ==

                ! starpu_conf_init: see fstarpu_conf_allocate

                function fstarpu_conf_allocate () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_conf_allocate
                end function fstarpu_conf_allocate

                subroutine fstarpu_conf_free (cl) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                end subroutine fstarpu_conf_free

                subroutine fstarpu_conf_set_sched_policy_name (conf, policy_name) bind(C)
                        use iso_c_binding, only: c_ptr, c_char
                        type(c_ptr), value, intent(in) :: conf
                        character(c_char), intent(in) :: policy_name
                end subroutine fstarpu_conf_set_sched_policy_name

                subroutine fstarpu_conf_set_min_prio (conf, min_prio) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: conf
                        integer(c_int), value, intent(in) :: min_prio
                end subroutine fstarpu_conf_set_min_prio

                subroutine fstarpu_conf_set_max_prio (conf, max_prio) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: conf
                        integer(c_int), value, intent(in) :: max_prio
                end subroutine fstarpu_conf_set_max_prio

                subroutine fstarpu_conf_set_ncpu (conf, ncpu) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: conf
                        integer(c_int), value, intent(in) :: ncpu
                end subroutine fstarpu_conf_set_ncpu

                subroutine fstarpu_conf_set_ncuda (conf, ncuda) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: conf
                        integer(c_int), value, intent(in) :: ncuda
                end subroutine fstarpu_conf_set_ncuda

                subroutine fstarpu_conf_set_nopencl (conf, nopencl) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: conf
                        integer(c_int), value, intent(in) :: nopencl
                end subroutine fstarpu_conf_set_nopencl

                subroutine fstarpu_conf_set_nmic (conf, nmic) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: conf
                        integer(c_int), value, intent(in) :: nmic
                end subroutine fstarpu_conf_set_nmic

                subroutine fstarpu_conf_set_nscc (conf, nscc) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: conf
                        integer(c_int), value, intent(in) :: nscc
                end subroutine fstarpu_conf_set_nscc

                ! starpu_init: see fstarpu_init
                ! starpu_initialize: see fstarpu_init

                ! void starpu_pause(void);
                subroutine fstarpu_pause() bind(C,name="starpu_pause")
                end subroutine fstarpu_pause

                ! void starpu_resume(void);
                subroutine fstarpu_resume() bind(C,name="starpu_resume")
                end subroutine fstarpu_resume

                ! void starpu_shutdown(void);
                subroutine fstarpu_shutdown () bind(C,name="starpu_shutdown")
                end subroutine fstarpu_shutdown

                ! starpu_topology_print

                ! int starpu_asynchronous_copy_disabled(void);
                function fstarpu_asynchronous_copy_disabled() bind(C,name="starpu_asynchronous_copy_disabled")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_asynchronous_copy_disabled
                end function fstarpu_asynchronous_copy_disabled

                ! int starpu_asynchronous_cuda_copy_disabled(void);
                function fstarpu_asynchronous_cuda_copy_disabled() bind(C,name="starpu_asynchronous_cuda_copy_disabled")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_asynchronous_cuda_copy_disabled
                end function fstarpu_asynchronous_cuda_copy_disabled

                ! int starpu_asynchronous_opencl_copy_disabled(void);
                function fstarpu_asynchronous_opencl_copy_disabled() bind(C,name="starpu_asynchronous_opencl_copy_disabled")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_asynchronous_opencl_copy_disabled
                end function fstarpu_asynchronous_opencl_copy_disabled

                ! int starpu_asynchronous_mic_copy_disabled(void);
                function fstarpu_asynchronous_mic_copy_disabled() bind(C,name="starpu_asynchronous_mic_copy_disabled")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_asynchronous_mic_copy_disabled
                end function fstarpu_asynchronous_mic_copy_disabled

                ! void starpu_display_stats();
                subroutine fstarpu_display_stats() bind(C,name="starpu_display_stats")
                end subroutine fstarpu_display_stats

                ! void starpu_get_version(int *major, int *minor, int *release);
                subroutine fstarpu_get_version(major,minor,release) bind(C,name="starpu_get_version")
                        use iso_c_binding, only: c_int
                        integer(c_int), intent(out) :: major,minor,release
                end subroutine fstarpu_get_version

                ! == starpu_worker.h ==

                ! unsigned starpu_worker_get_count(void);
                function fstarpu_worker_get_count() bind(C,name="starpu_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_get_count
                end function fstarpu_worker_get_count

                ! unsigned starpu_combined_worker_get_count(void);
                function fstarpu_combined_worker_get_count() bind(C,name="starpu_combined_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_combined_worker_get_count
                end function fstarpu_combined_worker_get_count

                ! unsigned starpu_worker_is_combined_worker(int id);
                function fstarpu_worker_is_combined_worker(id) bind(C,name="starpu_worker_is_combined_worker")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_is_combined_worker
                        integer(c_int), value, intent(in) :: id
                end function fstarpu_worker_is_combined_worker


                ! unsigned starpu_cpu_worker_get_count(void);
                function fstarpu_cpu_worker_get_count() bind(C,name="starpu_cpu_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_cpu_worker_get_count
                end function fstarpu_cpu_worker_get_count

                ! unsigned starpu_cuda_worker_get_count(void);
                function fstarpu_cuda_worker_get_count() bind(C,name="starpu_cuda_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_cuda_worker_get_count
                end function fstarpu_cuda_worker_get_count

                ! unsigned starpu_opencl_worker_get_count(void);
                function fstarpu_opencl_worker_get_count() bind(C,name="starpu_opencl_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_opencl_worker_get_count
                end function fstarpu_opencl_worker_get_count

                ! unsigned starpu_mic_worker_get_count(void);
                function fstarpu_mic_worker_get_count() bind(C,name="starpu_mic_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_mic_worker_get_count
                end function fstarpu_mic_worker_get_count

                ! unsigned starpu_scc_worker_get_count(void);
                function fstarpu_scc_worker_get_count() bind(C,name="starpu_scc_worker_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_scc_worker_get_count
                end function fstarpu_scc_worker_get_count

                ! int starpu_worker_get_id(void);
                function fstarpu_worker_get_id() bind(C,name="starpu_worker_get_id")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_get_id
                end function fstarpu_worker_get_id

                ! _starpu_worker_get_id_check
                ! starpu_worker_get_id_check

                ! int starpu_worker_get_bindid(int workerid);
                function fstarpu_worker_get_bindid(id) bind(C,name="starpu_worker_get_bindid")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_get_bindid
                        integer(c_int), value, intent(in) :: id
                end function fstarpu_worker_get_bindid

                ! int starpu_combined_worker_get_id(void);
                function fstarpu_combined_worker_get_id() bind(C,name="starpu_combined_worker_get_id")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_combined_worker_get_id
                end function fstarpu_combined_worker_get_id

                ! int starpu_combined_worker_get_size(void);
                function fstarpu_combined_worker_get_size() bind(C,name="starpu_combined_worker_get_size")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_combined_worker_get_size
                end function fstarpu_combined_worker_get_size

                ! int starpu_combined_worker_get_rank(void);
                function fstarpu_combined_worker_get_rank() bind(C,name="starpu_combined_worker_get_rank")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_combined_worker_get_rank
                end function fstarpu_combined_worker_get_rank

                ! enum starpu_worker_archtype starpu_worker_get_type(int id);
                ! int starpu_worker_get_count_by_type(enum starpu_worker_archtype type);
                ! int starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);
                ! int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);
                ! int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);
                ! void starpu_worker_get_name(int id, char *dst, size_t maxlen);

                ! int starpu_worker_get_devid(int id);
                function fstarpu_worker_get_devid(id) bind(C,name="starpu_worker_get_devid")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_get_devid
                        integer(c_int), value, intent(in) :: id
                end function fstarpu_worker_get_devid

                ! int starpu_worker_get_mp_nodeid(int id);
                function fstarpu_worker_get_mp_nodeid(id) bind(C,name="starpu_worker_get_mp_nodeid")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_get_mp_nodeid
                        integer(c_int), value, intent(in) :: id
                end function fstarpu_worker_get_mp_nodeid

                ! struct starpu_tree* starpu_workers_get_tree(void);
                ! unsigned starpu_worker_get_sched_ctx_list(int worker, unsigned **sched_ctx);

                ! unsigned starpu_worker_is_blocked(int workerid);
                function fstarpu_worker_is_blocked(id) bind(C,name="starpu_worker_is_blocked")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_is_blocked
                        integer(c_int), value, intent(in) :: id
                end function fstarpu_worker_is_blocked

                ! unsigned starpu_worker_is_slave_somewhere(int workerid);
                function fstarpu_worker_is_slave_somewhere(id) bind(C,name="starpu_worker_is_slave_somewhere")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_is_slave_somewhere
                        integer(c_int), value, intent(in) :: id
                end function fstarpu_worker_is_slave_somewhere

                ! char *starpu_worker_get_type_as_string(enum starpu_worker_archtype type);
                ! int starpu_bindid_get_workerids(int bindid, int **workerids);

                ! == starpu_task.h ==

                ! starpu_tag_declare_deps
                ! starpu_tag_declare_deps_array
                ! starpu_task_declare_deps_array
                ! starpu_tag_wait
                ! starpu_tag_wait_array
                ! starpu_tag_notify_from_apps
                ! starpu_tag_restart
                ! starpu_tag_remove
                ! starpu_task_init
                ! starpu_task_clean
                ! starpu_task_create
                ! starpu_task_destroy
                ! starpu_task_submit
                ! starpu_task_submit_to_ctx
                ! starpu_task_finished
                ! starpu_task_wait

                ! int starpu_task_wait_for_all(void);
                subroutine fstarpu_task_wait_for_all () bind(C,name="starpu_task_wait_for_all")
                end subroutine fstarpu_task_wait_for_all

                ! starpu_task_wait_for_n_submitted
                ! starpu_task_wait_for_all_in_ctx
                subroutine fstarpu_task_wait_for_all_in_ctx (ctx) bind(C,name="starpu_task_wait_for_all_in_ctx")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: ctx
                end subroutine fstarpu_task_wait_for_all_in_ctx

                ! starpu_task_wait_for_n_submitted_in_ctx
                ! starpu_task_wait_for_no_ready
                ! starpu_task_nready
                ! starpu_task_nsubmitted
                ! starpu_codelet_init
                ! starpu_codelet_display_stats
                ! starpu_task_get_current
                ! starpu_parallel_task_barrier_init
                ! starpu_parallel_task_barrier_init_n
                ! starpu_task_dup
                ! starpu_task_set_implementation
                ! starpu_task_get_implementation
                ! --

                function fstarpu_codelet_allocate () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_codelet_allocate
                end function fstarpu_codelet_allocate

                subroutine fstarpu_codelet_free (cl) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                end subroutine fstarpu_codelet_free

                subroutine fstarpu_codelet_set_name (cl, cl_name) bind(C)
                        use iso_c_binding, only: c_ptr, c_char
                        type(c_ptr), value, intent(in) :: cl
                        character(c_char), intent(in) :: cl_name
                end subroutine fstarpu_codelet_set_name

                subroutine fstarpu_codelet_add_cpu_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_cpu_func

                subroutine fstarpu_codelet_add_cuda_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_cuda_func

                subroutine fstarpu_codelet_add_opencl_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_opencl_func

                subroutine fstarpu_codelet_add_mic_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_mic_func

                subroutine fstarpu_codelet_add_scc_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_scc_func

                subroutine fstarpu_codelet_add_buffer (cl, mode) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_ptr), value, intent(in) :: mode ! C function expects an intptr_t
                end subroutine fstarpu_codelet_add_buffer

                ! == starpu_data_interface.h ==

                ! uintptr_t starpu_malloc_on_node_flags(unsigned dst_node, size_t size, int flags);

                ! uintptr_t starpu_malloc_on_node(unsigned dst_node, size_t size);
                function fstarpu_malloc_on_node(node,sz) bind(C,name="starpu_malloc_on_node")
                        use iso_c_binding, only: c_int,c_intptr_t,c_size_t
                        integer(c_intptr_t) :: fstarpu_malloc_on_node
                        integer(c_int), value, intent(in) :: node
                        integer(c_size_t), value, intent(in) :: sz
                end function fstarpu_malloc_on_node

                ! void starpu_free_on_node_flags(unsigned dst_node, uintptr_t addr, size_t size, int flags);

                ! void starpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size);
                subroutine fstarpu_free_on_node(node,addr,sz) bind(C,name="starpu_free_on_node")
                        use iso_c_binding, only: c_int,c_intptr_t,c_size_t
                        integer(c_int), value, intent(in) :: node
                        integer(c_intptr_t), value, intent(in) :: addr
                        integer(c_size_t), value, intent(in) :: sz
                end subroutine fstarpu_free_on_node

                ! void starpu_malloc_on_node_set_default_flags(unsigned node, int flags);

                ! int starpu_data_interface_get_next_id(void);
                ! void starpu_data_register(starpu_data_handle_t *handleptr, unsigned home_node, void *data_interface, struct starpu_data_interface_ops *ops);


                ! void starpu_data_ptr_register(starpu_data_handle_t handle, unsigned node);
                subroutine fstarpug_data_ptr_register (dh,node) bind(C,name="starpu_data_ptr_register")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                end subroutine fstarpug_data_ptr_register

                ! void starpu_data_register_same(starpu_data_handle_t *handledst, starpu_data_handle_t handlesrc);
                subroutine fstarpu_data_register_same (dh_dst,dh_src) bind(C,name="starpu_data_register_same")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), intent(out) :: dh_dst
                        type(c_ptr), value, intent(in) :: dh_src
                end subroutine fstarpu_data_register_same

                ! void *starpu_data_handle_to_pointer(starpu_data_handle_t handle, unsigned node);
                function fstarpu_data_handle_to_pointer (dh,node) bind(C,name="starpu_data_handle_to_pointer")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_data_handle_to_pointer
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                end function fstarpu_data_handle_to_pointer

                ! void *starpu_data_get_local_ptr(starpu_data_handle_t handle);
                function fstarpu_data_get_local_ptr (dh) bind(C,name="starpu_data_get_local_ptr")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_data_get_local_ptr
                        type(c_ptr), value, intent(in) :: dh
                end function fstarpu_data_get_local_ptr

                ! void *starpu_data_get_interface_on_node(starpu_data_handle_t handle, unsigned memory_node);

                ! == starpu_data_interface.h: matrix ==

                ! starpu_matrix_data_register: see fstarpu_matrix_data_register
                function fstarpu_matrix_data_register(matrix, ldy, ny, nx, elt_size, ram) bind(C)
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr) :: fstarpu_matrix_data_register
                        type(c_ptr), value, intent(in) :: matrix
                        integer(c_int), value, intent(in) :: ldy
                        integer(c_int), value, intent(in) :: ny
                        integer(c_int), value, intent(in) :: nx
                        integer(c_size_t), value, intent(in) :: elt_size
                        integer(c_int), value, intent(in) :: ram
                end function fstarpu_matrix_data_register

                ! starpu_matrix_ptr_register

                function fstarpu_matrix_get_ptr(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_matrix_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ptr

                function fstarpu_matrix_get_ld(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_ld
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ld

                function fstarpu_matrix_get_ny(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_ny
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ny

                function fstarpu_matrix_get_nx(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_nx
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_nx

                ! == starpu_data_interface.h: vector ==

                ! starpu_vector_data_register: see fstarpu_vector_data_register
                function fstarpu_vector_data_register(vector, nx, elt_size, ram) bind(C)
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr) :: fstarpu_vector_data_register
                        type(c_ptr), value, intent(in) :: vector
                        integer(c_int), value, intent(in) :: nx
                        integer(c_size_t), value, intent(in) :: elt_size
                        integer(c_int), value, intent(in) :: ram
                end function fstarpu_vector_data_register

                ! starpu_vector_ptr_register

                function fstarpu_vector_get_ptr(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_vector_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_vector_get_ptr

                function fstarpu_vector_get_nx(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_vector_get_nx
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_vector_get_nx

                ! == starpu_data_interface.h: variable ==

                ! starpu_variable_data_register: see fstarpu_variable_data_register
                function fstarpu_variable_data_register(ptr, sz, ram) bind(C)
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr) :: fstarpu_variable_data_register
                        type(c_ptr), value, intent(in) :: ptr
                        integer(c_size_t), value, intent(in) :: sz
                        integer(c_int), value, intent(in) :: ram
                end function fstarpu_variable_data_register

                ! starpu_variable_ptr_register

                function fstarpu_variable_get_ptr(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_variable_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_variable_get_ptr

                ! == starpu_data_interface.h: void ==

                ! starpu_void_data_register: see fstarpu_void_data_register
                function fstarpu_void_data_register() bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_void_data_register
                end function fstarpu_void_data_register

                ! == starpu_data.h ==

                ! void starpu_data_unregister(starpu_data_handle_t handle);
                subroutine fstarpu_data_unregister (dh) bind(C,name="starpu_data_unregister")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_unregister

                ! void starpu_data_unregister_no_coherency(starpu_data_handle_t handle);
                subroutine fstarpu_data_unregister_no_coherency (dh) bind(C,name="starpu_data_unregister_no_coherency")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_unregister_no_coherency

                ! void starpu_data_unregister_submit(starpu_data_handle_t handle);
                subroutine fstarpu_data_unregister_submit (dh) bind(C,name="starpu_data_unregister_submit")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_unregister_submit

                ! void starpu_data_invalidate(starpu_data_handle_t handle);
                subroutine fstarpu_data_invalidate (dh) bind(C,name="starpu_data_invalidate")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_invalidate

                ! void starpu_data_invalidate_submit(starpu_data_handle_t handle);
                subroutine fstarpu_data_invalidate_submit (dh) bind(C,name="starpu_data_invalidate_submit")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_invalidate_submit

                ! void starpu_data_advise_as_important(starpu_data_handle_t handle, unsigned is_important);
                subroutine fstarpu_data_advise_as_important (dh,is_important) bind(C,name="starpu_data_advise_as_important")
                        use iso_c_binding, only: c_ptr,c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: is_important
                end subroutine fstarpu_data_advise_as_important

                ! starpu_data_acquire: see fstarpu_data_acquire
                subroutine fstarpu_data_acquire (dh, mode) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: mode ! C function expects an intptr_t
                end subroutine fstarpu_data_acquire

                ! int starpu_data_acquire_on_node(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode);
                ! int starpu_data_acquire_cb(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg);
                ! int starpu_data_acquire_on_node_cb(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg);
                ! int starpu_data_acquire_cb_sequential_consistency(starpu_data_handle_t handle, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);
                ! int starpu_data_acquire_on_node_cb_sequential_consistency(starpu_data_handle_t handle, int node, enum starpu_data_access_mode mode, void (*callback)(void *), void *arg, int sequential_consistency);

                ! void starpu_data_release(starpu_data_handle_t handle);
                subroutine fstarpu_data_release (dh) bind(C,name="starpu_data_release")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_release

                ! void starpu_data_release_on_node(starpu_data_handle_t handle, int node);
                subroutine fstarpu_data_release_on_node (dh, node) bind(C,name="starpu_data_release_on_node")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                end subroutine fstarpu_data_release_on_node

                ! void starpu_data_assign_arbiter(starpu_data_handle_t handle, starpu_arbiter_t arbiter);
                ! void starpu_arbiter_destroy(starpu_arbiter_t arbiter);

                ! void starpu_data_display_memory_stats();
                subroutine fstarpu_display_memory_stats() bind(C,name="starpu_display_memory_stats")
                end subroutine fstarpu_display_memory_stats

                ! int starpu_data_request_allocation(starpu_data_handle_t handle, unsigned node);
                subroutine fstarpu_data_request_allocation (dh, node) &
                                bind(C,name="starpu_data_request_allocation")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                end subroutine fstarpu_data_request_allocation

                ! int starpu_data_fetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);
                subroutine fstarpu_data_fetch_on_node (dh, node, async) &
                                bind(C,name="starpu_data_fetch_on_node")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                        integer(c_int), value, intent(in) :: async
                end subroutine fstarpu_data_fetch_on_node

                ! int starpu_data_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);
                subroutine fstarpu_data_prefetch_on_node (dh, node, async) &
                                bind(C,name="starpu_data_prefetch_on_node")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                        integer(c_int), value, intent(in) :: async
                end subroutine fstarpu_data_prefetch_on_node

                ! int starpu_data_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);
                subroutine fstarpu_data_prefetch_on_node_prio (dh, node, async, prio) &
                                bind(C,name="starpu_data_prefetch_on_node_prio")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                        integer(c_int), value, intent(in) :: async
                        integer(c_int), value, intent(in) :: prio
                end subroutine fstarpu_data_prefetch_on_node_prio

                ! int starpu_data_idle_prefetch_on_node(starpu_data_handle_t handle, unsigned node, unsigned async);
                subroutine fstarpu_data_idle_prefetch_on_node (dh, node, async) &
                                bind(C,name="starpu_data_idle_prefetch_on_node")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                        integer(c_int), value, intent(in) :: async
                end subroutine fstarpu_data_idle_prefetch_on_node

                ! int starpu_data_idle_prefetch_on_node_prio(starpu_data_handle_t handle, unsigned node, unsigned async, int prio);
                subroutine fstarpu_data_idle_prefetch_on_node_prio (dh, node, async, prio) &
                                bind(C,name="starpu_data_idle_prefetch_on_node_prio")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                        integer(c_int), value, intent(in) :: async
                        integer(c_int), value, intent(in) :: prio
                end subroutine fstarpu_data_idle_prefetch_on_node_prio

                ! void starpu_data_wont_use(starpu_data_handle_t handle);
                subroutine fstarpu_data_wont_use (dh) bind(c,name="starpu_data_wont_use")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_data_wont_use

                ! unsigned starpu_worker_get_memory_node(unsigned workerid);
                function fstarpu_worker_get_memory_node(id) bind(C,name="starpu_worker_get_memory_node")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_worker_get_memory_node
                        integer(c_int), value, intent(in) :: id
                end function fstarpu_worker_get_memory_node

                ! unsigned starpu_memory_nodes_get_count(void);
                function fstarpu_memory_nodes_get_count() bind(C,name="starpu_memory_nodes_get_count")
                        use iso_c_binding, only: c_int
                        integer(c_int)              :: fstarpu_memory_nodes_get_count
                end function fstarpu_memory_nodes_get_count

                ! enum starpu_node_kind starpu_node_get_kind(unsigned node);
                ! void starpu_data_set_wt_mask(starpu_data_handle_t handle, uint32_t wt_mask);
                ! void starpu_data_set_sequential_consistency_flag(starpu_data_handle_t handle, unsigned flag);
                ! unsigned starpu_data_get_sequential_consistency_flag(starpu_data_handle_t handle);
                ! unsigned starpu_data_get_default_sequential_consistency_flag(void);
                ! void starpu_data_set_default_sequential_consistency_flag(unsigned flag);
                ! void starpu_data_query_status(starpu_data_handle_t handle, int memory_node, int *is_allocated, int *is_valid, int *is_requested);

                ! void starpu_data_set_reduction_methods(starpu_data_handle_t handle, struct starpu_codelet *redux_cl, struct starpu_codelet *init_cl);
                subroutine fstarpu_data_set_reduction_methods (dh,redux_cl,init_cl) bind(C,name="starpu_data_set_reduction_methods")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: redux_cl
                        type(c_ptr), value, intent(in) :: init_cl
                end subroutine fstarpu_data_set_reduction_methods

                ! struct starpu_data_interface_ops* starpu_data_get_interface_ops(starpu_data_handle_t handle);

                ! unsigned starpu_data_test_if_allocated_on_node(starpu_data_handle_t handle, unsigned memory_node);
                function fstarpu_data_test_if_allocated_on_node(dh,mem_node) bind(C,name="starpu_data_test_if_allocated_on_node")
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int)              :: fstarpu_data_test_if_allocated_on_node
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: mem_node
                end function fstarpu_data_test_if_allocated_on_node

                ! void starpu_memchunk_tidy(unsigned memory_node);
                subroutine fstarpu_memchunk_tidy (mem_node) bind(c,name="starpu_memchunk_tidy")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: mem_node
                end subroutine fstarpu_memchunk_tidy

                ! == starpu_task_util.h ==
                subroutine fstarpu_insert_task(arglist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), dimension(:), intent(in) :: arglist
                end subroutine fstarpu_insert_task

                subroutine fstarpu_unpack_arg(cl_arg,bufferlist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl_arg
                        type(c_ptr), dimension(:), intent(in) :: bufferlist
                end subroutine fstarpu_unpack_arg

                ! == starpu_sched_ctx.h ==

                ! starpu_sched_ctx_create: see fstarpu_sched_ctx_create
                function fstarpu_sched_ctx_create(workers_array,nworkers,ctx_name) bind(C)
                        use iso_c_binding, only: c_int, c_char
                        integer(c_int) :: fstarpu_sched_ctx_create
                        integer(c_int), intent(in) :: workers_array(*)
                        integer(c_int), value, intent(in) :: nworkers
                        character(c_char), intent(in) :: ctx_name
                end function fstarpu_sched_ctx_create

                ! unsigned starpu_sched_ctx_create_inside_interval(const char *policy_name, const char *sched_ctx_name, int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus, unsigned allow_overlap);
                ! void starpu_sched_ctx_register_close_callback(unsigned sched_ctx_id, void (*close_callback)(unsigned sched_ctx_id, void* args), void *args);
                ! void starpu_sched_ctx_add_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);
                ! void starpu_sched_ctx_remove_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);
                ! starpu_sched_ctx_display_workers: see fstarpu_sched_ctx_display_workers
                subroutine fstarpu_sched_ctx_display_workers (ctx) bind(C)
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: ctx
                end subroutine fstarpu_sched_ctx_display_workers

                ! void starpu_sched_ctx_delete(unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_delete (ctx) bind(C,name="starpu_sched_ctx_delete")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: ctx
                end subroutine fstarpu_sched_ctx_delete

                ! void starpu_sched_ctx_set_inheritor(unsigned sched_ctx_id, unsigned inheritor);
                ! unsigned starpu_sched_ctx_get_inheritor(unsigned sched_ctx_id);
                ! unsigned starpu_sched_ctx_get_hierarchy_level(unsigned sched_ctx_id);

                ! void starpu_sched_ctx_set_context(unsigned *sched_ctx_id);
                subroutine fstarpu_sched_ctx_set_context (ctx_ptr) bind(C,name="starpu_sched_ctx_set_context")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: ctx_ptr
                end subroutine fstarpu_sched_ctx_set_context

                ! unsigned starpu_sched_ctx_get_context(void);
                function fstarpu_sched_ctx_get_context () bind(C,name="starpu_sched_ctx_get_context")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_context
                end function fstarpu_sched_ctx_get_context


                ! == starpu_fxt.h ==

                ! void starpu_fxt_options_init(struct starpu_fxt_options *options);
                ! void starpu_fxt_generate_trace(struct starpu_fxt_options *options);

                ! void starpu_fxt_autostart_profiling(int autostart);
                subroutine fstarpu_fxt_autostart_profiling (autostart) bind(c,name="starpu_fxt_autostart_profiling")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: autostart
                end subroutine fstarpu_fxt_autostart_profiling

                ! void starpu_fxt_start_profiling(void);
                subroutine fstarpu_fxt_start_profiling () bind(c,name="starpu_fxt_start_profiling")
                        use iso_c_binding
                end subroutine fstarpu_fxt_start_profiling

                ! void starpu_fxt_stop_profiling(void);
                subroutine fstarpu_fxt_stop_profiling () bind(c,name="starpu_fxt_stop_profiling")
                        use iso_c_binding
                end subroutine fstarpu_fxt_stop_profiling

                ! void starpu_fxt_write_data_trace(char *filename_in);
                subroutine fstarpu_fxt_write_data_trace (filename) bind(c,name="starpu_fxt_write_data_trace")
                        use iso_c_binding, only: c_char
                        character(c_char), intent(in) :: filename
                end subroutine fstarpu_fxt_write_data_trace

                ! void starpu_fxt_trace_user_event(unsigned long code);
                subroutine fstarpu_trace_user_event (code) bind(c,name="starpu_trace_user_event")
                        use iso_c_binding, only: c_long
                        integer(c_long), value, intent(in) :: code
                end subroutine fstarpu_trace_user_event

        end interface

        contains
                function ip_to_p(i) bind(C)
                        use iso_c_binding, only: c_ptr,c_intptr_t,C_NULL_PTR
                        type(c_ptr) :: ip_to_p
                        integer(c_intptr_t), value, intent(in) :: i
                        ip_to_p = transfer(i,C_NULL_PTR)
                end function ip_to_p

                function sz_to_p(sz) bind(C)
                        use iso_c_binding, only: c_ptr,c_size_t,c_intptr_t
                        type(c_ptr) :: sz_to_p
                        integer(c_size_t), value, intent(in) :: sz
                        sz_to_p = ip_to_p(int(sz,kind=c_intptr_t))
                end function sz_to_p

                function fstarpu_init (conf) bind(C)
                        use iso_c_binding
                        integer(c_int) :: fstarpu_init
                        type(c_ptr), value, intent(in) :: conf

                        integer(4) :: FSTARPU_SZ_INT4_dummy
                        integer(8) :: FSTARPU_SZ_INT8_dummy
                        real(4) :: FSTARPU_SZ_REAL4_dummy
                        real(8) :: FSTARPU_SZ_REAL8_dummy

                        ! Note: Referencing global C constants from Fortran has
                        ! been found unreliable on some architectures, notably
                        ! on Darwin. The get_integer/get_pointer_constant
                        ! scheme is a workaround to that issue.

                        interface
                                ! These functions are not exported to the end user
                                function fstarpu_get_constant(s) bind(C)
                                        use iso_c_binding, only: c_ptr,c_char
                                        type(c_ptr) :: fstarpu_get_constant ! C function returns an intptr_t
                                        character(kind=c_char) :: s
                                end function fstarpu_get_constant

                                function fstarpu_init_internal (conf) bind(C,name="starpu_init")
                                        use iso_c_binding, only: c_ptr,c_int
                                        integer(c_int) :: fstarpu_init_internal
                                        type(c_ptr), value :: conf
                                end function fstarpu_init_internal

                        end interface

                        ! Initialize Fortran constants from C peers
                        FSTARPU_R       = fstarpu_get_constant(C_CHAR_"FSTARPU_R"//C_NULL_CHAR)
                        FSTARPU_W       = fstarpu_get_constant(C_CHAR_"FSTARPU_W"//C_NULL_CHAR)
                        FSTARPU_RW      = fstarpu_get_constant(C_CHAR_"FSTARPU_RW"//C_NULL_CHAR)
                        FSTARPU_SCRATCH = fstarpu_get_constant(C_CHAR_"FSTARPU_SCRATCH"//C_NULL_CHAR)
                        FSTARPU_REDUX   = fstarpu_get_constant(C_CHAR_"FSTARPU_REDUX"//C_NULL_CHAR)
                        FSTARPU_DATA    = fstarpu_get_constant(C_CHAR_"FSTARPU_DATA"//C_NULL_CHAR)
                        FSTARPU_VALUE   = fstarpu_get_constant(C_CHAR_"FSTARPU_VALUE"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX   = fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX"//C_NULL_CHAR)
                        ! Initialize size constants as 'c_ptr'
                        FSTARPU_SZ_INT4         = sz_to_p(c_sizeof(FSTARPU_SZ_INT4_dummy))
                        FSTARPU_SZ_INT8         = sz_to_p(c_sizeof(FSTARPU_SZ_INT8_dummy))
                        FSTARPU_SZ_REAL4        = sz_to_p(c_sizeof(FSTARPU_SZ_REAL4_dummy))
                        FSTARPU_SZ_REAL8        = sz_to_p(c_sizeof(FSTARPU_SZ_REAL8_dummy))
                        ! Initialize StarPU
                        if (c_associated(conf)) then 
                                fstarpu_init = fstarpu_init_internal(conf)
                        else
                                fstarpu_init = fstarpu_init_internal(C_NULL_PTR)
                        end if
                end function fstarpu_init

                function fstarpu_csizet_to_cptr(i) bind(C)
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_csizet_to_cptr
                        integer(c_size_t) :: i
                        fstarpu_csizet_to_cptr = transfer(int(i,kind=c_intptr_t),C_NULL_PTR)
                end function fstarpu_csizet_to_cptr

                function fstarpu_int_to_cptr(i) bind(C)
                        use iso_c_binding
                        type(c_ptr) :: fstarpu_int_to_cptr
                        integer :: i
                        fstarpu_int_to_cptr = transfer(int(i,kind=c_intptr_t),C_NULL_PTR)
                end function fstarpu_int_to_cptr
end module fstarpu_mod
