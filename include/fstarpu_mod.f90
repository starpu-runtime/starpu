! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
!
module fstarpu_mod
        use iso_c_binding
        implicit none

        ! Note: Constants truly are intptr_t, but are declared as c_ptr to be
        ! readily usable in c_ptr arrays to mimic variadic functions.
        ! Note: Bitwise or operator is provided by the .ior. overloaded operator
        type(c_ptr), bind(C) :: FSTARPU_R
        type(c_ptr), bind(C) :: FSTARPU_W
        type(c_ptr), bind(C) :: FSTARPU_RW
        type(c_ptr), bind(C) :: FSTARPU_SCRATCH
        type(c_ptr), bind(C) :: FSTARPU_REDUX
        type(c_ptr), bind(C) :: FSTARPU_COMMUTE
        type(c_ptr), bind(C) :: FSTARPU_SSEND
        type(c_ptr), bind(C) :: FSTARPU_LOCALITY

        type(c_ptr), bind(C) :: FSTARPU_DATA_ARRAY
        type(c_ptr), bind(C) :: FSTARPU_DATA_MODE_ARRAY
        type(c_ptr), bind(C) :: FSTARPU_CL_ARGS
        type(c_ptr), bind(C) :: FSTARPU_CL_ARGS_NFREE
        type(c_ptr), bind(C) :: FSTARPU_TASK_DEPS_ARRAY
        type(c_ptr), bind(C) :: FSTARPU_CALLBACK
        type(c_ptr), bind(C) :: FSTARPU_CALLBACK_WITH_ARG
        type(c_ptr), bind(C) :: FSTARPU_CALLBACK_ARG
        type(c_ptr), bind(C) :: FSTARPU_PROLOGUE_CALLBACK
        type(c_ptr), bind(C) :: FSTARPU_PROLOGUE_CALLBACK_ARG
        type(c_ptr), bind(C) :: FSTARPU_PROLOGUE_CALLBACK_POP
        type(c_ptr), bind(C) :: FSTARPU_PROLOGUE_CALLBACK_POP_ARG
        type(c_ptr), bind(C) :: FSTARPU_PRIORITY
        type(c_ptr), bind(C) :: FSTARPU_EXECUTE_ON_NODE
        type(c_ptr), bind(C) :: FSTARPU_EXECUTE_ON_DATA
        type(c_ptr), bind(C) :: FSTARPU_EXECUTE_ON_WORKER
        type(c_ptr), bind(C) :: FSTARPU_WORKER_ORDER
        type(c_ptr), bind(C) :: FSTARPU_EXECUTE_WHERE
        type(c_ptr), bind(C) :: FSTARPU_HYPERVISOR_TAG
        type(c_ptr), bind(C) :: FSTARPU_POSSIBLY_PARALLEL
        type(c_ptr), bind(C) :: FSTARPU_FLOPS
        type(c_ptr), bind(C) :: FSTARPU_TAG
        type(c_ptr), bind(C) :: FSTARPU_TAG_ONLY
        type(c_ptr), bind(C) :: FSTARPU_NAME
        type(c_ptr), bind(C) :: FSTARPU_TASK_COLOR
        type(c_ptr), bind(C) :: FSTARPU_TASK_SYNCHRONOUS
        type(c_ptr), bind(C) :: FSTARPU_HANDLES_SEQUENTIAL_CONSISTENCY
        type(c_ptr), bind(C) :: FSTARPU_TASK_END_DEP
        type(c_ptr), bind(C) :: FSTARPU_NODE_SELECTION_POLICY
        type(c_ptr), bind(C) :: FSTARPU_TASK_SCHED_DATA

        type(c_ptr), bind(C) :: FSTARPU_VALUE
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX

        type(c_ptr), bind(C) :: FSTARPU_CPU_WORKER
        type(c_ptr), bind(C) :: FSTARPU_CUDA_WORKER
        type(c_ptr), bind(C) :: FSTARPU_OPENCL_WORKER
        type(c_ptr), bind(C) :: FSTARPU_MIC_WORKER
        type(c_ptr), bind(C) :: FSTARPU_ANY_WORKER

        integer(c_int), bind(C) :: FSTARPU_NMAXBUFS

        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_POLICY_NAME
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_POLICY_STRUCT
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_POLICY_MIN_PRIO
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_POLICY_MAX_PRIO
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_HIERARCHY_LEVEL
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_NESTED
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_AWAKE_WORKERS
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_POLICY_INIT
        type(c_ptr), bind(C) :: FSTARPU_SCHED_CTX_USER_DATA

        type(c_ptr), bind(C) :: FSTARPU_NOWHERE
        type(c_ptr), bind(C) :: FSTARPU_CPU
        type(c_ptr), bind(C) :: FSTARPU_CUDA
        type(c_ptr), bind(C) :: FSTARPU_OPENCL
        type(c_ptr), bind(C) :: FSTARPU_MIC

        type(c_ptr), bind(C) :: FSTARPU_CODELET_SIMGRID_EXECUTE
        type(c_ptr), bind(C) :: FSTARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT
        type(c_ptr), bind(C) :: FSTARPU_CUDA_ASYNC
        type(c_ptr), bind(C) :: FSTARPU_OPENCL_ASYNC

        ! (some) portable iso_c_binding types
        type(c_ptr), bind(C) :: FSTARPU_SZ_C_DOUBLE
        type(c_ptr), bind(C) :: FSTARPU_SZ_C_FLOAT
        type(c_ptr), bind(C) :: FSTARPU_SZ_C_CHAR
        type(c_ptr), bind(C) :: FSTARPU_SZ_C_INT
        type(c_ptr), bind(C) :: FSTARPU_SZ_C_INTPTR_T
        type(c_ptr), bind(C) :: FSTARPU_SZ_C_PTR
        type(c_ptr), bind(C) :: FSTARPU_SZ_C_SIZE_T

        ! (some) native Fortran types
        type(c_ptr), bind(C) :: FSTARPU_SZ_CHARACTER

        type(c_ptr), bind(C) :: FSTARPU_SZ_INTEGER
        type(c_ptr), bind(C) :: FSTARPU_SZ_INT4
        type(c_ptr), bind(C) :: FSTARPU_SZ_INT8

        type(c_ptr), bind(C) :: FSTARPU_SZ_REAL
        type(c_ptr), bind(C) :: FSTARPU_SZ_REAL4
        type(c_ptr), bind(C) :: FSTARPU_SZ_REAL8

        type(c_ptr), bind(C) :: FSTARPU_SZ_DOUBLE_PRECISION

        type(c_ptr), bind(C) :: FSTARPU_SZ_COMPLEX
        type(c_ptr), bind(C) :: FSTARPU_SZ_COMPLEX4
        type(c_ptr), bind(C) :: FSTARPU_SZ_COMPLEX8

        interface operator (.ior.)
                procedure or_cptrs
        end interface operator (.ior.)

        interface
                ! == starpu.h ==

                ! void starpu_conf_init(struct starpu_conf *conf);
                subroutine fstarpu_conf_init (conf) bind(C,name="starpu_conf_init")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: conf
                end subroutine fstarpu_conf_init

                function fstarpu_conf_allocate () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_conf_allocate
                end function fstarpu_conf_allocate

                subroutine fstarpu_conf_free (conf) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: conf
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
                subroutine fstarpu_topology_print () bind(C)
                end subroutine fstarpu_topology_print

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
                function fstarpu_worker_get_type(id) bind(C)
                        use iso_c_binding, only: c_int, c_ptr
                        type(c_ptr)              :: fstarpu_worker_get_type ! C function returns c_intptr_t
                        integer(c_int),value,intent(in) :: id
                        end function fstarpu_worker_get_type

                ! int starpu_worker_get_count_by_type(enum starpu_worker_archtype type);
                function fstarpu_worker_get_count_by_type(typeid) bind(C)
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int)              :: fstarpu_worker_get_count_by_type
                        type(c_ptr),value,intent(in) :: typeid ! c_intptr_t expected by C func
                end function fstarpu_worker_get_count_by_type

                ! int starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize);
                function fstarpu_worker_get_ids_by_type(typeid, workerids, maxsize) bind(C)
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int)              :: fstarpu_worker_get_ids_by_type
                        type(c_ptr),value,intent(in) :: typeid ! c_intptr_t expected by C func
                        integer(c_int),intent(out) :: workerids(*)
                        integer(c_int),value,intent(in) :: maxsize
                end function fstarpu_worker_get_ids_by_type

                ! int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num);
                function fstarpu_worker_get_by_type(typeid, num) bind(C)
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int)              :: fstarpu_worker_get_by_type
                        type(c_ptr),value,intent(in) :: typeid ! c_intptr_t expected by C func
                        integer(c_int),value,intent(in) :: num
                end function fstarpu_worker_get_by_type

                ! int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid);
                function fstarpu_worker_get_by_devid(typeid, devid) bind(C)
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int)              :: fstarpu_worker_get_by_type
                        type(c_ptr),value,intent(in) :: typeid ! c_intptr_t expected by C func
                        integer(c_int),value,intent(in) :: devid
                end function fstarpu_worker_get_by_devid

                ! void starpu_worker_get_name(int id, char *dst, size_t maxlen);
                subroutine fstarpu_worker_get_name(id, dst, maxlen) bind(C,name="starpu_worker_get_name")
                        use iso_c_binding, only: c_int, c_char, c_size_t
                        integer(c_int),value,intent(in) :: id
                        character(c_char),intent(out) :: dst(*)
                        integer(c_size_t),value,intent(in) :: maxlen
                end subroutine fstarpu_worker_get_name


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
                subroutine fstarpu_worker_get_type_as_string(typeid,dst,maxlen) bind(C)
                        use iso_c_binding, only: c_ptr, c_char, c_size_t
                        type(c_ptr),value,intent(in) :: typeid ! c_intptr_t expected by C func
                        character(c_char),intent(out) :: dst(*)
                        integer(c_size_t),value,intent(in) :: maxlen
                end subroutine fstarpu_worker_get_type_as_string

                ! int starpu_bindid_get_workerids(int bindid, int **workerids);

                ! == starpu_task.h ==

                ! void starpu_tag_declare_deps(starpu_tag_t id, unsigned ndeps, ...);

                ! void starpu_tag_declare_deps_array(starpu_tag_t id, unsigned ndeps, starpu_tag_t *array);
                subroutine fstarpu_tag_declare_deps_array(id,ndeps,tag_array) bind(C,name="starpu_tag_declare_deps_array")
                        use iso_c_binding, only: c_int, c_long_long
                        integer(c_int), value, intent(in) :: id
                        integer(c_int), value, intent(in) :: ndeps
                        integer(c_long_long), intent(in) :: tag_array(*)
                end subroutine fstarpu_tag_declare_deps_array

                ! void starpu_task_declare_deps_array(struct starpu_task *task, unsigned ndeps, struct starpu_task *task_array[]);
                subroutine fstarpu_task_declare_deps_array(task,ndeps,task_array) bind(C,name="starpu_task_declare_deps_array")
                        use iso_c_binding, only: c_int, c_ptr
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: ndeps
                        type(c_ptr), intent(in) :: task_array(*)
                end subroutine fstarpu_task_declare_deps_array

                ! int starpu_tag_wait(starpu_tag_t id);
                function fstarpu_tag_wait(id) bind(C,name="starpu_tag_wait")
                        use iso_c_binding, only: c_int, c_long_long
                        integer(c_int) :: fstarpu_tag_wait
                        integer(c_long_long), value, intent(in) :: id
                end function fstarpu_tag_wait

                ! int starpu_tag_wait_array(unsigned ntags, starpu_tag_t *id);
                function fstarpu_tag_wait_array(ntags,tag_array) bind(C,name="starpu_tag_wait_array")
                        use iso_c_binding, only: c_int, c_long_long
                        integer(c_int) :: fstarpu_tag_wait_array
                        integer(c_int), value, intent(in) :: ntags
                        integer(c_long_long), intent(in) :: tag_array(*)
                end function fstarpu_tag_wait_array

                ! void starpu_tag_notify_from_apps(starpu_tag_t id);
                subroutine fstarpu_tag_notify_from_apps(id) bind(C,name="starpu_tag_notify_from_apps")
                        use iso_c_binding, only: c_long_long
                        integer(c_long_long), value, intent(in) :: id
                end subroutine fstarpu_tag_notify_from_apps

                ! void starpu_tag_restart(starpu_tag_t id);
                subroutine fstarpu_tag_restart(id) bind(C,name="starpu_tag_restart")
                        use iso_c_binding, only: c_long_long
                        integer(c_long_long), value, intent(in) :: id
                end subroutine fstarpu_tag_restart

                ! void starpu_tag_remove(starpu_tag_t id);
                subroutine fstarpu_tag_remove(id) bind(C,name="starpu_tag_remove")
                        use iso_c_binding, only: c_long_long
                        integer(c_long_long), value, intent(in) :: id
                end subroutine fstarpu_tag_remove

                ! struct starpu_task *starpu_tag_get_task(starpu_tag_t id);
                function fstarpu_tag_get_task(id) bind(C,name="starpu_tag_get_task")
                        use iso_c_binding, only: c_ptr, c_long_long
                        type(c_ptr) :: fstarpu_tag_get_task
                        integer(c_long_long), value, intent(in) :: id
                end function fstarpu_tag_get_task


                ! void starpu_task_init(struct starpu_task *task);
                subroutine fstarpu_task_init (task) bind(C,name="starpu_task_init")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: task
                end subroutine fstarpu_task_init

                ! void starpu_task_clean(struct starpu_task *task);
                subroutine fstarpu_task_clean (task) bind(C,name="starpu_task_clean")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: task
                end subroutine fstarpu_task_clean

                ! struct starpu_task *starpu_task_create(void) STARPU_ATTRIBUTE_MALLOC;
                function fstarpu_task_create () bind(C,name="starpu_task_create")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_task_create
                end function fstarpu_task_create

                ! void starpu_task_destroy(struct starpu_task *task);
                subroutine fstarpu_task_destroy (task) bind(C,name="starpu_task_destroy")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: task
                end subroutine fstarpu_task_destroy

                ! int starpu_task_submit(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;
                function fstarpu_task_submit (task) bind(C,name="starpu_task_submit")
                        use iso_c_binding, only: c_int,c_ptr
                        integer(c_int) :: fstarpu_task_submit
                        type(c_ptr), value, intent(in) :: task
                end function fstarpu_task_submit

                ! int starpu_task_submit_to_ctx(struct starpu_task *task, unsigned sched_ctx_id);
                function fstarpu_task_submit_to_ctx (task,sched_ctx_id) bind(C,name="starpu_task_submit_to_ctx")
                        use iso_c_binding, only: c_int,c_ptr
                        integer(c_int) :: fstarpu_task_submit_to_ctx
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_task_submit_to_ctx

                ! int starpu_task_finished(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;
                function fstarpu_task_finished (task) bind(C,name="starpu_task_finished")
                        use iso_c_binding, only: c_int,c_ptr
                        integer(c_int) :: fstarpu_task_finished
                        type(c_ptr), value, intent(in) :: task
                end function fstarpu_task_finished

                ! int starpu_task_wait(struct starpu_task *task) STARPU_WARN_UNUSED_RESULT;
                function fstarpu_task_wait (task) bind(C,name="starpu_task_wait")
                        use iso_c_binding, only: c_int,c_ptr
                        integer(c_int) :: fstarpu_task_wait
                        type(c_ptr), value, intent(in) :: task
                end function fstarpu_task_wait

                ! int starpu_task_wait_array(struct starpu_task **tasks, unsigned nb_tasks) STARPU_WARN_UNUSED_RESULT;
                function fstarpu_task_wait_array(task_array,ntasks) bind(C,name="starpu_task_wait_array")
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int) :: fstarpu_task_wait_array
                        integer(c_int), value, intent(in) :: ntasks
                        type(c_ptr), intent(in) :: task_array
                end function fstarpu_task_wait_array


                ! int starpu_task_wait_for_all(void);
                subroutine fstarpu_task_wait_for_all () bind(C,name="starpu_task_wait_for_all")
                end subroutine fstarpu_task_wait_for_all

                ! int starpu_task_wait_for_n_submitted(unsigned n);
                subroutine fstarpu_task_wait_for_n_submitted (n) bind(C,name="starpu_task_wait_for_n_submitted")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: n
                end subroutine fstarpu_task_wait_for_n_submitted

                ! int starpu_task_wait_for_all_in_ctx(unsigned sched_ctx_id);
                subroutine fstarpu_task_wait_for_all_in_ctx (ctx) bind(C,name="starpu_task_wait_for_all_in_ctx")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: ctx
                end subroutine fstarpu_task_wait_for_all_in_ctx

                ! int starpu_task_wait_for_n_submitted_in_ctx(unsigned sched_ctx_id, unsigned n);
                subroutine fstarpu_task_wait_for_n_submitted_in_ctx (ctx,n) bind(C,name="starpu_task_wait_for_n_submitted_in_ctx")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: ctx
                        integer(c_int), value, intent(in) :: n
                end subroutine fstarpu_task_wait_for_n_submitted_in_ctx

                ! int starpu_task_wait_for_no_ready(void);
                function fstarpu_task_wait_for_no_ready () bind(C,name="starpu_task_wait_for_no_ready")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_task_wait_for_no_ready
                end function fstarpu_task_wait_for_no_ready

                ! int starpu_task_nready(void);
                function fstarpu_task_nready () bind(C,name="starpu_task_nready")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_task_nready
                end function fstarpu_task_nready

                ! int starpu_task_nsubmitted(void);
                function fstarpu_task_nsubmitted () bind(C,name="starpu_task_nsubmitted")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_task_nsubmitted
                end function fstarpu_task_nsubmitted

                ! void starpu_do_schedule(void);
                subroutine fstarpu_do_schedule () bind(C,name="starpu_do_schedule")
                end subroutine fstarpu_do_schedule

                ! starpu_codelet_init
                subroutine fstarpu_codelet_init (codelet) bind(C,name="starpu_codelet_init")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: codelet
                end subroutine fstarpu_codelet_init

                ! starpu_codelet_display_stats
                subroutine fstarpu_codelet_display_stats (codelet) bind(C,name="starpu_codelet_display_stats")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: codelet
                end subroutine fstarpu_codelet_display_stats


                ! struct starpu_task *starpu_task_get_current(void);
                function fstarpu_task_get_current () bind(C,name="starpu_task_get_current")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_task_get_current
                end function fstarpu_task_get_current

                ! void starpu_parallel_task_barrier_init(struct starpu_task *task, int workerid);
                subroutine fstarpu_parallel_task_barrier_init_init (task,id) &
                                bind(C,name="starpu_parallel_task_barrier_init_init")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: id
                end subroutine fstarpu_parallel_task_barrier_init_init

                ! void starpu_parallel_task_barrier_init_n(struct starpu_task *task, int worker_size);
                subroutine fstarpu_parallel_task_barrier_init_n_init_n (task,sz) &
                                bind(C,name="starpu_parallel_task_barrier_init_n_init_n")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: sz
                end subroutine fstarpu_parallel_task_barrier_init_n_init_n

                ! struct starpu_task *starpu_task_dup(struct starpu_task *task);
                function fstarpu_task_dup (task) bind(C,name="starpu_task_dup")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_task_dup
                        type(c_ptr), value, intent(in) :: task
                end function fstarpu_task_dup

                ! void starpu_task_set_implementation(struct starpu_task *task, unsigned impl);
                subroutine fstarpu_task_set_implementation (task,impl) &
                                bind(C,name="starpu_task_set_implementation")
                        use iso_c_binding, only: c_ptr,c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: impl
                end subroutine fstarpu_task_set_implementation

                ! unsigned starpu_task_get_implementation(struct starpu_task *task);
                function fstarpu_task_get_implementation (task) &
                                bind(C,name="starpu_task_get_implementation")
                        use iso_c_binding, only: c_ptr,c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int) :: fstarpu_task_get_implementation
                end function fstarpu_task_get_implementation

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

                subroutine fstarpu_codelet_add_cuda_flags (cl, flags) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_ptr), value, intent(in) :: flags ! C function expects an intptr_t
                end subroutine fstarpu_codelet_add_cuda_flags

                subroutine fstarpu_codelet_add_opencl_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_opencl_func

                subroutine fstarpu_codelet_add_opencl_flags (cl, flags) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_ptr), value, intent(in) :: flags ! C function expects an intptr_t
                end subroutine fstarpu_codelet_add_opencl_flags

                subroutine fstarpu_codelet_add_mic_func (cl, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_codelet_add_mic_func

                subroutine fstarpu_codelet_add_buffer (cl, mode) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_ptr), value, intent(in) :: mode ! C function expects an intptr_t
                end subroutine fstarpu_codelet_add_buffer

                subroutine fstarpu_codelet_set_variable_nbuffers (cl) bind(C)
                        use iso_c_binding, only: c_ptr,c_int
                        type(c_ptr), value, intent(in) :: cl
                end subroutine fstarpu_codelet_set_variable_nbuffers

                subroutine fstarpu_codelet_set_nbuffers (cl, nbuffers) bind(C)
                        use iso_c_binding, only: c_ptr,c_int
                        type(c_ptr), value, intent(in) :: cl
                        integer(c_int), value, intent(in) :: nbuffers
                end subroutine fstarpu_codelet_set_nbuffers

                subroutine fstarpu_codelet_set_flags (cl, flags) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_ptr), value, intent(in) :: flags ! C function expects an intptr_t
                end subroutine fstarpu_codelet_set_flags

                subroutine fstarpu_codelet_set_where (cl, where) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl
                        type(c_ptr), value, intent(in) :: where ! C function expects an intptr_t
                end subroutine fstarpu_codelet_set_where

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

                ! void *starpu_data_pointer_is_inside(starpu_data_handle_t handle, unsigned node, void *ptr);
                function fstarpu_data_pointer_is_inside (dh,node,ptr) bind(C,name="starpu_data_pointer_is_inside")
                        use iso_c_binding, only: c_ptr, c_int, c_ptr
                        integer(c_int) :: fstarpu_data_pointer_is_inside
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                        type(c_ptr), value, intent(in) :: ptr
                end function fstarpu_data_pointer_is_inside

                ! void *starpu_data_get_local_ptr(starpu_data_handle_t handle);
                function fstarpu_data_get_local_ptr (dh) bind(C,name="starpu_data_get_local_ptr")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_data_get_local_ptr
                        type(c_ptr), value, intent(in) :: dh
                end function fstarpu_data_get_local_ptr

                ! void *starpu_data_get_interface_on_node(starpu_data_handle_t handle, unsigned memory_node);

                ! == starpu_data_interface.h: block ==

                ! void starpu_block_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, uint32_t ldy, uint32_t ldz, uint32_t nx, uint32_t ny, uint32_t nz, size_t elemsize);
                subroutine fstarpu_block_data_register(dh, home_node, ptr, ldy, ldz, nx, ny, nz, elt_size) &
                                bind(C,name="starpu_block_data_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: home_node
                        type(c_ptr), value, intent(in) :: ptr
                        integer(c_int), value, intent(in) :: ldy
                        integer(c_int), value, intent(in) :: ldz
                        integer(c_int), value, intent(in) :: nx
                        integer(c_int), value, intent(in) :: ny
                        integer(c_int), value, intent(in) :: nz
                        integer(c_size_t), value, intent(in) :: elt_size
                end subroutine fstarpu_block_data_register

                ! void starpu_block_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t ldy, uint32_t ldz);
                subroutine fstarpu_block_ptr_register(dh, node, ptr, dev_handle, offset, ldy, ldz) &
                                bind(C,name="starpu_block_ptr_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: node
                        type(c_ptr), value, intent(in) :: ptr
                        type(c_ptr), value, intent(in) :: dev_handle
                        integer(c_size_t), value, intent(in) :: offset
                        integer(c_int), value, intent(in) :: ldy
                        integer(c_int), value, intent(in) :: ldz
                end subroutine fstarpu_block_ptr_register

                function fstarpu_block_get_ptr(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_block_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_block_get_ptr

                function fstarpu_block_get_ldy(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_block_get_ldy
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_block_get_ldy

                function fstarpu_block_get_ldz(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_block_get_ldz
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_block_get_ldz

                function fstarpu_block_get_nx(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_block_get_nx
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_block_get_nx

                function fstarpu_block_get_ny(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_block_get_ny
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_block_get_ny

                function fstarpu_block_get_nz(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_block_get_nz
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_block_get_nz

                ! == starpu_data_interface.h: matrix ==

                ! void starpu_matrix_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, uint32_t ld, uint32_t nx, uint32_t ny, size_t elemsize);
                subroutine fstarpu_matrix_data_register(dh, home_node, ptr, ld, nx, ny, elt_size) &
                                bind(C,name="starpu_matrix_data_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: home_node
                        type(c_ptr), value, intent(in) :: ptr
                        integer(c_int), value, intent(in) :: ld
                        integer(c_int), value, intent(in) :: nx
                        integer(c_int), value, intent(in) :: ny
                        integer(c_size_t), value, intent(in) :: elt_size
                end subroutine fstarpu_matrix_data_register

                ! void starpu_matrix_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset, uint32_t ld);
                subroutine fstarpu_matrix_ptr_register(dh, node, ptr, dev_handle, offset, ld) &
                                bind(C,name="starpu_matrix_ptr_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: node
                        type(c_ptr), value, intent(in) :: ptr
                        type(c_ptr), value, intent(in) :: dev_handle
                        integer(c_size_t), value, intent(in) :: offset
                        integer(c_int), value, intent(in) :: ld
                end subroutine fstarpu_matrix_ptr_register

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

                function fstarpu_matrix_get_nx(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_nx
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_nx

                function fstarpu_matrix_get_ny(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int) :: fstarpu_matrix_get_ny
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_matrix_get_ny

                ! == starpu_data_interface.h: vector ==

                ! void starpu_vector_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, uint32_t nx, size_t elemsize);
                subroutine fstarpu_vector_data_register(dh, home_node, ptr,nx, elt_size) &
                                bind(C,name="starpu_vector_data_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: home_node
                        type(c_ptr), value, intent(in) :: ptr
                        integer(c_int), value, intent(in) :: nx
                        integer(c_size_t), value, intent(in) :: elt_size
                end subroutine fstarpu_vector_data_register

                ! void starpu_vector_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset);
                subroutine fstarpu_vector_ptr_register(dh, node, ptr, dev_handle, offset, ld) &
                                bind(C,name="starpu_vector_ptr_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: node
                        type(c_ptr), value, intent(in) :: ptr
                        type(c_ptr), value, intent(in) :: dev_handle
                        integer(c_size_t), value, intent(in) :: offset
                end subroutine fstarpu_vector_ptr_register


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

                ! void starpu_variable_data_register(starpu_data_handle_t *handle, unsigned home_node, uintptr_t ptr, size_t size);
                subroutine fstarpu_variable_data_register(dh, home_node, ptr, elt_size) &
                                bind(C,name="starpu_variable_data_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: home_node
                        type(c_ptr), value, intent(in) :: ptr
                        integer(c_size_t), value, intent(in) :: elt_size
                end subroutine fstarpu_variable_data_register

                ! void starpu_variable_ptr_register(starpu_data_handle_t handle, unsigned node, uintptr_t ptr, uintptr_t dev_handle, size_t offset);
                subroutine fstarpu_variable_ptr_register(dh, node, ptr, dev_handle, offset, ld) &
                                bind(C,name="starpu_variable_ptr_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                        integer(c_int), value, intent(in) :: node
                        type(c_ptr), value, intent(in) :: ptr
                        type(c_ptr), value, intent(in) :: dev_handle
                        integer(c_size_t), value, intent(in) :: offset
                end subroutine fstarpu_variable_ptr_register

                function fstarpu_variable_get_ptr(buffers, i) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_variable_get_ptr
                        type(c_ptr), value, intent(in) :: buffers
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_variable_get_ptr

                ! == starpu_data_interface.h: void ==

                ! void starpu_void_data_register(starpu_data_handle_t *handle);
                subroutine fstarpu_void_data_register(dh) &
                                bind(C,name="starpu_void_data_register")
                        use iso_c_binding, only: c_ptr, c_int, c_size_t
                        type(c_ptr), intent(out) :: dh
                end subroutine fstarpu_void_data_register

                ! == starpu_data_filter.h ==

                function fstarpu_data_filter_allocate () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_data_filter_allocate
                end function fstarpu_data_filter_allocate

                subroutine fstarpu_data_filter_free (filter) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: filter
                end subroutine fstarpu_data_filter_free

                ! Note: use fstarpu_df_alloc_ prefix instead of fstarpu_data_filter_allocate_
                ! to fit within the Fortran id length limit */
                function fstarpu_df_alloc_bcsr_filter_canonical_block () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_bcsr_filter_canonical_block
                end function fstarpu_df_alloc_bcsr_filter_canonical_block

                function fstarpu_df_alloc_csr_filter_vertical_block () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_csr_filter_vertical_block
                end function fstarpu_df_alloc_csr_filter_vertical_block

                function fstarpu_df_alloc_matrix_filter_block () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_matrix_filter_block
                end function fstarpu_df_alloc_matrix_filter_block

                function fstarpu_df_alloc_matrix_filter_block_shadow () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_matrix_filter_block_shadow
                end function fstarpu_df_alloc_matrix_filter_block_shadow

                function fstarpu_df_alloc_matrix_filter_vertical_block () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_matrix_filter_vertical_block
                end function fstarpu_df_alloc_matrix_filter_vertical_block

                function fstarpu_df_alloc_matrix_filter_vertical_block_shadow () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_matrix_filter_vertical_block_shadow
                end function fstarpu_df_alloc_matrix_filter_vertical_block_shadow

                function fstarpu_df_alloc_vector_filter_block () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_vector_filter_block
                end function fstarpu_df_alloc_vector_filter_block

                function fstarpu_df_alloc_vector_filter_block_shadow () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_vector_filter_block_shadow
                end function fstarpu_df_alloc_vector_filter_block_shadow

                function fstarpu_df_alloc_vector_filter_list () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_vector_filter_list
                end function fstarpu_df_alloc_vector_filter_list

                function fstarpu_df_alloc_vector_filter_divide_in_2 () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_vector_filter_divide_in_2
                end function fstarpu_df_alloc_vector_filter_divide_in_2

                function fstarpu_df_alloc_block_filter_block () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_block_filter_block
                end function fstarpu_df_alloc_block_filter_block

                function fstarpu_df_alloc_block_filter_block_shadow () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_block_filter_block_shadow
                end function fstarpu_df_alloc_block_filter_block_shadow

                function fstarpu_df_alloc_block_filter_vertical_block () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_block_filter_vertical_block
                end function fstarpu_df_alloc_block_filter_vertical_block

                function fstarpu_df_alloc_block_filter_vertical_block_shadow () bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_df_alloc_block_filter_vertical_block_shadow
                end function fstarpu_df_alloc_block_filter_vertical_block_shadow

                subroutine fstarpu_data_filter_set_filter_func (filter, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: filter
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_data_filter_set_filter_func

                subroutine fstarpu_data_filter_set_nchildren (filter, nchildren) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: filter
                        integer(c_int), value, intent(in) :: nchildren
                end subroutine fstarpu_data_filter_set_nchildren

                subroutine fstarpu_data_filter_set_get_nchildren_func (filter, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: filter
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_data_filter_set_get_nchildren_func

                subroutine fstarpu_data_filter_set_get_child_ops_func (filter, f_ptr) bind(C)
                        use iso_c_binding, only: c_ptr, c_funptr
                        type(c_ptr), value, intent(in) :: filter
                        type(c_funptr), value, intent(in) :: f_ptr
                end subroutine fstarpu_data_filter_set_get_child_ops_func

                subroutine fstarpu_data_filter_set_filter_arg (filter, filter_arg) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: filter
                        integer(c_int), value, intent(in) :: filter_arg
                end subroutine fstarpu_data_filter_set_filter_arg

                subroutine fstarpu_data_filter_set_filter_arg_ptr (filter, filter_arg_ptr) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: filter_arg_ptr
                end subroutine fstarpu_data_filter_set_filter_arg_ptr

                ! void starpu_data_partition(starpu_data_handle_t initial_handle, struct starpu_data_filter *f);
                subroutine fstarpu_data_partition (dh,filter) bind(C,name="starpu_data_partition")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: filter
                end subroutine fstarpu_data_partition

                ! void starpu_data_unpartition(starpu_data_handle_t root_data, unsigned gathering_node);
                subroutine fstarpu_data_unpartition (root_dh,gathering_node) bind(C,name="starpu_data_unpartition")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: root_dh
                        integer(c_int), value, intent(in) :: gathering_node
                end subroutine fstarpu_data_unpartition

                ! void starpu_data_partition_plan(starpu_data_handle_t initial_handle, struct starpu_data_filter *f, starpu_data_handle_t *children);
                subroutine fstarpu_data_partition_plan (dh,filter,children) & 
                                bind(C,name="starpu_data_partition_plan")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), intent(in) :: children(*)
                end subroutine fstarpu_data_partition_plan

                ! void starpu_data_partition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);
                subroutine fstarpu_data_partition_submit (dh,nparts,children) &
                                bind(C,name="starpu_data_partition_submit")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: nparts
                        type(c_ptr), intent(in) :: children(*)
                end subroutine fstarpu_data_partition_submit

                ! void starpu_data_partition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);
                subroutine fstarpu_data_partition_readonly_submit (dh,nparts,children) &
                                bind(C,name="starpu_data_partition_readonly_submit")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: nparts
                        type(c_ptr), intent(in) :: children(*)
                end subroutine fstarpu_data_partition_readonly_submit

                ! void starpu_data_partition_readwrite_upgrade_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children);
                subroutine fstarpu_data_partition_readwrite_upgrade_submit (dh,nparts,children) &
                                bind(C,name="starpu_data_partition_readwrite_upgrade_submit")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: nparts
                        type(c_ptr), intent(in) :: children(*)
                end subroutine fstarpu_data_partition_readwrite_upgrade_submit

                ! void starpu_data_unpartition_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node);
                subroutine fstarpu_data_unpartition_submit (dh,nparts,children,gathering_node) &
                                bind(C,name="starpu_data_unpartition_submit")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: nparts
                        type(c_ptr), intent(in) :: children(*)
                        integer(c_int), value, intent(in) :: gathering_node
                end subroutine fstarpu_data_unpartition_submit

                ! void starpu_data_unpartition_readonly_submit(starpu_data_handle_t initial_handle, unsigned nparts, starpu_data_handle_t *children, int gathering_node);
                subroutine fstarpu_data_unpartition_readonly_submit (dh,nparts,children,gathering_node) &
                                bind(C,name="starpu_data_unpartition_readonly_submit")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: nparts
                        type(c_ptr), intent(in) :: children(*)
                        integer(c_int), value, intent(in) :: gathering_node
                end subroutine fstarpu_data_unpartition_readonly_submit

                ! void starpu_data_partition_clean(starpu_data_handle_t root_data, unsigned nparts, starpu_data_handle_t *children);
                subroutine fstarpu_data_partition_clean (dh,nparts,children) &
                                bind(C,name="starpu_data_partition_clean")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: nparts
                        type(c_ptr), intent(in) :: children(*)
                end subroutine fstarpu_data_partition_clean

                ! int starpu_data_get_nb_children(starpu_data_handle_t handle);
                function fstarpu_data_get_nb_children(dh) bind(C,name="starpu_data_get_nb_children")
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int)              :: fstarpu_data_get_nb_children
                        type(c_ptr), value, intent(in) :: dh
                end function fstarpu_data_get_nb_children

                ! starpu_data_handle_t starpu_data_get_child(starpu_data_handle_t handle, unsigned i);
                function fstarpu_data_get_child(dh,i) bind(C,name="starpu_data_get_child")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr)              :: fstarpu_data_get_child
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: i
                end function fstarpu_data_get_child

                ! starpu_data_handle_t starpu_data_get_sub_data(starpu_data_handle_t root_data, unsigned depth, ... );
                ! . see: fstarpu_data_get_sub_data
                ! starpu_data_handle_t starpu_data_vget_sub_data(starpu_data_handle_t root_data, unsigned depth, va_list pa);
                ! . see: fstarpu_data_get_sub_data

                ! note: defined in filters.c
                function fstarpu_data_get_sub_data (root_dh,depth,indices) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr)              :: fstarpu_data_get_sub_data
                        type(c_ptr), value, intent(in) :: root_dh
                        integer(c_int), value, intent(in) :: depth
                        integer(c_int), intent(in) :: indices(*)
                end function fstarpu_data_get_sub_data

                ! void starpu_data_map_filters(starpu_data_handle_t root_data, unsigned nfilters, ...);
                ! . see fstarpu_data_map_filters
                ! void starpu_data_vmap_filters(starpu_data_handle_t root_data, unsigned nfilters, va_list pa);
                ! . see fstarpu_data_map_filters

                ! note: defined in filters.c
                subroutine fstarpu_data_map_filters (root_dh,nfilters,filters) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: root_dh
                        integer(c_int), value, intent(in) :: nfilters
                        type(c_ptr), intent(in) :: filters(*)
                end subroutine fstarpu_data_map_filters

                ! void starpu_matrix_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_matrix_filter_block (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_matrix_filter_block")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_matrix_filter_block

                ! void starpu_matrix_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_matrix_filter_block_shadow (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_matrix_filter_block_shadow")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_matrix_filter_block_shadow

                ! void starpu_matrix_filter_vertical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_matrix_filter_vertical_block (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_matrix_filter_vertical_block")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_matrix_filter_vertical_block

                ! void starpu_matrix_filter_vertical_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_matrix_filter_vertical_block_shadow (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_matrix_filter_vertical_block_shadow")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_matrix_filter_vertical_block_shadow

                ! void starpu_vector_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_vector_filter_block (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_vector_filter_block")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_vector_filter_block

                ! void starpu_vector_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_vector_filter_block_shadow (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_vector_filter_block_shadow")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_vector_filter_block_shadow

                ! void starpu_vector_filter_list_long(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_vector_filter_list_long (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_vector_filter_list_long")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_vector_filter_list_long

                ! void starpu_vector_filter_list(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_vector_filter_list (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_vector_filter_list")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_vector_filter_list

                ! void starpu_vector_filter_divide_in_2(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_vector_divide_in_2 (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_vector_divide_in_2")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_vector_divide_in_2

                ! void starpu_block_filter_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_block_filter_block (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_block_filter_block")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_block_filter_block

                ! void starpu_block_filter_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_block_filter_block_shadow (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_block_filter_block_shadow")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_block_filter_block_shadow

                ! void starpu_block_filter_vertical_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_block_filter_vertical_block (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_block_filter_vertical_block")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_block_filter_vertical_block

                ! void starpu_block_filter_vertical_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_block_filter_vertical_block_shadow (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_block_filter_vertical_block_shadow")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_block_filter_vertical_block_shadow

                ! void starpu_block_filter_depth_block(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_block_filter_depth_block (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_block_filter_depth_block")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_block_filter_depth_block

                ! void starpu_block_filter_depth_block_shadow(void *father_interface, void *child_interface, struct starpu_data_filter *f, unsigned id, unsigned nparts);
                subroutine fstarpu_block_filter_depth_block_shadow (father_interface,child_interface,filter,id,nparts) &
                                bind(C,name="starpu_block_filter_depth_block_shadow")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: father_interface
                        type(c_ptr), value, intent(in) :: child_interface
                        type(c_ptr), value, intent(in) :: filter
                        type(c_ptr), value, intent(in) :: id
                        type(c_ptr), value, intent(in) :: nparts
                end subroutine fstarpu_block_filter_depth_block_shadow


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

                ! starpu_arbiter_t starpu_arbiter_create(void) STARPU_ATTRIBUTE_MALLOC;
                function fstarpu_arbiter_create () bind(C,name="starpu_arbiter_create")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_arbiter_create
                end function fstarpu_arbiter_create

                ! void starpu_data_assign_arbiter(starpu_data_handle_t handle, starpu_arbiter_t arbiter);
                subroutine fstarpu_data_assign_arbiter (dh,arbiter) bind(C,name="starpu_data_assign_arbiter")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), intent(out) :: dh
                        type(c_ptr), value, intent(in) :: arbiter
                end subroutine fstarpu_data_assign_arbiter

                ! void starpu_arbiter_destroy(starpu_arbiter_t arbiter);
                subroutine fstarpu_data_arbiter_destroy (arbiter) bind(C,name="starpu_data_arbiter_destroy")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: arbiter
                end subroutine fstarpu_data_arbiter_destroy

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

                !unsigned starpu_data_is_on_node(starpu_data_handle_t handle, unsigned node);
                function fstarpu_data_is_on_node(dh, node) &
                                bind(C,name="starpu_data_is_on_node")
                        use iso_c_binding, only: c_ptr, c_int
                        integer(c_int)                 :: fstarpu_data_is_on_node
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                end function fstarpu_data_is_on_node

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
                ! starpu_data_handle_t *fstarpu_data_handle_array_alloc(int nb);
                function fstarpu_data_handle_array_alloc (nb) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_data_handle_array_alloc
                        integer(c_int), value, intent(in) :: nb
                end function fstarpu_data_handle_array_alloc

                ! void fstarpu_data_handle_array_free(starpu_data_handle_t *handles);
                subroutine fstarpu_data_handle_array_free (handles) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: handles
                end subroutine fstarpu_data_handle_array_free

                ! void fstarpu_data_handle_array_set(starpu_data_handle_t *handles, int i, starpu_data_handle_t handle);
                subroutine fstarpu_data_handle_array_set (handles, i, handle) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: handles
                        integer(c_int), value, intent(in) :: i
                        type(c_ptr), value, intent(in) :: handle
                end subroutine fstarpu_data_handle_array_set

                ! struct starpu_data_descr *fstarpu_data_descr_array_alloc(int nb);
                function fstarpu_data_descr_array_alloc (nb) bind(C)
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr) :: fstarpu_data_descr_array_alloc
                        integer(c_int), value, intent(in) :: nb
                end function fstarpu_data_descr_array_alloc

                ! struct starpu_data_descr *fstarpu_data_descr_alloc(void);
                function fstarpu_data_descr_alloc (nb) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_data_descr_alloc
                end function fstarpu_data_descr_alloc

                ! void fstarpu_data_descr_array_free(struct starpu_data_descr *descrs);
                subroutine fstarpu_data_descr_array_free (descrs) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: descrs
                end subroutine fstarpu_data_descr_array_free

                ! void fstarpu_data_descr_free(struct starpu_data_descr *descr);
                subroutine fstarpu_data_descrg_free (descr) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: descr
                end subroutine fstarpu_data_descrg_free

                ! void fstarpu_data_descr_array_set(struct starpu_data_descr *descrs, int i, starpu_data_handle_t handle, intptr_t mode);
                subroutine fstarpu_data_descr_array_set (descrs, i, handle, mode) bind(C)
                        use iso_c_binding, only: c_ptr, c_int, c_intptr_t
                        type(c_ptr), value, intent(in) :: descrs
                        integer(c_int), value, intent(in) :: i
                        type(c_ptr), value, intent(in) :: handle
                        type(c_ptr), value, intent(in) :: mode ! C func expects c_intptr_t
                end subroutine fstarpu_data_descr_array_set

                ! void fstarpu_data_descr_set(struct starpu_data_descr *descr, starpu_data_handle_t handle, intptr_t mode);
                subroutine fstarpu_data_descr_set (descr, handle, mode) bind(C)
                        use iso_c_binding, only: c_ptr, c_intptr_t
                        type(c_ptr), value, intent(in) :: descr
                        type(c_ptr), value, intent(in) :: handle
                        type(c_ptr), value, intent(in) :: mode ! C func expects c_intptr_t
                end subroutine fstarpu_data_descr_set


                subroutine fstarpu_task_insert(arglist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), dimension(*), intent(in) :: arglist
                end subroutine fstarpu_task_insert
                subroutine fstarpu_insert_task(arglist) bind(C,name="fstarpu_task_insert")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), dimension(*), intent(in) :: arglist
                end subroutine fstarpu_insert_task

                subroutine fstarpu_unpack_arg(cl_arg,bufferlist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: cl_arg
                        type(c_ptr), dimension(*), intent(in) :: bufferlist
                end subroutine fstarpu_unpack_arg

                ! == starpu_sched_ctx.h ==

                ! starpu_sched_ctx_create: see fstarpu_sched_ctx_create
                function fstarpu_sched_ctx_create(workers_array,nworkers,ctx_name, arglist) bind(C)
                        use iso_c_binding, only: c_int, c_char, c_ptr
                        integer(c_int) :: fstarpu_sched_ctx_create
                        integer(c_int), intent(in) :: workers_array(*)
                        integer(c_int), value, intent(in) :: nworkers
                        character(c_char), intent(in) :: ctx_name
                        type(c_ptr), dimension(*), intent(in) :: arglist
                end function fstarpu_sched_ctx_create

                ! unsigned starpu_sched_ctx_create_inside_interval(const char *policy_name, const char *sched_ctx_name, int min_ncpus, int max_ncpus, int min_ngpus, int max_ngpus, unsigned allow_overlap);
                function fstarpu_sched_ctx_create_inside_interval(policy_name, sched_ctx_name, &
                                min_ncpus, max_ncpus, min_ngpus, max_ngpus, allow_overlap)     &
                                bind(C,name="starpu_sched_ctx_create_inside_interval")
                        use iso_c_binding, only: c_int, c_char
                        integer(c_int) :: fstarpu_sched_ctx_create_inside_interval
                        character(c_char), intent(in) :: policy_name
                        character(c_char), intent(in) :: sched_ctx_name
                        integer(c_int), value, intent(in) :: min_ncpus
                        integer(c_int), value, intent(in) :: max_ncpus
                        integer(c_int), value, intent(in) :: min_ngpus
                        integer(c_int), value, intent(in) :: max_ngpus
                        integer(c_int), value, intent(in) :: allow_overlap
                end function fstarpu_sched_ctx_create_inside_interval

                ! void starpu_sched_ctx_register_close_callback(unsigned sched_ctx_id, void (*close_callback)(unsigned sched_ctx_id, void* args), void *args);
                subroutine fstarpu_sched_ctx_register_close_callback (sched_ctx_id, close_callback, args) &
                                bind(c,name="starpu_sched_ctx_register_close_callback")
                        use iso_c_binding, only: c_ptr, c_funptr, c_int
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        type(c_funptr), value, intent(in) :: close_callback
                        type(c_ptr), value, intent(in) :: args
                end subroutine fstarpu_sched_ctx_register_close_callback

                ! void starpu_sched_ctx_add_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_add_workers(workerids,nworkers,ctx) bind(C,name="starpu_sched_ctx_add_workers")
                        use iso_c_binding, only: c_int
                        integer(c_int), intent(in) :: workerids (*)
                        integer(c_int), value, intent(in) :: nworkers
                        integer(c_int), value, intent(in) :: ctx
                end subroutine fstarpu_sched_ctx_add_workers

                ! void starpu_sched_ctx_remove_workers(int *workerids_ctx, int nworkers_ctx, unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_remove_workers(workerids,nworkers,ctx) bind(C,name="starpu_sched_ctx_remove_workers")
                        use iso_c_binding, only: c_int
                        integer(c_int), intent(in) :: workerids (*)
                        integer(c_int), value, intent(in) :: nworkers
                        integer(c_int), value, intent(in) :: ctx
                end subroutine fstarpu_sched_ctx_remove_workers

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
                subroutine fstarpu_sched_ctx_set_inheritor (ctx,inheritor) bind(C,name="starpu_sched_ctx_set_inheritor")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: ctx
                        integer(c_int), value, intent(in) :: inheritor
                end subroutine fstarpu_sched_ctx_set_inheritor

                ! unsigned starpu_sched_ctx_get_inheritor(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_inheritor (ctx) bind(C,name="starpu_sched_ctx_get_inheritor")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_inheritor
                        integer(c_int), value, intent(in) :: ctx
                end function fstarpu_sched_ctx_get_inheritor

                ! unsigned starpu_sched_ctx_get_hierarchy_level(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_hierarchy_level (ctx) bind(C,name="starpu_sched_ctx_get_hierarchy_level")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_hierarchy_level
                        integer(c_int), value, intent(in) :: ctx
                end function fstarpu_sched_ctx_get_hierarchy_level

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

                ! void starpu_sched_ctx_stop_task_submission(void);
                subroutine fstarpu_sched_ctx_stop_task_submission () bind(c,name="starpu_sched_ctx_stop_task_submission")
                        use iso_c_binding
                end subroutine fstarpu_sched_ctx_stop_task_submission

                ! void starpu_sched_ctx_finished_submit(unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_finished_submit (sched_ctx_id) bind(c,name="starpu_sched_ctx_finished_submit")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end subroutine fstarpu_sched_ctx_finished_submit

                ! unsigned starpu_sched_ctx_get_workers_list(unsigned sched_ctx_id, int **workerids);
                ! unsigned starpu_sched_ctx_get_workers_list_raw(unsigned sched_ctx_id, int **workerids);

                ! unsigned starpu_sched_ctx_get_nworkers(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_nworkers (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_nworkers")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_nworkers
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_nworkers

                ! unsigned starpu_sched_ctx_get_nshared_workers(unsigned sched_ctx_id, unsigned sched_ctx_id2);
                function fstarpu_sched_ctx_get_nshared_workers (sched_ctx_id, sched_ctx_id2) &
                                bind(c,name="starpu_sched_ctx_get_nshared_workers")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_nshared_workers
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), value, intent(in) :: sched_ctx_id2
                end function fstarpu_sched_ctx_get_nshared_workers

                ! unsigned starpu_sched_ctx_contains_worker(int workerid, unsigned sched_ctx_id);
                function fstarpu_sched_ctx_contains_worker (workerid, sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_contains_worker")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_contains_worker
                        integer(c_int), value, intent(in) :: workerid
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_contains_worker

                ! unsigned starpu_sched_ctx_contains_type_of_worker(enum starpu_worker_archtype arch, unsigned sched_ctx_id);
                function fstarpu_sched_ctx_contains_type_of_worker (arch, sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_contains_type_of_worker")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_contains_type_of_worker
                        integer(c_int), value, intent(in) :: arch
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_contains_type_of_worker

                ! unsigned starpu_sched_ctx_worker_get_id(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_worker_get_id (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_worker_get_id")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_worker_get_id
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_worker_get_id

                ! unsigned starpu_sched_ctx_get_ctx_for_task(struct starpu_task *task);
                function fstarpu_sched_ctx_get_ctx_for_task (task) &
                                bind(c,name="starpu_sched_ctx_get_ctx_for_task")
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int) :: fstarpu_sched_ctx_get_ctx_for_task
                        type(c_ptr), value, intent(in) :: task
                end function fstarpu_sched_ctx_get_ctx_for_task

                ! unsigned starpu_sched_ctx_overlapping_ctxs_on_worker(int workerid);
                function fstarpu_sched_ctx_overlapping_ctxs_on_worker (workerid) &
                                bind(c,name="starpu_sched_ctx_overlapping_ctxs_on_worker")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_overlapping_ctxs_on_worker
                        integer(c_int), value, intent(in) :: workerid
                end function fstarpu_sched_ctx_overlapping_ctxs_on_worker

                ! int starpu_sched_get_min_priority(void);
                function fstarpu_sched_get_min_priority () &
                                bind(c,name="starpu_sched_get_min_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_get_min_priority
                end function fstarpu_sched_get_min_priority

                ! int starpu_sched_get_max_priority(void);
                function fstarpu_sched_get_max_priority () &
                                bind(c,name="starpu_sched_get_max_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_get_max_priority
                end function fstarpu_sched_get_max_priority

                ! int starpu_sched_set_min_priority(int min_prio);
                function fstarpu_sched_set_min_priority (min_prio) &
                                bind(c,name="starpu_sched_set_min_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_set_min_priority
                        integer(c_int), value, intent(in) :: min_prio
                end function fstarpu_sched_set_min_priority

                ! int starpu_sched_set_max_priority(int max_prio);
                function fstarpu_sched_set_max_priority (max_prio) &
                                bind(c,name="starpu_sched_set_max_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_set_max_priority
                        integer(c_int), value, intent(in) :: max_prio
                end function fstarpu_sched_set_max_priority

                ! int starpu_sched_ctx_get_min_priority(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_min_priority (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_min_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_min_priority
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_min_priority

                ! int starpu_sched_ctx_get_max_priority(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_max_priority (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_max_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_max_priority
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_max_priority

                ! int starpu_sched_ctx_set_min_priority(unsigned sched_ctx_id, int min_prio);
                function fstarpu_sched_ctx_set_min_priority (sched_ctx_id, min_prio) &
                                bind(c,name="starpu_sched_ctx_set_min_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_set_min_priority
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), value, intent(in) :: min_prio
                end function fstarpu_sched_ctx_set_min_priority

                ! int starpu_sched_ctx_set_max_priority(unsigned sched_ctx_id, int max_prio);
                function fstarpu_sched_ctx_set_max_priority (sched_ctx_id, max_prio) &
                                bind(c,name="starpu_sched_ctx_set_max_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_set_max_priority
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), value, intent(in) :: max_prio
                end function fstarpu_sched_ctx_set_max_priority

                ! int starpu_sched_ctx_min_priority_is_set(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_min_priority_is_set (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_min_priority_is_set")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_min_priority_is_set
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_min_priority_is_set

                ! int starpu_sched_ctx_max_priority_is_set(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_max_priority_is_set (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_max_priority_is_set")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_max_priority_is_set
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_max_priority_is_set

                ! void *starpu_sched_ctx_get_user_data(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_user_data(sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_user_data")
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        type(c_ptr) :: fstarpu_sched_ctx_get_user_data
                end function fstarpu_sched_ctx_get_user_data

                ! struct starpu_worker_collection *starpu_sched_ctx_create_worker_collection(unsigned sched_ctx_id, enum starpu_worker_collection_type type) STARPU_ATTRIBUTE_MALLOC;

                ! void starpu_sched_ctx_delete_worker_collection(unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_delete_worker_collection (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_delete_worker_collection")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end subroutine fstarpu_sched_ctx_delete_worker_collection

                ! struct starpu_worker_collection *starpu_sched_ctx_get_worker_collection(unsigned sched_ctx_id);

                ! void starpu_sched_ctx_set_policy_data(unsigned sched_ctx_id, void *policy_data);
                subroutine fstarpu_sched_ctx_set_policy_data (sched_ctx_id, policy_data) &
                                bind(c,name="starpu_sched_ctx_set_policy_data")
                        use iso_c_binding, only: c_int, c_ptr
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        type(c_ptr), value, intent(in) :: policy_data
                end subroutine fstarpu_sched_ctx_set_policy_data

                ! void *starpu_sched_ctx_get_policy_data(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_policy_data (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_policy_data")
                        use iso_c_binding, only: c_int, c_ptr
                        type(c_ptr) :: fstarpu_sched_ctx_get_policy_data
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_policy_data

                ! void *starpu_sched_ctx_exec_parallel_code(void* (*func)(void*), void *param, unsigned sched_ctx_id);
                function fstarpu_sched_ctx_exec_parallel_code (func, param, sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_exec_parallel_code")
                        use iso_c_binding, only: c_int, c_funptr, c_ptr
                        type(c_ptr) :: fstarpu_sched_ctx_exec_parallel_code
                        type(c_funptr), value, intent(in) :: func
                        type(c_ptr), value, intent(in) :: param
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_exec_parallel_code


                ! int starpu_sched_ctx_get_nready_tasks(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_nready_tasks (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_nready_tasks")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_nready_tasks
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_nready_tasks

                ! double starpu_sched_ctx_get_nready_flops(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_nready_flops (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_nready_flops")
                        use iso_c_binding, only: c_int, c_double
                        real(c_double) :: fstarpu_sched_ctx_get_nready_flops
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_nready_flops

                ! void starpu_sched_ctx_list_task_counters_increment(unsigned sched_ctx_id, int workerid);
                subroutine fstarpu_sched_ctx_list_task_counters_increment (sched_ctx_id, workerid) &
                        bind(c,name="starpu_sched_ctx_list_task_counters_increment")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), value, intent(in) :: workerid
                end subroutine fstarpu_sched_ctx_list_task_counters_increment

                ! void starpu_sched_ctx_list_task_counters_decrement(unsigned sched_ctx_id, int workerid);
                subroutine fstarpu_sched_ctx_list_task_counters_decrement (sched_ctx_id, workerid) &
                        bind(c,name="starpu_sched_ctx_list_task_counters_decrement")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), value, intent(in) :: workerid
 
                end subroutine fstarpu_sched_ctx_list_task_counters_decrement

                ! void starpu_sched_ctx_list_task_counters_reset(unsigned sched_ctx_id, int workerid);
                subroutine fstarpu_sched_ctx_list_task_counters_reset (sched_ctx_id, workerid) &
                                bind(c,name="starpu_sched_ctx_list_task_counters_reset")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), value, intent(in) :: workerid
 
                end subroutine fstarpu_sched_ctx_list_task_counters_reset

                ! void starpu_sched_ctx_list_task_counters_increment_all(struct starpu_task *task, unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_list_task_counters_increment_all (task, sched_ctx_id) &
                        bind(c,name="starpu_sched_ctx_list_task_counters_increment_all")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end subroutine fstarpu_sched_ctx_list_task_counters_increment_all

                ! void starpu_sched_ctx_list_task_counters_decrement_all(struct starpu_task *task, unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_list_task_counters_decrement_all (task, sched_ctx_id) &
                        bind(c,name="starpu_sched_ctx_list_task_counters_decrement_all")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end subroutine fstarpu_sched_ctx_list_task_counters_decrement_all

                ! void starpu_sched_ctx_list_task_counters_reset_all(struct starpu_task *task, unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_list_task_counters_reset_all (task, sched_ctx_id) &
                        bind(c,name="starpu_sched_ctx_list_task_counters_reset_all")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end subroutine fstarpu_sched_ctx_list_task_counters_reset_all

                ! unsigned starpu_sched_ctx_get_priority(int worker, unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_priority (worker, sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_priority")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_priority
                        integer(c_int), value, intent(in) :: worker
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_priority

                ! void starpu_sched_ctx_get_available_cpuids(unsigned sched_ctx_id, int **cpuids, int *ncpuids);

                ! void starpu_sched_ctx_bind_current_thread_to_cpuid(unsigned cpuid);
                subroutine fstarpu_sched_ctx_bind_current_thread_to_cpuid (cpuid) &
                        bind(c,name="starpu_sched_ctx_bind_current_thread_to_cpuid")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: cpuid
                end subroutine fstarpu_sched_ctx_bind_current_thread_to_cpuid

                ! int starpu_sched_ctx_book_workers_for_task(unsigned sched_ctx_id, int *workerids, int nworkers);
                function fstarpu_sched_ctx_book_workers_for_task (sched_ctx_id, workerids, nworkers) &
                                bind(c,name="starpu_sched_ctx_book_workers_for_task")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_book_workers_for_task
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), intent(in) :: workerids(*)
                        integer(c_int), value, intent(in) :: nworkers
                end function fstarpu_sched_ctx_book_workers_for_task

                ! void starpu_sched_ctx_unbook_workers_for_task(unsigned sched_ctx_id, int master);
                subroutine fstarpu_sched_ctx_unbook_workers_for_task (sched_ctx_id, master) &
                                bind(c,name="starpu_sched_ctx_unbook_workers_for_task")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        integer(c_int), value, intent(in) :: master
                end subroutine fstarpu_sched_ctx_unbook_workers_for_task

                ! unsigned starpu_sched_ctx_worker_is_master_for_child_ctx(int workerid, unsigned sched_ctx_id);
                function fstarpu_sched_ctx_worker_is_master_for_child_ctx (workerid, sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_worker_is_master_for_child_ctx")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_worker_is_master_for_child_ctx
                        integer(c_int), value, intent(in) :: workerid
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_worker_is_master_for_child_ctx

                ! unsigned starpu_sched_ctx_master_get_context(int masterid);
                function fstarpu_sched_ctx_master_get_context (masterid) &
                                bind(c,name="starpu_sched_ctx_master_get_context")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_master_get_context
                        integer(c_int), value, intent(in) :: masterid
                end function fstarpu_sched_ctx_master_get_context

                ! void starpu_sched_ctx_revert_task_counters(unsigned sched_ctx_id, double flops);
                subroutine fstarpu_sched_ctx_revert_task_counters (sched_ctx_id, flops) &
                                bind(c,name="starpu_sched_ctx_revert_task_counters")
                        use iso_c_binding, only: c_int, c_double
                        integer(c_int), value, intent(in) :: sched_ctx_id
                        real(c_double), value, intent(in) :: flops
                end subroutine fstarpu_sched_ctx_revert_task_counters

                ! void starpu_sched_ctx_move_task_to_ctx(struct starpu_task *task, unsigned sched_ctx, unsigned manage_mutex);
                subroutine fstarpu_sched_ctx_move_task_to_ctx (task, sched_ctx, manage_mutex) &
                                bind(c,name="starpu_sched_ctx_move_task_to_ctx")
                        use iso_c_binding, only: c_ptr, c_int
                        type(c_ptr), value, intent(in) :: task
                        integer(c_int), value, intent(in) :: sched_ctx
                        integer(c_int), value, intent(in) :: manage_mutex
                end subroutine fstarpu_sched_ctx_move_task_to_ctx

                ! int starpu_sched_ctx_get_worker_rank(unsigned sched_ctx_id);
                function fstarpu_sched_ctx_get_worker_rank (sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_get_worker_rank")
                        use iso_c_binding, only: c_int
                        integer(c_int) :: fstarpu_sched_ctx_get_worker_rank
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end function fstarpu_sched_ctx_get_worker_rank

                ! unsigned starpu_sched_ctx_has_starpu_scheduler(unsigned sched_ctx_id, unsigned *awake_workers);

                ! void starpu_sched_ctx_call_pushed_task_cb(int workerid, unsigned sched_ctx_id);
                subroutine fstarpu_sched_ctx_call_pushed_task_cb (workerid, sched_ctx_id) &
                                bind(c,name="starpu_sched_ctx_call_pushed_task_cb")
                        use iso_c_binding, only: c_int
                        integer(c_int), value, intent(in) :: workerid
                        integer(c_int), value, intent(in) :: sched_ctx_id
                end subroutine fstarpu_sched_ctx_call_pushed_task_cb

                ! == starpu_fxt.h ==

                ! void starpu_fxt_options_init(struct starpu_fxt_options *options);
                subroutine fstarpu_fxt_options_init (fxt_options) bind(C,name="starpu_fxt_options_init")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: fxt_options
                end subroutine fstarpu_fxt_options_init

                ! void starpu_fxt_generate_trace(struct starpu_fxt_options *options);
                subroutine fstarpu_fxt_generate_trace (fxt_options) bind(C,name="starpu_fxt_generate_trace")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), value, intent(in) :: fxt_options
                end subroutine fstarpu_fxt_generate_trace

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
                function or_cptrs(op1,op2)
                        type(c_ptr) :: or_cptrs
                        type(c_ptr),intent(in) :: op1,op2
                        integer(c_intptr_t) :: i_op1,i_op2
                        i_op1 = transfer(op1,0_c_intptr_t)
                        i_op2 = transfer(op2,0_c_intptr_t)
                        or_cptrs = transfer(ior(i_op1,i_op2), C_NULL_PTR)
                end function

                function ip_to_p(i) bind(C)
                        use iso_c_binding, only: c_ptr,c_intptr_t,C_NULL_PTR
                        type(c_ptr) :: ip_to_p
                        integer(c_intptr_t), value, intent(in) :: i
                        ip_to_p = transfer(i,C_NULL_PTR)
                end function ip_to_p

                function p_to_ip(p) bind(C)
                        use iso_c_binding, only: c_ptr,c_intptr_t
                        integer(c_intptr_t) :: p_to_ip
                        type(c_ptr), value, intent(in) :: p
                        p_to_ip = transfer(p,0_c_intptr_t)
                end function p_to_ip

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

                        real(c_double) :: FSTARPU_SZ_C_DOUBLE_dummy
                        real(c_float) :: FSTARPU_SZ_C_FLOAT_dummy
                        character(c_char) :: FSTARPU_SZ_C_CHAR_dummy
                        integer(c_int) :: FSTARPU_SZ_C_INT_dummy
                        integer(c_intptr_t) :: FSTARPU_SZ_C_INTPTR_T_dummy
                        type(c_ptr) :: FSTARPU_SZ_C_PTR_dummy
                        integer(c_size_t) :: FSTARPU_SZ_C_SIZE_T_dummy

                        character :: FSTARPU_SZ_CHARACTER_dummy

                        integer :: FSTARPU_SZ_INTEGER_dummy
                        integer(4) :: FSTARPU_SZ_INT4_dummy
                        integer(8) :: FSTARPU_SZ_INT8_dummy

                        real :: FSTARPU_SZ_REAL_dummy
                        real(4) :: FSTARPU_SZ_REAL4_dummy
                        real(8) :: FSTARPU_SZ_REAL8_dummy

                        double precision :: FSTARPU_SZ_DOUBLE_PRECISION_dummy

                        complex :: FSTARPU_SZ_COMPLEX_dummy
                        complex(4) :: FSTARPU_SZ_COMPLEX4_dummy
                        complex(8) :: FSTARPU_SZ_COMPLEX8_dummy

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
                        FSTARPU_COMMUTE   = fstarpu_get_constant(C_CHAR_"FSTARPU_COMMUTE"//C_NULL_CHAR)
                        FSTARPU_SSEND   = fstarpu_get_constant(C_CHAR_"FSTARPU_SSEND"//C_NULL_CHAR)
                        FSTARPU_LOCALITY   = fstarpu_get_constant(C_CHAR_"FSTARPU_LOCALITY"//C_NULL_CHAR)

                        FSTARPU_DATA_ARRAY      = fstarpu_get_constant(C_CHAR_"FSTARPU_DATA_ARRAY"//C_NULL_CHAR)
                        FSTARPU_DATA_MODE_ARRAY = fstarpu_get_constant(C_CHAR_"FSTARPU_DATA_MODE_ARRAY"//C_NULL_CHAR)
                        FSTARPU_CL_ARGS = fstarpu_get_constant(C_CHAR_"FSTARPU_CL_ARGS"//C_NULL_CHAR)
                        FSTARPU_CL_ARGS_NFREE = fstarpu_get_constant(C_CHAR_"FSTARPU_CL_ARGS_NFREE"//C_NULL_CHAR)
                        FSTARPU_TASK_DEPS_ARRAY = fstarpu_get_constant(C_CHAR_"FSTARPU_TASK_DEPS_ARRAY"//C_NULL_CHAR)
                        FSTARPU_CALLBACK        = fstarpu_get_constant(C_CHAR_"FSTARPU_CALLBACK"//C_NULL_CHAR)
                        FSTARPU_CALLBACK_WITH_ARG       = fstarpu_get_constant(C_CHAR_"FSTARPU_CALLBACK_WITH_ARG"//C_NULL_CHAR)
                        FSTARPU_CALLBACK_ARG    = fstarpu_get_constant(C_CHAR_"FSTARPU_CALLBACK_ARG"//C_NULL_CHAR)
                        FSTARPU_PROLOGUE_CALLBACK       = fstarpu_get_constant(C_CHAR_"FSTARPU_PROLOGUE_CALLBACK"//C_NULL_CHAR)
                        FSTARPU_PROLOGUE_CALLBACK_ARG   = fstarpu_get_constant(C_CHAR_"FSTARPU_PROLOGUE_CALLBACK_ARG"//C_NULL_CHAR)
                        FSTARPU_PROLOGUE_CALLBACK_POP   = fstarpu_get_constant(C_CHAR_"FSTARPU_PROLOGUE_CALLBACK_POP"//C_NULL_CHAR)
                        FSTARPU_PROLOGUE_CALLBACK_POP_ARG       = &
                                fstarpu_get_constant(C_CHAR_"FSTARPU_PROLOGUE_CALLBACK_POP_ARG"//C_NULL_CHAR)
                        FSTARPU_PRIORITY        = fstarpu_get_constant(C_CHAR_"FSTARPU_PRIORITY"//C_NULL_CHAR)
                        FSTARPU_EXECUTE_ON_NODE = fstarpu_get_constant(C_CHAR_"FSTARPU_EXECUTE_ON_NODE"//C_NULL_CHAR)
                        FSTARPU_EXECUTE_ON_DATA = fstarpu_get_constant(C_CHAR_"FSTARPU_EXECUTE_ON_DATA"//C_NULL_CHAR)
                        FSTARPU_EXECUTE_ON_WORKER       = fstarpu_get_constant(C_CHAR_"FSTARPU_EXECUTE_ON_WORKER"//C_NULL_CHAR)
                        FSTARPU_WORKER_ORDER    = fstarpu_get_constant(C_CHAR_"FSTARPU_WORKER_ORDER"//C_NULL_CHAR)
                        FSTARPU_EXECUTE_WHERE       = fstarpu_get_constant(C_CHAR_"FSTARPU_EXECUTE_WHERE"//C_NULL_CHAR)
                        FSTARPU_HYPERVISOR_TAG  = fstarpu_get_constant(C_CHAR_"FSTARPU_HYPERVISOR_TAG"//C_NULL_CHAR)
                        FSTARPU_POSSIBLY_PARALLEL       = fstarpu_get_constant(C_CHAR_"FSTARPU_POSSIBLY_PARALLEL"//C_NULL_CHAR)
                        FSTARPU_FLOPS   = fstarpu_get_constant(C_CHAR_"FSTARPU_FLOPS"//C_NULL_CHAR)
                        FSTARPU_TAG     = fstarpu_get_constant(C_CHAR_"FSTARPU_TAG"//C_NULL_CHAR)
                        FSTARPU_TAG_ONLY        = fstarpu_get_constant(C_CHAR_"FSTARPU_TAG_ONLY"//C_NULL_CHAR)
                        FSTARPU_NAME    = fstarpu_get_constant(C_CHAR_"FSTARPU_NAME"//C_NULL_CHAR)
                        FSTARPU_NODE_SELECTION_POLICY   = fstarpu_get_constant(C_CHAR_"FSTARPU_NODE_SELECTION_POLICY"//C_NULL_CHAR)
                        FSTARPU_TASK_SCHED_DATA = fstarpu_get_constant(C_CHAR_"FSTARPU_TASK_SCHED_DATA"//C_NULL_CHAR)

                        FSTARPU_VALUE   = fstarpu_get_constant(C_CHAR_"FSTARPU_VALUE"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX   = fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX"//C_NULL_CHAR)
                        FSTARPU_CPU_WORKER   = fstarpu_get_constant(C_CHAR_"FSTARPU_CPU_WORKER"//C_NULL_CHAR)
                        FSTARPU_CUDA_WORKER   = fstarpu_get_constant(C_CHAR_"FSTARPU_CUDA_WORKER"//C_NULL_CHAR)
                        FSTARPU_OPENCL_WORKER   = fstarpu_get_constant(C_CHAR_"FSTARPU_OPENCL_WORKER"//C_NULL_CHAR)
                        FSTARPU_MIC_WORKER   = fstarpu_get_constant(C_CHAR_"FSTARPU_MIC_WORKER"//C_NULL_CHAR)
                        FSTARPU_ANY_WORKER   = fstarpu_get_constant(C_CHAR_"FSTARPU_ANY_WORKER"//C_NULL_CHAR)

                        FSTARPU_NMAXBUFS   = int(p_to_ip(fstarpu_get_constant(C_CHAR_"FSTARPU_NMAXBUFS"//C_NULL_CHAR)),c_int)

                        FSTARPU_SCHED_CTX_POLICY_NAME    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_POLICY_NAME"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_POLICY_STRUCT    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_POLICY_STRUCT"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_POLICY_MIN_PRIO    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_POLICY_MIN_PRIO"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_POLICY_MAX_PRIO    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_POLICY_MAX_PRIO"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_HIERARCHY_LEVEL    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_HIERARCHY_LEVEL"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_NESTED    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_NESTED"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_AWAKE_WORKERS    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_AWAKE_WORKERS"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_POLICY_INIT    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_POLICY_INIT"//C_NULL_CHAR)
                        FSTARPU_SCHED_CTX_USER_DATA    = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_SCHED_CTX_USER_DATA"//C_NULL_CHAR)

                        FSTARPU_NOWHERE = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_NOWHERE"//C_NULL_CHAR)
                        FSTARPU_CPU = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_CPU"//C_NULL_CHAR)
                        FSTARPU_CUDA = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_CUDA"//C_NULL_CHAR)
                        FSTARPU_OPENCL = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_OPENCL"//C_NULL_CHAR)
                        FSTARPU_MIC = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_MIC"//C_NULL_CHAR)

                        FSTARPU_CODELET_SIMGRID_EXECUTE = &
                             fstarpu_get_constant(C_CHAR_"FSTARPU_CODELET_SIMGRID_EXECUTE"//C_NULL_CHAR)
                        FSTARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT = &
                             fstarpu_get_constant(C_CHAR_"FSTARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT"//C_NULL_CHAR)
                        FSTARPU_CUDA_ASYNC = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_CUDA_ASYNC"//C_NULL_CHAR)
                        FSTARPU_OPENCL_ASYNC = &
                            fstarpu_get_constant(C_CHAR_"FSTARPU_OPENCL_ASYNC"//C_NULL_CHAR)

                        ! Initialize size constants as 'c_ptr'
                        FSTARPU_SZ_C_DOUBLE        = sz_to_p(c_sizeof(FSTARPU_SZ_C_DOUBLE_dummy))
                        FSTARPU_SZ_C_FLOAT        = sz_to_p(c_sizeof(FSTARPU_SZ_C_FLOAT_dummy))
                        FSTARPU_SZ_C_CHAR        = sz_to_p(c_sizeof(FSTARPU_SZ_C_CHAR_dummy))
                        FSTARPU_SZ_C_INT        = sz_to_p(c_sizeof(FSTARPU_SZ_C_INT_dummy))
                        FSTARPU_SZ_C_INTPTR_T   = sz_to_p(c_sizeof(FSTARPU_SZ_C_INTPTR_T_dummy))
                        FSTARPU_SZ_C_PTR        = sz_to_p(c_sizeof(FSTARPU_SZ_C_PTR_dummy))
                        FSTARPU_SZ_C_SIZE_T        = sz_to_p(c_sizeof(FSTARPU_SZ_C_SIZE_T_dummy))

                        FSTARPU_SZ_CHARACTER        = sz_to_p(c_sizeof(FSTARPU_SZ_CHARACTER_dummy))

                        FSTARPU_SZ_INTEGER         = sz_to_p(c_sizeof(FSTARPU_SZ_INTEGER_dummy))
                        FSTARPU_SZ_INT4         = sz_to_p(c_sizeof(FSTARPU_SZ_INT4_dummy))
                        FSTARPU_SZ_INT8         = sz_to_p(c_sizeof(FSTARPU_SZ_INT8_dummy))

                        FSTARPU_SZ_REAL        = sz_to_p(c_sizeof(FSTARPU_SZ_REAL_dummy))
                        FSTARPU_SZ_REAL4        = sz_to_p(c_sizeof(FSTARPU_SZ_REAL4_dummy))
                        FSTARPU_SZ_REAL8        = sz_to_p(c_sizeof(FSTARPU_SZ_REAL8_dummy))

                        FSTARPU_SZ_DOUBLE_PRECISION        = sz_to_p(c_sizeof(FSTARPU_SZ_DOUBLE_PRECISION_dummy))

                        FSTARPU_SZ_COMPLEX        = sz_to_p(c_sizeof(FSTARPU_SZ_COMPLEX_dummy))
                        FSTARPU_SZ_COMPLEX4        = sz_to_p(c_sizeof(FSTARPU_SZ_COMPLEX4_dummy))
                        FSTARPU_SZ_COMPLEX8        = sz_to_p(c_sizeof(FSTARPU_SZ_COMPLEX8_dummy))

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
