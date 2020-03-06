! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
MODULE starpu_mod
  ! == starpu.h ==

  ! starpu_conf_init
  INTERFACE
     SUBROUTINE starpu_conf_init(conf) BIND(C)
       USE iso_c_binding
       TYPE(C_PTR), VALUE :: conf
     END SUBROUTINE starpu_conf_init
  END INTERFACE

  ! starpu_init
  INTERFACE
     FUNCTION starpu_init(conf) BIND(C)
       USE iso_c_binding
       TYPE(C_PTR), VALUE :: conf
       INTEGER(KIND=C_INT) :: starpu_init
     END FUNCTION starpu_init
  END INTERFACE

  ! starpu_initialize

  ! starpu_pause
  INTERFACE
     SUBROUTINE starpu_pause() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_pause
  END INTERFACE

  ! starpu_resume
  INTERFACE
     SUBROUTINE starpu_resume() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_resume
  END INTERFACE

  ! starpu_shutdown
  INTERFACE
     SUBROUTINE starpu_shutdown() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_shutdown
  END INTERFACE

  ! starpu_topology_print

  ! starpu_asynchronous_copy_disabled
  INTERFACE
     SUBROUTINE starpu_asynchronous_copy_disabled() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_asynchronous_copy_disabled
  END INTERFACE

  ! starpu_asynchronous_cuda_copy_disabled
  INTERFACE
     SUBROUTINE starpu_asynchronous_cuda_copy_disabled() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_asynchronous_cuda_copy_disabled
  END INTERFACE

  ! starpu_asynchronous_opencl_copy_disabled
  INTERFACE
     SUBROUTINE starpu_asynchronous_opencl_copy_disabled() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_asynchronous_opencl_copy_disabled
  END INTERFACE

  ! starpu_asynchronous_mic_copy_disabled
  INTERFACE
     SUBROUTINE starpu_asynchronous_mic_copy_disabled() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_asynchronous_mic_copy_disabled
  END INTERFACE

  ! starpu_display_stats
  INTERFACE
     SUBROUTINE starpu_display_stats() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_display_stats
  END INTERFACE

  ! starpu_get_version
  INTERFACE
     SUBROUTINE starpu_get_version(major,minor,release) BIND(C)
       USE iso_c_binding
       INTEGER(KIND=C_INT), INTENT(OUT) :: major,minor,release
     END SUBROUTINE starpu_get_version
  END INTERFACE

  ! starpu_cpu_worker_get_count
  INTERFACE
     FUNCTION starpu_cpu_worker_get_count() BIND(C)
       USE iso_c_binding
       INTEGER(KIND=C_INT)              :: starpu_cpu_worker_get_count
     END FUNCTION starpu_cpu_worker_get_count
  END INTERFACE

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
  ! starpu_task_wait_for_all
  INTERFACE
     SUBROUTINE starpu_task_wait_for_all() BIND(C)
       USE iso_c_binding
     END SUBROUTINE starpu_task_wait_for_all
  END INTERFACE
  ! starpu_task_wait_for_n_submitted
  ! starpu_task_wait_for_all_in_ctx
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

END MODULE starpu_mod
