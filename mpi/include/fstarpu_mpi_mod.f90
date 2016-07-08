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

module fstarpu_mpi_mod
        use iso_c_binding
        use fstarpu_mod
        implicit none

        ! TODO:
        ! starpu_mpi_data_register
        ! starpu_mpi_get_data_on_node
        ! starpu_mpi_data_set_rank
        ! starpu_mpi_init
        ! starpu_mpi_shutdown
        ! starpu_mpi_comm_rank
        ! starpu_mpi_comm_size
        ! starpu_mpi_task_insert

        interface
                ! == mpi/include/starpu_mpi.h ==
                ! int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, int mpi_tag, MPI_Comm comm);
                ! int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *req, int source, int mpi_tag, MPI_Comm comm);
                ! int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, int mpi_tag, MPI_Comm comm);
                ! int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, MPI_Status *status);
                ! int starpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dest, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
                ! int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
                ! int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, int mpi_tag, MPI_Comm comm);
                ! int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
                ! int starpu_mpi_wait(starpu_mpi_req *req, MPI_Status *status);
                ! int starpu_mpi_test(starpu_mpi_req *req, int *flag, MPI_Status *status);
                ! int starpu_mpi_barrier(MPI_Comm comm);
                ! int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency);

                ! int starpu_mpi_init_comm(int *argc, char ***argv, int initialize_mpi, MPI_Comm comm);
                ! -> cf fstarpu_mpi_init
                ! int starpu_mpi_init(int *argc, char ***argv, int initialize_mpi);
                ! -> cf fstarpu_mpi_init

                ! int starpu_mpi_initialize(void) STARPU_DEPRECATED;
                ! -> cf fstarpu_mpi_init
                ! int starpu_mpi_initialize_extended(int *rank, int *world_size) STARPU_DEPRECATED;
                ! -> cf fstarpu_mpi_init

                ! int starpu_mpi_shutdown(void);
                function fstarpu_mpi_shutdown () bind(C,name="starpu_mpi_shutdown")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_shutdown
                end function fstarpu_mpi_shutdown

                ! struct starpu_task *starpu_mpi_task_build(MPI_Comm comm, struct starpu_codelet *codelet, ...);
                ! int starpu_mpi_task_post_build(MPI_Comm comm, struct starpu_codelet *codelet, ...);
                ! int starpu_mpi_task_insert(MPI_Comm comm, struct starpu_codelet *codelet, ...);
                ! /* the function starpu_mpi_insert_task has the same semantics as starpu_mpi_task_insert, it is kept to avoid breaking old codes */
                ! int starpu_mpi_insert_task(MPI_Comm comm, struct starpu_codelet *codelet, ...);
                ! void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node);
                ! void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg);
                ! void starpu_mpi_redux_data(MPI_Comm comm, starpu_data_handle_t data_handle);
                ! int starpu_mpi_scatter_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);
                ! int starpu_mpi_gather_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);
                ! int starpu_mpi_isend_detached_unlock_tag(starpu_data_handle_t data_handle, int dest, int mpi_tag, MPI_Comm comm, starpu_tag_t tag);
                ! int starpu_mpi_irecv_detached_unlock_tag(starpu_data_handle_t data_handle, int source, int mpi_tag, MPI_Comm comm, starpu_tag_t tag);
                ! int starpu_mpi_isend_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, int *mpi_tag, MPI_Comm *comm, starpu_tag_t tag);
                ! int starpu_mpi_irecv_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *source, int *mpi_tag, MPI_Comm *comm, starpu_tag_t tag);
                ! void starpu_mpi_comm_amounts_retrieve(size_t *comm_amounts);
                ! void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle);
                ! void starpu_mpi_cache_flush_all_data(MPI_Comm comm);
                ! int starpu_mpi_comm_size(MPI_Comm comm, int *size);
                ! int starpu_mpi_comm_rank(MPI_Comm comm, int *rank);

                ! int starpu_mpi_world_rank(void);
                function fstarpu_mpi_world_rank() bind(C,name="starpu_mpi_world_rank")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_world_rank
                end function fstarpu_mpi_world_rank

                ! int starpu_mpi_world_size(void);
                function fstarpu_mpi_world_size() bind(C,name="starpu_mpi_world_size")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_world_size
                end function fstarpu_mpi_world_size

                ! int starpu_mpi_get_communication_tag(void);
                ! void starpu_mpi_set_communication_tag(int tag);
                ! void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, int tag, int rank, MPI_Comm comm);
                ! #define starpu_mpi_data_register(data_handle, tag, rank) starpu_mpi_data_register_comm(data_handle, tag, rank, MPI_COMM_WORLD)
                ! void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm);
                ! #define starpu_mpi_data_set_rank(handle, rank) starpu_mpi_data_set_rank_comm(handle, rank, MPI_COMM_WORLD)
                ! void starpu_mpi_data_set_tag(starpu_data_handle_t handle, int tag);
                ! #define starpu_data_set_rank starpu_mpi_data_set_rank
                ! #define starpu_data_set_tag starpu_mpi_data_set_tag
                ! int starpu_mpi_data_get_rank(starpu_data_handle_t handle);
                ! int starpu_mpi_data_get_tag(starpu_data_handle_t handle);
                ! #define starpu_data_get_rank starpu_mpi_data_get_rank
                ! #define starpu_data_get_tag starpu_mpi_data_get_tag
                ! #define STARPU_MPI_NODE_SELECTION_CURRENT_POLICY -1
                ! #define STARPU_MPI_NODE_SELECTION_MOST_R_DATA    0
                ! typedef int (*starpu_mpi_select_node_policy_func_t)(int me, int nb_nodes, struct starpu_data_descr *descr, int nb_data);
                ! int starpu_mpi_node_selection_register_policy(starpu_mpi_select_node_policy_func_t policy_func);
                ! int starpu_mpi_node_selection_unregister_policy(int policy);
                ! int starpu_mpi_node_selection_get_current_policy();
                ! int starpu_mpi_node_selection_set_current_policy(int policy);
                ! int starpu_mpi_cache_is_enabled();
                ! int starpu_mpi_cache_set(int enabled);
                ! int starpu_mpi_wait_for_all(MPI_Comm comm);
                ! int starpu_mpi_datatype_register(starpu_data_handle_t handle, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func);
                ! int starpu_mpi_datatype_unregister(starpu_data_handle_t handle);
        end interface

        contains
                function fstarpu_mpi_init (initialize_mpi,mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_init
                        integer(c_int), intent(in) :: initialize_mpi
                        integer(c_int), optional, intent(in) :: mpi_comm
                        type(c_ptr) :: argcv
                        integer(c_int) :: fargc,i,farg_len
                        character(len=1) :: farg_1
                        character(len=:), allocatable :: farg
                        integer(c_int) :: mpi_comm_present, mpi_comm_or_0
                        integer(c_int) :: ret

                        interface
                                function fstarpu_mpi_argcv_alloc(argc, initialize_mpi, comm_present, comm) bind(C)
                                        use iso_c_binding
                                        implicit none
                                        type(c_ptr) :: fstarpu_mpi_argcv_alloc
                                        integer(c_int),value,intent(in) :: argc
                                        integer(c_int),value,intent(in) :: initialize_mpi
                                        integer(c_int),value,intent(in) :: comm_present
                                        integer(c_int),value,intent(in) :: comm
                                end function fstarpu_mpi_argcv_alloc

                                subroutine fstarpu_mpi_argcv_set_arg(argcv, i, l, s) bind(C)
                                        use iso_c_binding
                                        implicit none
                                        type(c_ptr),value,intent(in) :: argcv
                                        integer(c_int),value,intent(in) :: i
                                        integer(c_int),value,intent(in) :: l
                                        character(c_char),intent(in) :: s
                                end subroutine fstarpu_mpi_argcv_set_arg

                                subroutine fstarpu_mpi_argcv_free(argcv) bind(C)
                                        use iso_c_binding
                                        implicit none
                                        type(c_ptr),value,intent(in) :: argcv
                                end subroutine fstarpu_mpi_argcv_free

                                function fstarpu_mpi_init_c(argcv) bind(C)
                                        use iso_c_binding
                                        implicit none
                                        integer(c_int) :: fstarpu_mpi_init_c
                                        type(c_ptr),value,intent(in) :: argcv
                                end function fstarpu_mpi_init_c
                        end interface

                        fargc = command_argument_count()
                        write(*,*) "fargc",fargc
                        if (present(mpi_comm)) then
                                mpi_comm_present = 1
                                mpi_comm_or_0 = mpi_comm
                        else
                                mpi_comm_present = 0
                                mpi_comm_or_0 = 0
                        end if
                        write(*,*) "initialize_mpi",initialize_mpi
                        write(*,*) "mpi_comm_present",mpi_comm_present
                        argcv = fstarpu_mpi_argcv_alloc(fargc, initialize_mpi, mpi_comm_present, mpi_comm_or_0)
                        do i=0,fargc-1
                                call get_command_argument(i, farg_1, farg_len)
                                allocate (character(len=farg_len) :: farg)
                                call get_command_argument(i, farg)
                                call fstarpu_mpi_argcv_set_arg(argcv, i, farg_len, farg)
                                deallocate (farg)
                        end do
                        ret = fstarpu_mpi_init_c(argcv)
                        call fstarpu_mpi_argcv_free(argcv)
                        fstarpu_mpi_init = ret
                end function fstarpu_mpi_init

end module fstarpu_mpi_mod
