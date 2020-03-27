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
module fstarpu_mpi_mod
        use iso_c_binding
        use fstarpu_mod
        implicit none

        interface
                ! == mpi/include/starpu_mpi.h ==
                ! int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);
                function fstarpu_mpi_isend (dh, mpi_req, dst, data_tag, mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: mpi_req
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_isend

                ! == mpi/include/starpu_mpi.h ==
                ! int starpu_mpi_isend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);
                function fstarpu_mpi_isend_prio (dh, mpi_req, dst, data_tag, prio, mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend_prio
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: mpi_req
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: prio
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_isend_prio

                ! int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *req, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm);
                function fstarpu_mpi_irecv (dh, mpi_req, src, data_tag, mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_irecv
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: mpi_req
                        integer(c_int), value, intent(in) :: src
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_irecv

                ! int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);
                function fstarpu_mpi_send (dh, dst, data_tag, mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_send
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_send

                ! int starpu_mpi_send_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);
                function fstarpu_mpi_send_prio (dh, dst, data_tag, prio, mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_send_prio
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: prio
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_send_prio

                ! int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, MPI_Status *status);
                function fstarpu_mpi_recv (dh, src, data_tag, mpi_comm, mpi_status) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_recv
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: src
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: mpi_status
                end function fstarpu_mpi_recv

                ! int starpu_mpi_isend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
                function fstarpu_mpi_isend_detached (dh, dst, data_tag, mpi_comm, callback, arg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend_detached
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: callback
                        type(c_ptr), value, intent(in) :: arg
                end function fstarpu_mpi_isend_detached

                ! int starpu_mpi_isend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg);
                function fstarpu_mpi_isend_detached_prio (dh, dst, data_tag, prio, mpi_comm, callback, arg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend_detached_prio
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: prio
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: callback
                        type(c_ptr), value, intent(in) :: arg
                end function fstarpu_mpi_isend_detached_prio

                ! int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
                function fstarpu_mpi_recv_detached (dh, src, data_tag, mpi_comm, callback, arg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_recv_detached
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: src
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: callback
                        type(c_ptr), value, intent(in) :: arg
                end function fstarpu_mpi_recv_detached

                ! int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm);
                function fstarpu_mpi_issend (dh, mpi_req, dst, data_tag, mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_issend
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: mpi_req
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_issend

                ! int starpu_mpi_issend_prio(starpu_data_handle_t data_handle, starpu_mpi_req *req, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm);
                function fstarpu_mpi_issend_prio (dh, mpi_req, dst, data_tag, prio, mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_issend_prio
                        type(c_ptr), value, intent(in) :: dh
                        type(c_ptr), value, intent(in) :: mpi_req
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: prio
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_issend_prio

                ! int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg);
                function fstarpu_mpi_issend_detached (dh, dst, data_tag, mpi_comm, callback, arg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_issend_detached
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: callback
                        type(c_ptr), value, intent(in) :: arg
                end function fstarpu_mpi_issend_detached

                ! int starpu_mpi_issend_detached_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, void (*callback)(void *), void *arg);
                function fstarpu_mpi_issend_detached_prio (dh, dst, data_tag, prio, mpi_comm, callback, arg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_issend_detached_prio
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: prio
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: callback
                        type(c_ptr), value, intent(in) :: arg
                end function fstarpu_mpi_issend_detached_prio

                ! int starpu_mpi_wait(starpu_mpi_req *req, MPI_Status *status);
                function fstarpu_mpi_wait(req,st) bind(C,name="starpu_mpi_wait")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_wait
                        type(c_ptr), value, intent(in) :: req
                        type(c_ptr), value, intent(in) :: st
                end function fstarpu_mpi_wait

                ! int starpu_mpi_test(starpu_mpi_req *req, int *flag, MPI_Status *status);
                function fstarpu_mpi_test(req,flag,st) bind(C,name="starpu_mpi_test")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_test
                        type(c_ptr), value, intent(in) :: req
                        type(c_ptr), value, intent(in) :: flag
                        type(c_ptr), value, intent(in) :: st
                end function fstarpu_mpi_test

                ! int starpu_mpi_barrier(MPI_Comm comm);
                function fstarpu_mpi_barrier (mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_barrier
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_barrier

                ! int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency);
                function fstarpu_mpi_recv_detached_sequential_consistency (dh, src, data_tag, mpi_comm, callback, arg, seq_const) &
                                bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_recv_detached_sequential_consistency
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: src
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: callback
                        type(c_ptr), value, intent(in) :: arg
                        integer(c_int), value, intent(in) :: seq_const
                end function fstarpu_mpi_recv_detached_sequential_consistency


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
                function fstarpu_mpi_task_build(arglist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr) :: fstarpu_mpi_task_build
                        type(c_ptr), dimension(*), intent(in) :: arglist
                end function fstarpu_mpi_task_build

                ! int starpu_mpi_task_post_build(MPI_Comm comm, struct starpu_codelet *codelet, ...);
                subroutine fstarpu_mpi_task_post_build(arglist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), dimension(*), intent(in) :: arglist
                end subroutine fstarpu_mpi_task_post_build

                ! int starpu_mpi_task_insert(MPI_Comm comm, struct starpu_codelet *codelet, ...);
                subroutine fstarpu_mpi_task_insert(arglist) bind(C)
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), dimension(*), intent(in) :: arglist
                end subroutine fstarpu_mpi_task_insert
                subroutine fstarpu_mpi_insert_task(arglist) bind(C,name="fstarpu_mpi_task_insert")
                        use iso_c_binding, only: c_ptr
                        type(c_ptr), dimension(*), intent(in) :: arglist
                end subroutine fstarpu_mpi_insert_task

                ! void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node);
                subroutine fstarpu_mpi_get_data_on_node(mpi_comm,dh,node) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                end subroutine fstarpu_mpi_get_data_on_node

                ! void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg);
                subroutine fstarpu_mpi_get_data_on_node_detached(mpi_comm,dh,node,callback,arg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: node
                        type(c_funptr), value, intent(in) :: callback
                        type(c_ptr), value, intent(in) :: arg
                end subroutine fstarpu_mpi_get_data_on_node_detached

                ! void starpu_mpi_redux_data(MPI_Comm comm, starpu_data_handle_t data_handle);
                subroutine fstarpu_mpi_redux_data(mpi_comm,dh) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_mpi_redux_data

                ! void starpu_mpi_redux_data_prio(MPI_Comm comm, starpu_data_handle_t data_handle, int prio);
                subroutine fstarpu_mpi_redux_data_prio(mpi_comm,dh, prio) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: prio
                end subroutine fstarpu_mpi_redux_data_prio

                ! int starpu_mpi_scatter_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);
                function fstarpu_mpi_scatter_detached (dhs, cnt, root, mpi_comm, scallback, sarg, rcallback, rarg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_scatter_detached
                        type(c_ptr), intent(in) :: dhs(*)
                        integer(c_int), value, intent(in) :: cnt
                        integer(c_int), value, intent(in) :: root
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: scallback
                        type(c_ptr), value, intent(in) :: sarg
                        type(c_funptr), value, intent(in) :: rcallback
                        type(c_ptr), value, intent(in) :: rarg
                end function fstarpu_mpi_scatter_detached

                ! int starpu_mpi_gather_detached(starpu_data_handle_t *data_handles, int count, int root, MPI_Comm comm, void (*scallback)(void *), void *sarg, void (*rcallback)(void *), void *rarg);
                function fstarpu_mpi_gather_detached (dhs, cnt, root, mpi_comm, scallback, sarg, rcallback, rarg) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_gather_detached
                        type(c_ptr), intent(in) :: dhs(*)
                        integer(c_int), value, intent(in) :: cnt
                        integer(c_int), value, intent(in) :: root
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_funptr), value, intent(in) :: scallback
                        type(c_ptr), value, intent(in) :: sarg
                        type(c_funptr), value, intent(in) :: rcallback
                        type(c_ptr), value, intent(in) :: rarg
                end function fstarpu_mpi_gather_detached


                ! int starpu_mpi_isend_detached_unlock_tag(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag);
                function fstarpu_mpi_isend_detached_unlock_tag (dh, dst, data_tag, mpi_comm, starpu_tag) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend_detached_unlock_tag
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: starpu_tag
                end function fstarpu_mpi_isend_detached_unlock_tag

                ! int starpu_mpi_isend_detached_unlock_tag_prio(starpu_data_handle_t data_handle, int dest, starpu_mpi_tag_t data_tag, int prio, MPI_Comm comm, starpu_tag_t tag);
                function fstarpu_mpi_isend_detached_unlock_tag_prio (dh, dst, data_tag, prio, mpi_comm, starpu_tag) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend_detached_unlock_tag_prio
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: dst
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: prio
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: starpu_tag
                end function fstarpu_mpi_isend_detached_unlock_tag_prio

                ! int starpu_mpi_irecv_detached_unlock_tag(starpu_data_handle_t data_handle, int source, starpu_mpi_tag_t data_tag, MPI_Comm comm, starpu_tag_t tag);
                function fstarpu_mpi_recv_detached_unlock_tag (dh, src, data_tag, mpi_comm, starpu_tag) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_recv_detached_unlock_tag
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: src
                        integer(c_int64_t), value, intent(in) :: data_tag
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: starpu_tag
                end function fstarpu_mpi_recv_detached_unlock_tag

                ! int starpu_mpi_isend_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, int *data_tag, MPI_Comm *comm, starpu_tag_t tag);
                function fstarpu_mpi_isend_array_detached_unlock_tag (array_size, dhs, dsts, data_tags, mpi_comms, starpu_tag) &
                                bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend_array_detached_unlock_tag
                        integer(c_int), value, intent(in) :: array_size
                        type(c_ptr), intent(in) :: dhs(*)
                        integer(c_int), intent(in) :: dsts(*)
                        integer(c_int64_t), intent(in) :: data_tags(*)
                        integer(c_int), intent(in) :: mpi_comms(*)
                        type(c_ptr), value, intent(in) :: starpu_tag
                end function fstarpu_mpi_isend_array_detached_unlock_tag

                ! int starpu_mpi_isend_array_detached_unlock_tag_prio(unsigned array_size, starpu_data_handle_t *data_handle, int *dest, int *data_tag, int *prio, MPI_Comm *comm, starpu_tag_t tag);
                function fstarpu_mpi_isend_array_detached_unlock_tag_prio (array_size, dhs, dsts, data_tags, prio, mpi_comms, &
                                starpu_tag) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_isend_array_detached_unlock_tag_prio
                        integer(c_int), value, intent(in) :: array_size
                        type(c_ptr), intent(in) :: dhs(*)
                        integer(c_int), intent(in) :: dsts(*)
                        integer(c_int64_t), intent(in) :: data_tags(*)
                        integer(c_int), intent(in) :: prio(*)
                        integer(c_int), intent(in) :: mpi_comms(*)
                        type(c_ptr), value, intent(in) :: starpu_tag
                end function fstarpu_mpi_isend_array_detached_unlock_tag_prio

                ! int starpu_mpi_irecv_array_detached_unlock_tag(unsigned array_size, starpu_data_handle_t *data_handle, int *source, int *data_tag, MPI_Comm *comm, starpu_tag_t tag);
                function fstarpu_mpi_recv_array_detached_unlock_tag (array_size, dhs, srcs, data_tags, mpi_comms, starpu_tag) &
                                bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_recv_array_detached_unlock_tag
                        integer(c_int), value, intent(in) :: array_size
                        type(c_ptr), intent(in) :: dhs(*)
                        integer(c_int), intent(in) :: srcs(*)
                        integer(c_int64_t), intent(in) :: data_tags(*)
                        integer(c_int), intent(in) :: mpi_comms(*)
                        type(c_ptr), value, intent(in) :: starpu_tag
                end function fstarpu_mpi_recv_array_detached_unlock_tag

                ! void starpu_mpi_comm_amounts_retrieve(size_t *comm_amounts);
                subroutine fstarpu_mpi_comm_amounts_retrieve (comm_amounts) bind(C,name="starpu_mpi_comm_amounts_retrieve")
                        use iso_c_binding
                        implicit none
                        integer(c_size_t), intent(in) :: comm_amounts(*)
                end subroutine fstarpu_mpi_comm_amounts_retrieve


                ! void starpu_mpi_cache_flush(MPI_Comm comm, starpu_data_handle_t data_handle);
                subroutine fstarpu_mpi_cache_flush(mpi_comm,dh) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: dh
                end subroutine fstarpu_mpi_cache_flush

                ! void starpu_mpi_cache_flush_all_data(MPI_Comm comm);
                subroutine fstarpu_mpi_cache_flush_all_data(mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                end subroutine fstarpu_mpi_cache_flush_all_data

                ! int starpu_mpi_comm_size(MPI_Comm comm, int *size);
                function fstarpu_mpi_comm_size(mpi_comm,sz) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        integer(c_int), intent(out) :: sz
                        integer(c_int) :: fstarpu_mpi_comm_size
                end function fstarpu_mpi_comm_size

                ! int starpu_mpi_comm_rank(MPI_Comm comm, int *rank);
                function fstarpu_mpi_comm_rank(mpi_comm,rank) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        integer(c_int), intent(out) :: rank
                        integer(c_int) :: fstarpu_mpi_comm_rank
                end function fstarpu_mpi_comm_rank


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

                ! int starpu_mpi_world_size(void);
                function fstarpu_mpi_world_comm() bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_world_comm
                end function fstarpu_mpi_world_comm

                ! int starpu_mpi_get_communication_tag(void);
                function fstarpu_mpi_get_communication_tag() bind(C,name="starpu_mpi_get_communication_tag")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_get_communication_tag
                end function fstarpu_mpi_get_communication_tag

                ! void starpu_mpi_set_communication_tag(int tag);
                subroutine fstarpu_mpi_set_communication_tag(tag) bind(C,name="starpu_mpi_set_communication_tag")
                        use iso_c_binding
                        implicit none
                        integer(c_int64_t), value, intent(in) :: tag
                end subroutine fstarpu_mpi_set_communication_tag

                ! void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, int tag, int rank, MPI_Comm comm);
                subroutine fstarpu_mpi_data_register_comm(dh,tag,rank,mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int64_t), value, intent(in) :: tag
                        integer(c_int), value, intent(in) :: rank
                        integer(c_int), value, intent(in) :: mpi_comm
                end subroutine fstarpu_mpi_data_register_comm

                ! #define starpu_mpi_data_register(data_handle, tag, rank) starpu_mpi_data_register_comm(data_handle, tag, rank, MPI_COMM_WORLD)
                subroutine fstarpu_mpi_data_register(dh,tag,rank) bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int64_t), value, intent(in) :: tag
                        integer(c_int), value, intent(in) :: rank
                end subroutine fstarpu_mpi_data_register

                ! void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm);
                subroutine fstarpu_mpi_data_set_rank_comm(dh,rank,mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: rank
                        integer(c_int), value, intent(in) :: mpi_comm
                end subroutine fstarpu_mpi_data_set_rank_comm

                ! #define starpu_mpi_data_set_rank(handle, rank) starpu_mpi_data_set_rank_comm(handle, rank, MPI_COMM_WORLD)
                subroutine fstarpu_mpi_data_set_rank(dh,rank) bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: rank
                end subroutine fstarpu_mpi_data_set_rank

                ! void starpu_mpi_data_set_tag(starpu_data_handle_t handle, int tag);
                subroutine fstarpu_mpi_data_set_tag(dh,tag) bind(C,name="starpu_mpi_data_set_tag")
                        use iso_c_binding
                        implicit none
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int64_t), value, intent(in) :: tag
                end subroutine fstarpu_mpi_data_set_tag

                ! int starpu_mpi_data_get_rank(starpu_data_handle_t handle);
                function fstarpu_mpi_data_get_rank(dh) bind(C,name="starpu_mpi_data_get_rank")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_data_get_rank
                        type(c_ptr), value, intent(in) :: dh
                end function fstarpu_mpi_data_get_rank

                ! int starpu_mpi_data_get_tag(starpu_data_handle_t handle);
                function fstarpu_mpi_data_get_tag(dh) bind(C,name="starpu_mpi_data_get_tag")
                        use iso_c_binding
                        implicit none
                        integer(c_int64_t) :: fstarpu_mpi_data_get_tag
                        type(c_ptr), value, intent(in) :: dh
                end function fstarpu_mpi_data_get_tag

                ! void starpu_mpi_data_migrate(MPI_Comm comm, starpu_data_handle_t handle, int rank);
                subroutine fstarpu_mpi_data_migrate(mpi_comm,dh,rank) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int), value, intent(in) :: mpi_comm
                        type(c_ptr), value, intent(in) :: dh
                        integer(c_int), value, intent(in) :: rank
                end subroutine fstarpu_mpi_data_migrate

                ! #define STARPU_MPI_NODE_SELECTION_CURRENT_POLICY -1
                ! #define STARPU_MPI_NODE_SELECTION_MOST_R_DATA    0

                ! int starpu_mpi_node_selection_register_policy(starpu_mpi_select_node_policy_func_t policy_func);
                function fstarpu_mpi_node_selection_register_policy(policy_func) &
                                bind(C,name="starpu_mpi_node_selection_register_policy")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_node_selection_register_policy
                        type(c_funptr), value, intent(in) :: policy_func
                end function fstarpu_mpi_node_selection_register_policy

                ! int starpu_mpi_node_selection_unregister_policy(int policy);
                function fstarpu_mpi_node_selection_unregister_policy(policy) &
                                bind(C,name="starpu_mpi_node_selection_unregister_policy")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_node_selection_unregister_policy
                        type(c_ptr), value, intent(in) :: policy
                end function fstarpu_mpi_node_selection_unregister_policy

                ! int starpu_mpi_node_selection_get_current_policy();
                function fstarpu_mpi_data_selection_get_current_policy() &
                                bind(C,name="starpu_mpi_data_selection_get_current_policy")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_data_selection_get_current_policy
                end function fstarpu_mpi_data_selection_get_current_policy

                ! int starpu_mpi_node_selection_set_current_policy(int policy);
                function fstarpu_mpi_data_selection_set_current_policy(policy) &
                                bind(C,name="starpu_mpi_data_selection_set_current_policy")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_data_selection_set_current_policy
                        type(c_ptr), value, intent(in) :: policy
                end function fstarpu_mpi_data_selection_set_current_policy

                ! int starpu_mpi_cache_is_enabled();
                function fstarpu_mpi_cache_is_enabled() bind(C,name="starpu_mpi_cache_is_enabled")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_cache_is_enabled
                end function fstarpu_mpi_cache_is_enabled

                ! int starpu_mpi_cache_set(int enabled);
                function fstarpu_mpi_cache_set(enabled) bind(C,name="starpu_mpi_cache_set")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_cache_set
                        integer(c_int), value, intent(in) :: enabled
                end function fstarpu_mpi_cache_set

                ! int starpu_mpi_wait_for_all(MPI_Comm comm);
                function fstarpu_mpi_wait_for_all (mpi_comm) bind(C)
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_wait_for_all
                        integer(c_int), value, intent(in) :: mpi_comm
                end function fstarpu_mpi_wait_for_all

                ! int starpu_mpi_datatype_register(starpu_data_handle_t handle, starpu_mpi_datatype_allocate_func_t allocate_datatype_func, starpu_mpi_datatype_free_func_t free_datatype_func);
                function fstarpu_mpi_datatype_register(dh, alloc_func, free_func) bind(C,name="starpu_mpi_datatype_register")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_datatype_register
                        type(c_ptr), value, intent(in) :: dh
                        type(c_funptr), value, intent(in) :: alloc_func
                        type(c_funptr), value, intent(in) :: free_func
                end function fstarpu_mpi_datatype_register

                ! int starpu_mpi_datatype_unregister(starpu_data_handle_t handle);
                function fstarpu_mpi_datatype_unregister(dh) bind(C,name="starpu_mpi_datatype_unregister")
                        use iso_c_binding
                        implicit none
                        integer(c_int) :: fstarpu_mpi_datatype_unregister
                        type(c_ptr), value, intent(in) :: dh
                end function fstarpu_mpi_datatype_unregister


                function fstarpu_mpi_req_alloc() bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr) :: fstarpu_mpi_req_alloc
                end function fstarpu_mpi_req_alloc

                subroutine fstarpu_mpi_req_free(req) bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr),value,intent(in) :: req
                end subroutine fstarpu_mpi_req_free

                function fstarpu_mpi_status_alloc() bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr) :: fstarpu_mpi_status_alloc
                end function fstarpu_mpi_status_alloc

                subroutine fstarpu_mpi_status_free(st) bind(C)
                        use iso_c_binding
                        implicit none
                        type(c_ptr),value,intent(in) :: st
                end subroutine fstarpu_mpi_status_free



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
