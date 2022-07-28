! StarPU --- Runtime system for heterogeneous multicore architectures.
!
! Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
program nf_mm_2dbc
  use iso_c_binding       ! C interfacing module
  use fstarpu_mod         ! StarPU interfacing module
  use fstarpu_mpi_mod     ! StarPU-MPI interfacing module
  use nf_mm_cl
  use nf_mm_cl_blas
  implicit none

  type block_type
     real(kind=c_double), allocatable :: c(:,:)
     type(c_ptr)                      :: h
     integer                          :: owner
  end type block_type

  type dsmat_type
     integer                       :: m, n, b
     type(block_type), allocatable :: blocks(:,:)
  end type dsmat_type


  logical, parameter              :: verbose = .false.
  logical                         :: trace = .false.
  integer(c_int)                  :: comm_size, comm_rank
  integer(c_int), target          :: comm_world

  integer                         :: bs
  integer(c_int)                  :: m, mb
  integer(c_int)                  :: n, nb
  integer(c_int)                  :: k, kb
  character(len=20)               :: str

  type(dsmat_type),target         :: A, B, C
  real(kind=c_double), target     :: alpha, beta, zbeta
  type(c_ptr)                     :: cl_mm, cl_fill
  integer(c_int)                  :: ncpu
  integer(c_int)                  :: ret
  integer                         :: i, j, l, p , q, trial, t
  integer                         :: te, ts, tr
  real                            :: tf, gflops

  ret = fstarpu_init(C_NULL_PTR)
  if (ret == -19) then
     stop 77
  else if (ret /= 0) then
     stop 1
  end if

  ret = fstarpu_mpi_init(1)
  if (ret /= 0) then
     write(*,'("fstarpu_mpi_init status:",i4)') ret
     stop 1
  end if

  ! stop there if no CPU worker available
  ncpu = fstarpu_cpu_worker_get_count()
  if (ncpu == 0) then
     call fstarpu_shutdown()
     stop 77
  end if

  comm_world = fstarpu_mpi_world_comm()
  comm_size  = fstarpu_mpi_world_size()
  comm_rank  = fstarpu_mpi_world_rank()

  if (comm_size < 2) then
     call fstarpu_shutdown()
     ret = fstarpu_mpi_shutdown()
     stop 77
  end if

  if (command_argument_count() >= 1) then
     call get_command_argument(1, value=str, length=i)
     read(str(1:i),*) m
  else
     m = 10
  end if
  if (command_argument_count() >= 2) then
     call get_command_argument(2, value=str, length=i)
     read(str(1:i),*) n
  else
     n = 10
  end if
  if (command_argument_count() >= 3) then
     call get_command_argument(3, value=str, length=i)
     read(str(1:i),*) k
  else
     k = 10
  end if
  if (command_argument_count() >= 4) then
     call get_command_argument(4, value=str, length=i)
     read(str(1:i),*) bs
  else
     bs = 1
  end if
  if (command_argument_count() >= 5) then
     call get_command_argument(5, value=str, length=i)
     read(str(1:i),*) p
  else
     p = 1
  end if
  if (command_argument_count() >= 6) then
     call get_command_argument(6, value=str, length=i)
     read(str(1:i),*) q
  else
     q = 1
  end if
  if (command_argument_count() >= 8) then
     call get_command_argument(7, value=str, length=i)
     read(str(1:i),*) t
  else
     t = 1
  end if
  if (command_argument_count() == 8) then
     trace = .true.
  end if

  if (mod(m,bs).ne.0) stop 75
  if (mod(n,bs).ne.0) stop 75
  if (mod(k,bs).ne.0) stop 75
  mb = m/bs
  nb = n/bs
  kb = k/bs
  if (comm_rank.eq.0) then
     write(*,'("========================================")')
     write(*,'("mxnxk    = ",i5,"x",i5,"x",i5)') m, n, k
     write(*,'("mbxnbxkb = ",i5,"x",i5,"x",i5)') mb, nb, kb
     write(*,'("B        = ",i5)') bs
     write(*,'("PxQ      = ",i3,"x",i3)') p,q
     write(*,'("trace    = ",l)') trace
     write(*,'("========================================")')
  end if
  ret = fstarpu_mpi_barrier(comm_world)

  ! intialize codelets
  call initialize_codelets()
  alpha = 0.42
  beta = 3.14

  do trial=1,t
     ! allocate matrices
     call initialize_matrix(a,mb,kb,"A")
     call initialize_matrix(b,kb,nb,"B")
     call initialize_matrix(c,mb,nb,"C")
     ret = fstarpu_mpi_barrier(comm_world)

     call fill_matrix(A, mb,kb,"A")
     ret = fstarpu_mpi_wait_for_all(comm_world)
     ret = fstarpu_mpi_barrier(comm_world)

     call fill_matrix(B, kb,nb,"B")
     ret = fstarpu_mpi_wait_for_all(comm_world)
     ret = fstarpu_mpi_barrier(comm_world)

     call fill_matrix(C, mb,nb,"C")
     ret = fstarpu_mpi_wait_for_all(comm_world)
     ret = fstarpu_mpi_barrier(comm_world)

     call system_clock(ts)
     ! submit matrix multiplication
     do i=1,mb
        do j=1,nb
           do l=1,kb
              ! if (comm_rank.eq.0) write(*,*) "GEMM", b_col,b_row,b_aisle
              if (l.eq.1) then; zbeta = beta; else; zbeta = 1.0d0; end if
              call fstarpu_mpi_task_insert((/ c_loc(comm_world), cl_mm, &
                   FSTARPU_VALUE, c_loc(alpha), FSTARPU_SZ_REAL8,       &
                   FSTARPU_VALUE, c_loc(zbeta), FSTARPU_SZ_REAL8,       &
                   FSTARPU_R,  A%blocks(i,l)%h,                         &
                   FSTARPU_R,  B%blocks(l,j)%h,                         &
                   FSTARPU_RW, C%blocks(i,j)%h,                         &
                   c_null_ptr /))
           end do
        end do
     end do

     ret = fstarpu_mpi_wait_for_all(comm_world)
     ret = fstarpu_mpi_barrier(comm_world)
     call system_clock(te,tr)
     tf = max(real(te-ts)/real(tr),1e-20)
     gflops = 2.0*m*n*k/(tf*10**9)
     if (comm_rank.eq.0) write(*,'("RANK ",i3," -> took ",e15.8," s | ", e15.8,"Gflop/s")') &
             comm_rank, tf, gflops

     ! unregister matrices
     call unregister_matrix(A,mb,kb)
     call unregister_matrix(B,kb,nb)
     call unregister_matrix(C,mb,nb)
  end do


  call fstarpu_codelet_free(cl_mm)
  call fstarpu_codelet_free(cl_fill)
  call fstarpu_shutdown()

  ret = fstarpu_mpi_shutdown()
  if (ret /= 0) then
     write(*,'("fstarpu_mpi_shutdown status:",i4)') ret
     stop 1
  end if

contains

  subroutine initialize_codelets()
    implicit none
    cl_mm = fstarpu_codelet_allocate()
    call fstarpu_codelet_set_name(cl_mm, c_char_"nf_gemm_cl"//c_null_char)
    call fstarpu_codelet_add_cpu_func(cl_mm, C_FUNLOC(cl_cpu_gemm))
    call fstarpu_codelet_add_buffer(cl_mm, FSTARPU_R)
    call fstarpu_codelet_add_buffer(cl_mm, FSTARPU_R)
    call fstarpu_codelet_add_buffer(cl_mm, FSTARPU_RW)
    cl_fill = fstarpu_codelet_allocate()
    call fstarpu_codelet_set_name(cl_fill, c_char_"nf_fill_cl"//c_null_char)
    call fstarpu_codelet_add_cpu_func(cl_fill, C_FUNLOC(cl_cpu_fill))
    call fstarpu_codelet_add_buffer(cl_fill, FSTARPU_W)
  end subroutine initialize_codelets

  subroutine initialize_matrix(X,mb,nb,cname)
    implicit none
    type(dsmat_type), target        :: x
    integer                         :: mb, nb
    character                       :: cname

    integer                         :: i, j
    type(block_type), pointer       :: xij
    integer(c_int64_t), save        :: tag = 1

    x%m = mb*bs
    x%n = nb*bs
    x%b = bs
    allocate(x%blocks(mb,nb))
    do i=1,mb
       do j=1,nb
          xij => x%blocks(i,j)
          xij%owner = mod(i-1,p)*q + mod(j-1,q)
          if (comm_rank.eq.xij%owner) then
             ! write(*,*) comm_rank,"] I own ",cname,"_",i,j,"so I register it with tag",tag
             allocate(xij%c(bs,bs))
             call fstarpu_matrix_data_register( xij%h, 0, c_loc( xij%c(1,1) ), &
                  bs, bs, bs, c_sizeof(xij%c(1,1)) )
          else
             ! write(*,*) comm_rank,"] ",xij%owner," owns ",cname,"_",i,j,"so it registers it with tag",tag
             call fstarpu_matrix_data_register( xij%h, -1, c_null_ptr, &
                  bs, bs, bs, c_sizeof(alpha) )
          end if
          call fstarpu_mpi_data_register(xij%h, tag, xij%owner)
          tag = tag + 1
       end do
    end do
  end subroutine initialize_matrix

  subroutine fill_matrix(x,mb,nb,cname)
    implicit none
    type(dsmat_type), target        :: x
    integer                         :: mb, nb
    character                       :: cname

    integer                         :: i, j
    type(block_type), pointer       :: xij

    do i=1,mb
       do j=1,nb
          xij => x%blocks(i,j)
          if (comm_rank.eq.xij%owner) then
             ! write(*,*) comm_rank,"] I own ",cname,"_",i,j,"so I fill it"
             call fstarpu_mpi_task_insert((/ c_loc(comm_world), cl_fill, &
                  FSTARPU_W, xij%h, &
                  c_null_ptr /))
          else
             !write(*,*) comm_rank,"] ",xij%owner,"owns ",cname,"_",i,j,"so it fills it"
          end if
       end do
    end do
  end subroutine fill_matrix

  subroutine unregister_matrix(x,mb,nb)
    implicit none
    integer                         :: mb, nb
    type(block_type), pointer       :: xij
    type(dsmat_type), target        :: x

    integer         :: i, j

    do i=1,mb
       do j=1,nb
          xij => x%blocks(i,j)
          call fstarpu_data_unregister(xij%h)
          if (comm_rank.eq.xij%owner) then
             deallocate(xij%c)
          end if
       end do
    end do
    deallocate(x%blocks)
  end subroutine unregister_matrix

end program
