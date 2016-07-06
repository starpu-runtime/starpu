module nf_dynbuf_cl
contains
recursive subroutine cl_cpu_func_dynbuf_big (buffers, cl_args) bind(C)
        use iso_c_binding       ! C interfacing module
        use fstarpu_mod         ! StarPU interfacing module
        implicit none

        type(c_ptr), value, intent(in) :: buffers, cl_args ! cl_args is unused
        integer(c_int),target :: nb_data
        integer(c_int),pointer :: val
        integer(c_int) :: i

        call fstarpu_unpack_arg(cl_args,(/ c_loc(nb_data) /))
        write(*,*) "number of data:", nb_data
        do i=0,nb_data-1
                call c_f_pointer(fstarpu_variable_get_ptr(buffers, i), val)
                write(*,*) "i:", i, ", val:", val
                if (val /= 42) then
                        stop 1
                end if
        end do
end subroutine cl_cpu_func_dynbuf_big
end module nf_dynbuf_cl
