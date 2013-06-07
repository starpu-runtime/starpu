#define STARPU_COMPLEX_GET_REAL(interface)	\
        (((struct starpu_complex_interface *)(interface))->real)
#define STARPU_COMPLEX_GET_IMAGINARY(interface)	\
        (((struct starpu_complex_interface *)(interface))->imaginary)
#define STARPU_COMPLEX_GET_NX(interface)	\
        (((struct starpu_complex_interface *)(interface))->nx)
