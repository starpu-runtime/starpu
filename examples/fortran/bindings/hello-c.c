#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <f77.h>

#define fline_length 80

extern F77_SUBROUTINE(hellosub)( INTEGER(i) TRAIL(line) );


void dummy_c_func_(INTEGER(i))
{
	fprintf(stderr, "i = %d\n", *INTEGER_ARG(i));

	F77_CALL(hellosub)(INTEGER_ARG(i)TRAIL_ARG(fline));
}
