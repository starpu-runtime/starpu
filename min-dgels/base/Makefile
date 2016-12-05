#
#  Top Level Makefile for LAPACK
#  Version 3.2.1
#  June 2009
#

include make.inc

all: lib

lib: f2clib blaslib

clean: cleanlib

blaslib:
	( cd BLAS/SRC; $(MAKE) )

f2clib:
	    ( cd F2CLIBS/libf2c; $(MAKE) )

cleanlib:
	( cd BLAS/SRC; $(MAKE) clean )
	( cd SRC; $(MAKE) clean )
	( cd SRC/VARIANTS; $(MAKE) clean )
	( cd F2CLIBS/libf2c; $(MAKE) clean )
