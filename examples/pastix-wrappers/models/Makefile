#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

all: reg_gemm reg_trsm 

reg_axpy: reg_axpy.c num_rec
	gcc -o reg_axpy reg_axpy.c num_recipes/*.o -Inum_recipes/ -lm

reg_copy: reg_copy.c num_rec
	gcc -o reg_copy reg_copy.c num_recipes/*.o -Inum_recipes/ -lm

reg_geam: reg_geam.c num_rec
	gcc -o reg_geam reg_geam.c num_recipes/*.o -Inum_recipes/ -lm

reg_gemm: reg_gemm.c num_rec
	gcc -g -o reg_gemm reg_gemm.c num_recipes/*.o -Inum_recipes/ -lm

reg_pof: reg_pof.c num_rec
	gcc -o reg_pof reg_pof.c num_recipes/*.o -Inum_recipes/ -lm

reg_ppf: reg_ppf.c num_rec
	gcc -o reg_ppf reg_ppf.c num_recipes/*.o -Inum_recipes/ -lm

reg_scal: reg_scal.c num_rec
	gcc -o reg_scal reg_scal.c num_recipes/*.o -Inum_recipes/ -lm

reg_trsm: reg_trsm.c num_rec
	gcc -g -o reg_trsm reg_trsm.c num_recipes/*.o -Inum_recipes/ -lm

num_rec: covsrt.o gaussj.o lfit.o nrutil.o num_recipes/nr.h num_recipes/nrutil.h num_recipes/complex.h

covsrt.o: num_recipes/covsrt.c
	gcc -g -o num_recipes/covsrt.o -c num_recipes/covsrt.c

gaussj.o: num_recipes/gaussj.c
	gcc -g -o num_recipes/gaussj.o -c num_recipes/gaussj.c

lfit.o: num_recipes/lfit.c
	gcc -g -o num_recipes/lfit.o -c num_recipes/lfit.c

nrutil.o: num_recipes/nrutil.c
	gcc -g -o num_recipes/nrutil.o -c num_recipes/nrutil.c

clean:
	rm -f reg_gemm reg_trsm
	rm -rf *.o num_recipes/*.o
