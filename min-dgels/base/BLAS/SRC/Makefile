TOPDIR=../..
include $(TOPDIR)/make.inc

#######################################################################
#  This is the makefile to create a library for the BLAS.
#  The files are grouped as follows:
#
#       SBLAS1 -- Single precision real BLAS routines
#       CBLAS1 -- Single precision complex BLAS routines
#       DBLAS1 -- Double precision real BLAS routines
#       ZBLAS1 -- Double precision complex BLAS routines
#
#       CB1AUX -- Real BLAS routines called by complex routines
#       ZB1AUX -- D.P. real BLAS routines called by d.p. complex
#                 routines
#
#      ALLBLAS -- Auxiliary routines for Level 2 and 3 BLAS
#
#       SBLAS2 -- Single precision real BLAS2 routines
#       CBLAS2 -- Single precision complex BLAS2 routines
#       DBLAS2 -- Double precision real BLAS2 routines
#       ZBLAS2 -- Double precision complex BLAS2 routines
#
#       SBLAS3 -- Single precision real BLAS3 routines
#       CBLAS3 -- Single precision complex BLAS3 routines
#       DBLAS3 -- Double precision real BLAS3 routines
#       ZBLAS3 -- Double precision complex BLAS3 routines
#
#  The library can be set up to include routines for any combination
#  of the four precisions.  To create or add to the library, enter make
#  followed by one or more of the precisions desired.  Some examples:
#       make single
#       make single complex
#       make single double complex complex16
#  Note that these commands are not safe for parallel builds.
#
#  Alternatively, the commands
#       make all
#  or
#       make
#  without any arguments creates a library of all four precisions.
#  The name of the library is held in BLASLIB, which is set in the
#  top-level make.inc
#
#  To remove the object files after the library is created, enter
#       make clean
#  To force the source files to be recompiled, enter, for example,
#       make single FRC=FRC
#
#---------------------------------------------------------------------
#
#  Edward Anderson, University of Tennessee
#  March 26, 1990
#  Susan Ostrouchov, Last updated September 30, 1994
#  ejr, May 2006.
#
#######################################################################

all: $(BLASLIB)
 
#---------------------------------------------------------
#  Comment out the next 6 definitions if you already have
#  the Level 1 BLAS.
#---------------------------------------------------------

DBLAS1 = idamax.o dasum.o daxpy.o dcopy.o ddot.o dnrm2.o \
	drot.o drotg.o dscal.o dsdot.o dswap.o drotmg.o drotm.o
$(DBLAS1): $(FRC)

#---------------------------------------------------------------------
#  The following line defines auxiliary routines needed by both the
#  Level 2 and Level 3 BLAS.  Comment it out only if you already have
#  both the Level 2 and 3 BLAS.
#---------------------------------------------------------------------

ALLBLAS  = lsame.o xerbla.o xerbla_array.o
$(ALLBLAS) : $(FRC)

#---------------------------------------------------------
#  Comment out the next 4 definitions if you already have
#  the Level 2 BLAS.
#---------------------------------------------------------

DBLAS2 = dgemv.o dgbmv.o dsymv.o dsbmv.o dspmv.o \
	dtrmv.o dtbmv.o dtpmv.o dtrsv.o dtbsv.o dtpsv.o \
	dger.o dsyr.o dspr.o dsyr2.o dspr2.o
$(DBLAS2): $(FRC)

#---------------------------------------------------------
#  Comment out the next 4 definitions if you already have
#  the Level 3 BLAS.
#---------------------------------------------------------

DBLAS3 = dgemm.o dsymm.o dsyrk.o dsyr2k.o dtrmm.o dtrsm.o
$(DBLAS3): $(FRC)

ALLOBJ= $(DBLAS1) $(DBLAS2) $(DBLAS3) $(ALLBLAS)

$(BLASLIB): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

double: $(DBLAS1) $(ALLBLAS) $(DBLAS2) $(DBLAS3)
	$(ARCH) $(ARCHFLAGS) $(BLASLIB) $(DBLAS1) $(ALLBLAS) \
	$(DBLAS2) $(DBLAS3)
	$(RANLIB) $(BLASLIB)

FRC:
	@FRC=$(FRC)

clean:
	rm -f *.o

.c.o: 
	$(CC) $(CFLAGS) -c $< -o $@
