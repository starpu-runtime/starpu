TOPDIR=../..
include $(TOPDIR)/make.inc

#######################################################################
#  This is the makefile to create a wrapper for the CBLAS.
#######################################################################

all: libcblaswr.a

libcblaswr.a: cblaswr.o
	$(ARCH) $(ARCHFLAGS) libcblaswr.a cblaswr.o
	$(RANLIB) libcblaswr.a
	
libfblaswr.a: fblaswr.o
	$(ARCH) $(ARCHFLAGS) libfblaswr.a fblaswr.o
	$(RANLIB) libfblaswr.a

clean:
	rm -f *.o *.a

.c.o: 
	$(CC) $(CFLAGS) -c $*.c

