.PHONY: interfaces spu common

CFLAGS+= -I../

ifeq ($(CUBLAS), 1)
	CFLAGS+= -I$(CUDAINSTALLDIR)/include
	CFLAGS+= -DUSE_CUBLAS
endif

ifdef SPU
	SPU_TARGET=spu
endif

OBJS := coherency.o copy-driver.o hierarchy.o memalloc.o footprint.o

DWOBJDEPS += ../common/threads.o
DWOBJDEPS += ../common/hash.o
DWOBJDEPS += ../common/timing.o 
DWOBJDEPS += ../common/htable32.o 
DWOBJDEPS += ../common/mutex.o 
DWOBJDEPS += ../common/rwlock.o 
DWOBJDEPS += copy-driver.o
DWOBJDEPS += coherency.o 
DWOBJDEPS += hierarchy.o 
DWOBJDEPS += memalloc.o
DWOBJDEPS += footprint.o
DWOBJDEPS += interfaces/blas_filters.o
DWOBJDEPS += interfaces/csr_filters.o
DWOBJDEPS += interfaces/bcsr_filters.o
DWOBJDEPS += interfaces/vector_filters.o
DWOBJDEPS += interfaces/blas_interface.o
DWOBJDEPS += interfaces/csr_interface.o
DWOBJDEPS += interfaces/bcsr_interface.o
DWOBJDEPS += interfaces/vector_interface.o



all: datawizard.a interfaces $(SPE_TARGET) $(OBJS)

datawizard.so: common interfaces $(SPE_TARGET) $(OBJS) 
	gcc --shared -o datawizard.so $(DWOBJDEPS)

datawizard.a: common interfaces $(SPE_TARGET) $(OBJS)
	$(AR) rcs $@ $(DWOBJDEPS)

common:
	@make -C ../common/

interfaces:
	@make -C interfaces

spu:
	@make -C spu

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

clean:
	@make -C interfaces clean
	@make -C spu clean
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f *.a *.so
