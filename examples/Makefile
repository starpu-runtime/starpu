.PHONY: incrementer tag_example cuda heat mult spmv cholesky strassen pastix-wrappers 

TARGET=tag_example 
TARGET+= incrementer
#TARGET+= dw_sparse_cg

STARPU= $(PWD)/../starpu.a

EXTRADEP=

ifeq ($(CHECK), 1)
	CFLAGS += -DCHECK_OUTPUT
	CFLAGS += -DCHECK_RESULTS
endif

ifdef ATLAS
	TARGET += mult
	TARGET+= spmv
	TARGET+= heat 
	TARGET+= cholesky
	TARGET+= strassen
	TARGET+= pastix-wrappers
endif

CFLAGS += -I../

ifdef USE_FXT
	LDFLAGS += -lfxt
endif

all: $(TARGET)

pastix-wrappers:
	make -C pastix-wrappers

strassen:
	make -C strassen

incrementer:
	make -C incrementer

tag_example:
	make -C tag_example

cuda:
	make -C cuda

heat:
	make -C heat

mult:
	make -C mult

spmv:
	make -C spmv

cholesky:
	make -C cholesky

clean:
	@make -C pastix-wrappers clean
	@make -C cuda clean
	@make -C heat clean
	@make -C incrementer clean
	@make -C mult clean
	@make -C spmv clean
	@make -C cholesky clean
	@make -C strassen clean
	@make -C tag_example clean
	@rm -f *.o *.d *.gcno *.gcda
