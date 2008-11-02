CFLAGS+= -Iexternals/scalp/
CFLAGS+= -I../..

have_gordon := $(wildcard externals/scalp)

ifeq ($(strip $(have_gordon)),externals/scalp) 
all: externals/scalp/Make.env externals/scalp/cell/gordon/gordon.o externals/scalp/cell/gordon/pputil.o externals/scalp/cell/gordon/spuembed.o externals/scalp/util/util.o driver_gordon.o

externals/scalp/cell/gordon/gordon.o:
	@make -C externals/scalp/cell/gordon gordon.o

externals/scalp/cell/gordon/pputil.o:
	@make -C externals/scalp/cell/gordon pputil.o

externals/scalp/cell/gordon/spuembed.o:
	@make -C externals/scalp/cell/gordon spuembed.o

externals/scalp/Make.env: externals/scalp/Make.env.looping
	@make -C externals/scalp looping

externals/scalp/util/util.o:
	@make -C externals/scalp util/util.o

driver_gordon.o: driver_gordon.c driver_gordon.h
	$(CC) $(CFLAGS) driver_gordon.c -c -o driver_gordon.o

clean:  externals/scalp/Make.env 
	@make -C externals/scalp clean
	@rm -f *.o


else
all:

clean:
endif
