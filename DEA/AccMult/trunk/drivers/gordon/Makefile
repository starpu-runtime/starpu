CFLAGS+= -Iexternals/scalp/
CFLAGS+= -I../..

#gordon.o pputil.o spuembed.o

all: externals/scalp/Make.env externals/scalp/cell/gordon/gordon.o externals/scalp/cell/gordon/pputil.o externals/scalp/cell/gordon/spuembed.o externals/scalp/util/util.o mult_gordon.o

externals/scalp/cell/gordon/gordon.o:
	@make -C externals/scalp/cell/gordon gordon.o

externals/scalp/cell/gordon/pputil.o:
	@make -C externals/scalp/cell/gordon pputil.o

externals/scalp/cell/gordon/spuembed.o:
	@make -C externals/scalp/cell/gordon spuembed.o

externals/scalp/Make.env: externals/scalp/Make.env.ps3
	@make -C externals/scalp ps3

externals/scalp/util/util.o:
	@make -C externals/scalp util/util.o

mult_gordon.o: mult_gordon.c mult_gordon.h
	$(CC) $(CFLAGS) mult_gordon.c -c -o mult_gordon.o

clean: externals/scalp/Make.env
	@make -C externals/scalp clean
	@rm -f *.o
