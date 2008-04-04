
externals/scalp/cell/gordon/libgordon.a:
	make -C externals/scalp/cell/gordon

externals/scalp/Make.env:
	make -C externals/scalp example

externals/scalp/util/util.o:
	make -C externals/scalp util/util.o

all: externals/scalp/Make.env externals/scalp/cell/gordon/libgordon.a externals/scalp/util/util.o

clean:
	make -C externals/scalp clean
