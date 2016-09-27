include min-make.inc

all: clean install

install:
	mkdir -p $(BUILD);
	cd $(CLAPACK) && $(MAKE) blaslib
	cd $(CLAPACK) && $(MAKE) f2clib
	cd $(ADDITIONAL) && gcc -c -fPIC *.c && $(ARCH) $(ARCHFLAGS) $(DGELSLIB) *.o && ranlib $(DGELSLIB)

clean:
	cd $(CLAPACK) && $(MAKE) clean && rm -rf *~
	cd $(ADDITIONAL) && rm -rf *.o *~
	rm -rf $(BUILD) *~

distclean: clean

dist:
	echo "Not implemented"
