include min-make.inc

all: clean install

install:
	mkdir -p $(BUILD);
	cd $(CLAPACK) && $(MAKE) -j5 blaslib
	cd $(CLAPACK) && $(MAKE) -j5 f2clib
	cd $(ADDITIONAL) && gcc -c -fPIC *.c && $(ARCH) $(ARCHFLAGS) $(DGELSLIB) *.o && ranlib $(DGELSLIB)

clean:
	cd $(CLAPACK) && $(MAKE) clean && rm -rf *~
	cd $(ADDITIONAL) && rm -rf *.o *~
	rm -rf $(BUILD) *~
