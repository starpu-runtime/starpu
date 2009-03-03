#NVCCFLAGS+= -deviceemu
#CXXFLAGS+=  -D__DEVICE_EMULATION__

all: 

%.cubin: %.cu
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

clean:
	@rm -f *.o *.cu_o *.cpp_o *.cubin *.linkinfo
