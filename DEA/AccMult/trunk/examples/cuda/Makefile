.PHONY: cutil

export CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
export CUDAINSTALLDIR=/usr/local/cuda/


CXXFLAGS += -I$(CUDAINSTALLDIR)/include -Wall
CXXFLAGS += -I$(CUDAINSTALLDIR)/include
CXXFLAGS += -I/home/gonnet/NVIDIA_CUDA_SDK/common/inc/

UTILSRCDIR=/home/gonnet/NVIDIA_CUDA_SDK/common/src/
CUTILSRCDIR=/home/gonnet/NVIDIA_CUDA_SDK/common/src/
SDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDADIR=/usr/local/cuda/

LDXXFLAGS+= -lcuda -lcudart -L/usr/local/cuda/lib

LDXXFLAGS += -L$(SDKDIR)/lib
LDXXFLAGS += -L$(CUDADIR)/lib -lcudart -lncurses #-lcuda
LDXXFLAGS += -lGL -lGLU

#NVCCFLAGS+= -deviceemu
#CXXFLAGS+=  -D__DEVICE_EMULATION__

all: test_spmv

test_spmv: cutil test_spmv.cu_o
	$(CXX) $(CXXFLAGS) -fPIC  $(LDXXFLAGS) -o test_spmv test_spmv.cu_o cutil.cpp_o stopwatch.cpp_o stopwatch_linux.cpp_o multithreading.cpp_o  bank_checker.cpp_o cmd_arg_reader.cpp_o

test_spmv.cu_o: test_spmv.cu spmv.cu
	nvcc $(CXXFLAGS) $(NVCCFLAGS) --compiler-options -fno-strict-aliasing  -DUNIX -O3 -o test_spmv.cu_o -c test_spmv.cu

# that's temporaty ... 
cutil:
	$(CXX) $(CXXFLAGS)   -o bank_checker.cpp_o -c $(CUTILSRCDIR)/bank_checker.cpp
	$(CXX) $(CXXFLAGS)   -o cmd_arg_reader.cpp_o -c $(CUTILSRCDIR)/cmd_arg_reader.cpp
	$(CXX) $(CXXFLAGS)   -o cutil.cpp_o -c $(CUTILSRCDIR)/cutil.cpp
	$(CXX) $(CXXFLAGS)   -o stopwatch.cpp_o -c $(CUTILSRCDIR)/stopwatch.cpp
	$(CXX) $(CXXFLAGS)   -o stopwatch_linux.cpp_o -c $(CUTILSRCDIR)/stopwatch_linux.cpp
	$(CXX) $(CXXFLAGS)   -o multithreading.cpp_o -c $(CUTILSRCDIR)/multithreading.cpp

clean:
	rm -f *.o *.cu_o *.cpp_o
	rm -f test_spmv
