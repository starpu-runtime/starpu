CFLAGS += $(shell pkg-config --cflags starpu-1.0) -g -O0 -Wall -Werror
LDFLAGS += $(shell pkg-config --libs starpu-1.0) -g -O0 -Wall -Werror
 
all: bfs
 
bfs : bfs.o bfs_omp_func.o
	g++ bfs.o bfs_omp_func.o $(CFLAGS) $(LDFLAGS) -fopenmp -O3 -o bfs
bfs.o : bfs.cpp
	g++ bfs.cpp $(CFLAGS) -fopenmp -O3 -c -o bfs.o
bfs_omp_func.o : ./bfs_func/bfs_omp_func.cpp
	g++ ./bfs_func/bfs_omp_func.cpp $(CFLAGS) -fopenmp -O3 -c -o bfs_omp_func.o
 
clean:
	rm -f bfs *.o *~
