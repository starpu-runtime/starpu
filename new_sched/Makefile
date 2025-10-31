# Standalone build for graph-based scheduler example

PKG ?= starpu-1.4
CXX ?= c++
CXXSTD ?= -std=c++17
CXXFLAGS ?= -O2 -g -pthread -fPIC -D_GNU_SOURCE -D_XOPEN_SOURCE=700

# Use pkg-config for StarPU
PKG_CFLAGS := $(shell pkg-config --cflags $(PKG) 2>/dev/null)
PKG_LIBS   := $(shell pkg-config --libs $(PKG) 2>/dev/null) -pthread

ifeq ($(PKG_CFLAGS),)
$(warning pkg-config could not find $(PKG). Ensure StarPU is installed and pkg-config path is set.)
endif

# Allow users to pass EXTRA_CFLAGS (optional)
ALL_CXXFLAGS := $(CXXSTD) $(CXXFLAGS) $(PKG_CFLAGS) $(EXTRA_CFLAGS)

LIB := libgraph_standalone_sched.so
LIB_OBJS := libgraph_sched.o

DEMO := demo_graph_sched
DEMO_OBJS := main.o

all: $(LIB) $(DEMO)

$(LIB): $(LIB_OBJS)
	$(CXX) -shared -o $@ $^ $(PKG_LIBS)

$(DEMO): $(DEMO_OBJS) $(LIB)
	$(CXX) -o $@ $(DEMO_OBJS) -L. -lgraph_standalone_sched $(PKG_LIBS)

%.o: %.cpp
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

.PHONY: clean run

clean:
	rm -f $(LIB_OBJS) $(DEMO_OBJS) $(LIB) $(DEMO)

run:	$(LIB) $(DEMO)
	STARPU_SCHED=graph_standalone STARPU_SCHED_LIB=./$(LIB) ./$(DEMO)

run2:	$(DEMO)
	./$(DEMO)
