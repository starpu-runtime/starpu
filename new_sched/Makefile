# graph_recorder loadable scheduler + demo

PKG ?= starpu-1.4
CXX ?= c++
CXXSTD ?= -std=c++17
CXXFLAGS ?= -O0 -g -pthread -fPIC -D_GNU_SOURCE -D_XOPEN_SOURCE=700

# Use pkg-config for StarPU
# Use parent tree's src and build for headers (when building from starpu-sched)
# After `make install` from the StarPU build dir, pkg-config supplies include paths.
# -I../include picks up starpu_graph_recorder.h from the same source tree when needed.
PKG_CFLAGS := $(shell pkg-config --cflags $(PKG) 2>/dev/null) -I../include -pthread
PKG_LIBS   := $(shell pkg-config --libs $(PKG) 2>/dev/null) -pthread

ifeq ($(PKG_CFLAGS),)
$(warning pkg-config could not find $(PKG). Ensure StarPU is installed and pkg-config path is set.)
endif

# Allow users to pass EXTRA_CFLAGS (optional)
ALL_CXXFLAGS := $(CXXSTD) $(CXXFLAGS) $(PKG_CFLAGS) $(EXTRA_CFLAGS)

LIB := libgraph_recorder_sched.so
LIB_OBJS := libgraph_sched.o

DEMO := demo_graph_sched
DEMO_OBJS := main.o

all: $(LIB) $(DEMO)

$(LIB): $(LIB_OBJS)
	$(CXX) -shared -o $@ $^ $(PKG_LIBS)

$(DEMO): $(DEMO_OBJS) $(LIB)
	$(CXX) -o $@ $(DEMO_OBJS) -L. -lgraph_recorder_sched $(PKG_LIBS)

%.o: %.cpp
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

.PHONY: clean run run-fxt

clean:
	rm -f $(LIB_OBJS) $(DEMO_OBJS) $(LIB) $(DEMO)

run:	$(LIB) $(DEMO)
	LD_LIBRARY_PATH=. STARPU_SCHED=graph_recorder STARPU_SCHED_LIB=./$(LIB) STARPU_NCPU=1 \
	STARPU_SILENT=1 STARPU_GRAPH_SCHED_VERBOSE=2 ./$(DEMO)

# Run with FxT traces enabled; traces stored in new_sched/traces/
# Requires StarPU built with --with-fxt. Use starpu_fxt_tool to convert raw traces.
run-fxt:	$(LIB) $(DEMO)
	@mkdir -p traces
	LD_LIBRARY_PATH=. STARPU_SCHED=graph_recorder STARPU_SCHED_LIB=./$(LIB) STARPU_NCPU=1 \
	STARPU_GRAPH_SCHED_VERBOSE=1 STARPU_SILENT=1 \
	STARPU_FXT_TRACE=1 STARPU_FXT_PREFIX=$(CURDIR)/traces STARPU_GENERATE_TRACE=1 \
	./$(DEMO)
	@echo "Traces saved in $(CURDIR)/traces/"

run2:	$(DEMO)
	@lb="$$(pkg-config --variable=libdir $(PKG) 2>/dev/null)"; \
	LD_LIBRARY_PATH="$$lb:$$LD_LIBRARY_PATH:." STARPU_NCPU=1 ./$(DEMO)
