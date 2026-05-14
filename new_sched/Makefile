# SGOC graph scheduler (loadable StarPU policy) + demo
CXX ?= c++
CXXSTD ?= -std=c++17
CXXFLAGS ?= -O0 -g -pthread -fPIC -D_GNU_SOURCE -D_XOPEN_SOURCE=700

GRAPH_SCHED_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
STARPU_SRC_ROOT := $(abspath $(GRAPH_SCHED_DIR)/..)
# After ./configure && make in STARPU_SRC_ROOT, starpu-1.4.pc appears under $(STARPU_BUILD)/lib/pkgconfig
STARPU_BUILD ?= $(abspath $(STARPU_SRC_ROOT)/build)

PKG ?= starpu-1.4

# $(shell pkg-config …) does not reliably see PKG_CONFIG_PATH unless exported; prepend in-tree StarPU build too.
export PKG_CONFIG_PATH := $(STARPU_BUILD)/lib/pkgconfig:$(PKG_CONFIG_PATH)

PKG_PC_CFLAGS := $(shell pkg-config --cflags $(PKG) 2>/dev/null)
PKG_PC_LIBS := $(shell pkg-config --libs $(PKG) 2>/dev/null)
PKG_CFLAGS := $(PKG_PC_CFLAGS) -I$(STARPU_SRC_ROOT)/include -pthread
PKG_LIBS   := $(PKG_PC_LIBS) -pthread

PKG_LIBDIR := $(shell pkg-config --variable=libdir $(PKG) 2>/dev/null)
STARPU_SO  := $(firstword $(wildcard $(PKG_LIBDIR)/libstarpu-1.4.so $(PKG_LIBDIR)/libstarpu.so $(PKG_LIBDIR)/libstarpu*.so))
# StarPU ≥ ~1.5: victim selector API (nm on the shared lib; override with make STARPUSGOC_HAS_VICTIM_SELECTOR=1).
STARPUSGOC_HAS_VICTIM_SELECTOR ?= $(shell if test -n "$(STARPU_SO)" && nm -D "$(STARPU_SO)" 2>/dev/null | grep -q ' starpu_data_register_victim_selector$$' && nm -D "$(STARPU_SO)" 2>/dev/null | grep -q ' starpu_data_get_node_data$$'; then echo 1; else echo 0; fi)

ifeq ($(PKG_PC_CFLAGS),)
$(warning pkg-config could not find $(PKG) (empty --cflags). Set PKG_CONFIG_PATH to the directory containing starpu-1.4.pc, or build StarPU so $(STARPU_BUILD)/lib/pkgconfig exists.)
endif

ALL_CXXFLAGS := $(CXXSTD) $(CXXFLAGS) $(PKG_CFLAGS) -DSTARPUSGOC_HAS_VICTIM_SELECTOR=$(STARPUSGOC_HAS_VICTIM_SELECTOR) $(EXTRA_CFLAGS)

LIB := libgraph_sgoc_sched.so
LIB_OBJS := libgraph_sgoc_sched.o graph_sched_offload.o graph_sched_policy_lookup.o graph_sched_sgoc_victim.o

DEMO := demo_graph_sched
DEMO_OBJS := main.o

all: $(LIB) $(DEMO)

build: all

$(LIB): $(LIB_OBJS)
	$(CXX) -shared -o $@ $^ $(PKG_LIBS)

$(DEMO): $(DEMO_OBJS) $(LIB)
	$(CXX) -o $@ $(DEMO_OBJS) -L. -lgraph_sgoc_sched $(PKG_LIBS)

libgraph_sgoc_sched.o: libgraph_sgoc_sched.cpp graph_sgoc.cpp graph_sched_internal.hpp graph_sgoc_bundle.inc graph_sched_pin_worker_extract.inc
	$(CXX) $(ALL_CXXFLAGS) -c libgraph_sgoc_sched.cpp -o $@

graph_sched_offload.o: graph_sched_offload.cpp graph_sched_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sched_offload.cpp -o $@

graph_sched_policy_lookup.o: graph_sched_policy_lookup.cpp graph_sched_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sched_policy_lookup.cpp -o $@

graph_sched_sgoc_victim.o: graph_sched_sgoc_victim.cpp graph_sched_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sched_sgoc_victim.cpp -o $@

main.o: main.cpp graph_sched.h
	$(CXX) $(ALL_CXXFLAGS) -c main.cpp -o $@

.PHONY: all build clean run run-fxt

clean:
	rm -f $(LIB_OBJS) $(DEMO_OBJS) $(LIB) $(DEMO)

run:	$(LIB) $(DEMO)
	@lb="$$(pkg-config --variable=libdir $(PKG) 2>/dev/null)"; \
	test -n "$$lb" || lb='$(PKG_LIBDIR)'; \
	LD_LIBRARY_PATH="$$lb:$$LD_LIBRARY_PATH:." STARPU_SCHED=sgoc STARPU_SCHED_LIB=./$(LIB) \
	STARPU_NCPU=1 STARPU_NCUDA=1 STARPU_SILENT=1 STARPU_GRAPH_SCHED_VERBOSE=6 STARPU_GRAPH_SCHED_CHECKPOINT_MAX=100 ./$(DEMO)

run-fxt:	$(LIB) $(DEMO)
	@mkdir -p traces
	@lb="$$(pkg-config --variable=libdir $(PKG) 2>/dev/null)"; \
	test -n "$$lb" || lb='$(PKG_LIBDIR)'; \
	LD_LIBRARY_PATH="$$lb:$$LD_LIBRARY_PATH:." STARPU_SCHED=sgoc STARPU_SCHED_LIB=./$(LIB) \
	STARPU_NCPU=1 STARPU_NCUDA=1 STARPU_GRAPH_SCHED_VERBOSE=1 STARPU_SILENT=1 \
	STARPU_FXT_TRACE=1 STARPU_FXT_PREFIX=$(CURDIR)/traces STARPU_GENERATE_TRACE=1 \
	./$(DEMO)
	@echo "Traces saved in $(CURDIR)/traces/"

run2:	$(DEMO)
	@lb="$$(pkg-config --variable=libdir $(PKG) 2>/dev/null)"; \
	test -n "$$lb" || lb='$(PKG_LIBDIR)'; \
	LD_LIBRARY_PATH="$$lb:$$LD_LIBRARY_PATH:." STARPU_NCPU=1 ./$(DEMO)
