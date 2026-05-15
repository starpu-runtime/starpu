# SGOC graph scheduler (loadable StarPU policy)
CXX ?= c++
CXXSTD ?= -std=c++17
CXXFLAGS ?= -O0 -g -pthread -fPIC -D_GNU_SOURCE -D_XOPEN_SOURCE=700

GRAPH_SGOC_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
STARPU_SRC_ROOT := $(abspath $(GRAPH_SGOC_DIR)/..)
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
LIB_OBJS := libgraph_sgoc_sched.o graph_sgoc_dag.o graph_sgoc_parse.o graph_sgoc_topo.o graph_sgoc_mm_plan.o \
	graph_sgoc_checkpoint_wrr.o graph_sgoc_checkpoint.o graph_sgoc_capture_linearize.o graph_sgoc_flush.o graph_sgoc_capture.o graph_sgoc_runtime.o \
	graph_sgoc_env.o graph_sgoc_pinned_worker.o graph_sgoc_offload.o graph_sgoc_policy_lookup.o graph_sgoc_victim.o

SELF_TEST := graph_sgoc_self_test

all: $(LIB)

build: all

$(LIB): $(LIB_OBJS)
	$(CXX) -shared -o $@ $^ $(PKG_LIBS)

libgraph_sgoc_sched.o: libgraph_sgoc_sched.cpp graph_sgoc_internal.hpp graph_sgoc_types.hpp graph_sgoc_env.hpp
	$(CXX) $(ALL_CXXFLAGS) -c libgraph_sgoc_sched.cpp -o $@

GRAPH_SGOC_HDRS := graph_sgoc_internal.hpp graph_sgoc_types.hpp graph_sgoc_common.hpp graph_sgoc_bundle_detail.hpp graph_sgoc_bundle_checkpoint.hpp \
	graph_sgoc_bundle_dag.hpp graph_sgoc_bundle_flush.hpp graph_sgoc_bundle_mm.hpp graph_sgoc_bundle_parse.hpp \
	graph_sgoc_bundle_topo.hpp graph_sgoc_env.hpp graph_sgoc_timing.hpp

graph_sgoc_dag.o: graph_sgoc_dag.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_dag.cpp -o $@

graph_sgoc_parse.o: graph_sgoc_parse.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_parse.cpp -o $@

graph_sgoc_topo.o: graph_sgoc_topo.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_topo.cpp -o $@

graph_sgoc_mm_plan.o: graph_sgoc_mm_plan.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_mm_plan.cpp -o $@

graph_sgoc_checkpoint_wrr.o: graph_sgoc_checkpoint_wrr.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_checkpoint_wrr.cpp -o $@

graph_sgoc_checkpoint.o: graph_sgoc_checkpoint.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_checkpoint.cpp -o $@

graph_sgoc_capture_linearize.o: graph_sgoc_capture_linearize.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_capture_linearize.cpp -o $@

graph_sgoc_flush.o: graph_sgoc_flush.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_flush.cpp -o $@

graph_sgoc_capture.o: graph_sgoc_capture.cpp graph_sgoc_internal.hpp graph_sgoc_env.hpp graph_sgoc_timing.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_capture.cpp -o $@

graph_sgoc_runtime.o: graph_sgoc_runtime.cpp graph_sgoc_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_runtime.cpp -o $@

graph_sgoc_env.o: graph_sgoc_env.cpp graph_sgoc_env.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_env.cpp -o $@

graph_sgoc_pinned_worker.o: graph_sgoc_pinned_worker.cpp graph_sgoc_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_pinned_worker.cpp -o $@

graph_sgoc_offload.o: graph_sgoc_offload.cpp graph_sgoc_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_offload.cpp -o $@

graph_sgoc_policy_lookup.o: graph_sgoc_policy_lookup.cpp graph_sgoc_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_policy_lookup.cpp -o $@

graph_sgoc_victim.o: graph_sgoc_victim.cpp graph_sgoc_internal.hpp
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_victim.cpp -o $@

graph_sgoc_self_test.o: graph_sgoc_self_test.cpp $(GRAPH_SGOC_HDRS)
	$(CXX) $(ALL_CXXFLAGS) -c graph_sgoc_self_test.cpp -o $@

$(SELF_TEST): graph_sgoc_self_test.o graph_sgoc_topo.o graph_sgoc_dag.o graph_sgoc_env.o graph_sgoc_checkpoint_wrr.o \
		graph_sgoc_parse.o
	$(CXX) -o $@ $^ $(PKG_LIBS)

.PHONY: all build clean check

check: $(SELF_TEST)
	@lb="$$(pkg-config --variable=libdir $(PKG) 2>/dev/null)"; \
	test -n "$$lb" || lb='$(PKG_LIBDIR)'; \
	LD_LIBRARY_PATH="$$lb:$$LD_LIBRARY_PATH" ./$(SELF_TEST)

clean:
	rm -f $(LIB_OBJS) $(LIB) $(SELF_TEST) graph_sgoc_self_test.o
