# Orchestration Makefile for rebuilding StarPU + graph scheduler, NNTile
# (nntile_core + nntile_graph), and running the GPT-2 LMHead notebook workflow with small
# sequence/batch settings.
#
# This Makefile lives under starpu-sched/new_sched/nntile/. Paths default to:
#   GRAPH_SCHED_DIR = new_sched, STARPU_SRC = starpu-sched root, NNTILE_ROOT = sibling nntile/.
# Override if your layout differs:
#   make build NNTILE_ROOT=/path/to/nntile STARPU_SRC=/path/to/starpu
#
# Optional: STARPU_BUILD (default: $(STARPU_SRC)/build), NNTILE_BUILD, CMAKE.

THIS_DIR        := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GRAPH_SCHED_DIR ?= $(abspath $(THIS_DIR)/..)
STARPU_SRC      ?= $(abspath $(GRAPH_SCHED_DIR)/..)
NNTILE_ROOT     ?= $(abspath $(STARPU_SRC)/../nntile)
NNTILE_BUILD    ?= $(NNTILE_ROOT)/build

STARPU_BUILD    ?= $(STARPU_SRC)/build
# Prefer this tree’s libstarpu (e.g. graph capture for sgoc) over pkg-config when present.
STARPU_BUILD_LIBDIR ?= $(abspath $(STARPU_BUILD)/src/.libs)
STARPU_LD       ?= $$(pkg-config --variable=libdir starpu-1.4 2>/dev/null)

CMAKE           ?= cmake
GRAPH_SCHED_SO  := $(abspath $(GRAPH_SCHED_DIR)/libgraph_sgoc_sched.so)
GRAPH_SCHED_LD  := $(abspath $(GRAPH_SCHED_DIR))

NOTEBOOK_DIR    := $(NNTILE_ROOT)/notebooks
NNTILE_PYTHON   := $(NNTILE_BUILD)/wrappers/python
TRAIN_SEED      ?= 0

# Env vars from notebooks/gpt2_lmhead.ipynb (single line for one shell)
STARPU_ENV := OMP_NUM_THREADS=1 \
	LD_LIBRARY_PATH=$(STARPU_BUILD_LIBDIR):$(STARPU_LD):$(GRAPH_SCHED_LD):$$LD_LIBRARY_PATH \
	PYTHONPATH=$(NNTILE_PYTHON) \
	STARPU_NCPU=0 STARPU_NCUDA=1 STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1 \
	STARPU_PROFILING=1 STARPU_BUS_STATS=1 STARPU_PREFETCH=1 \
	STARPU_HOME=$(NOTEBOOK_DIR)/starpu \
	STARPU_PERF_MODEL_DIR=$(NOTEBOOK_DIR)/starpu/sampling \
	STARPU_PERF_MODEL_HOMOGENEOUS_CPU=1 STARPU_PERF_MODEL_HOMOGENEOUS_CUDA=1 \
	STARPU_HOSTNAME=GPT2_LMHead_example \
	STARPU_FXT_PREFIX=$(NOTEBOOK_DIR)/starpu/fxt \
	STARPU_GRAPH_SCHED_VERBOSE=3 \
	STARPU_LIMIT_CUDA_MEM=2000 \
	STARPU_GRAPH_SCHED_SGOC_MEM_DEBUG=1 \
	STARPU_GRAPH_SCHED_MM_ORDER_TRACE=1 \
	STARPU_GRAPH_SCHED_STARPU_MEM_AVAILABLE_FRACTION=1.0 \
	STARPU_GRAPH_SCHED_MM_EXECUTE_HINTS=1 \
	STARPU_GRAPH_SCHED_MEM_OFFLOAD_AUTO=1

# Max WRR graph checkpoints per sgoc flush (see graph_sched.h). Example: make run_graph CKPT_MAX=24
CKPT_MAX ?= 0

# Copy pure-Python graph scheduler helpers from NNTILE source into the build tree so ``make run`` picks up
# graph_recorder_sched + graph_capture_sched without a full CMake rebuild.
sync_nntile_graph_sched_py:
	@mkdir -p $(NNTILE_PYTHON)/nntile/graph_recorder_sched
	@cp -f $(NNTILE_ROOT)/wrappers/python/nntile/graph_recorder_sched/__init__.py \
		$(NNTILE_PYTHON)/nntile/graph_recorder_sched/__init__.py
	@mkdir -p $(NNTILE_PYTHON)/nntile/graph_capture_sched
	@cp -f $(NNTILE_ROOT)/wrappers/python/nntile/graph_capture_sched/__init__.py \
		$(NNTILE_PYTHON)/nntile/graph_capture_sched/__init__.py

# Training invocation from notebook §3.1, with requested hyperparameters
GPT2_TRAIN := python ../wrappers/python/examples/gpt2_lmhead_training.py \
	--restrict=cuda \
	--pretrained=local \
	--config-path=../wrappers/python/examples/gpt2_default_config.json \
	--save-checkpoint-path=.model/nntile_checkpoint.pt \
	--optimizer=adam \
	--lr=1e-4 \
	--dtype=fp32 \
	--nepochs=1 \
	--seq-len=1024 \
	--batch-size=4 \
	--minibatch-size=1 \
	--dataset-file=tinystories/train.bin \
	--seed=$(TRAIN_SEED)

.PHONY: build build_starpu install_starpu build_graph_sched configure_nntile \
	nntile_python nntile_core nntile_graph \
	data run run_dmdasd run_graph sync_nntile_graph_sched_py

# ---------------------------------------------------------------------------
# build: StarPU (compile + install) -> graph scheduler (in-tree, no install)
#        -> CMake configure NNTile (incremental) -> nntile_core + nntile_graph
# ---------------------------------------------------------------------------
build: build_starpu install_starpu build_graph_sched configure_nntile nntile_python

build_starpu:
	$(MAKE) -C $(STARPU_BUILD)

install_starpu:
	$(MAKE) -C $(STARPU_BUILD) install

build_graph_sched:
	@echo "Building SGOC graph scheduler ($(GRAPH_SCHED_DIR)) ..."
	@$(MAKE) -C $(GRAPH_SCHED_DIR) all

configure_nntile:
	$(CMAKE) -S $(NNTILE_ROOT) -B $(NNTILE_BUILD) -GNinja \
		-DCMAKE_CUDA_ARCHITECTURES=90 \
		-DCMAKE_PREFIX_PATH=$$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') \
		-DCMAKE_BUILD_TYPE=Release

# Pybind extensions into $(NNTILE_PYTHON)/nntile/ (core ops + graph API)
nntile_python: configure_nntile
	$(CMAKE) --build $(NNTILE_BUILD) --target nntile_core --target nntile_graph --parallel

nntile_core: configure_nntile
	$(CMAKE) --build $(NNTILE_BUILD) --target nntile_core --parallel

nntile_graph: configure_nntile
	$(CMAKE) --build $(NNTILE_BUILD) --target nntile_graph --parallel

# ---------------------------------------------------------------------------
# data: causal_lm_data_preparation.py (same script as gpt2_lmhead.ipynb)
# ---------------------------------------------------------------------------
data:
	cd $(NOTEBOOK_DIR) && python ../wrappers/python/examples/causal_lm_data_preparation.py \
		--hf-tokenizer=openai-community/gpt2 \
		--seq-len=1024 \
		--batch-size=4 \
		--dataset-select=100

# PYTHONPATH -> $(NNTILE_BUILD)/wrappers/python (via STARPU_ENV); default: graph scheduler
run: run_graph

run_dmdasd:
	cd $(NOTEBOOK_DIR) && $(STARPU_ENV) STARPU_SCHED=dmdasd $(GPT2_TRAIN)

run_darts:
	cd $(NOTEBOOK_DIR) && $(STARPU_ENV) STARPU_SCHED=darts $(GPT2_TRAIN)

# Loadable SGOC graph policy from $(GRAPH_SCHED_DIR) (see that tree's Makefile).
# SGOC pins recorded replay using STARPU_GRAPH_SCHED_WORKER (see starpu-sched new_sched).
# An inherited value (e.g. CPU:0 from your shell) is fatal if this StarPU run has no CPU workers
# (STARPU_NCPU=0, CPU driver disabled, etc.). STARPU_GRAPH_SCHED_WORKER= in STARPU_ENV clears that
# so the policy uses its default (CUDA:0 if a GPU exists, else CPU:0). Override on the make line if needed.
run_graph: sync_nntile_graph_sched_py
	@echo "new_sched/nntile: STARPU_SCHED_LIB=$(GRAPH_SCHED_SO) (rebuild with: make -C $(GRAPH_SCHED_DIR))" >&2
	cd $(NOTEBOOK_DIR) && $(STARPU_ENV) STARPU_GRAPH_SCHED_CHECKPOINT_MAX=$(CKPT_MAX) STARPU_SCHED=sgoc \
		STARPU_SCHED_LIB=$(GRAPH_SCHED_SO) \
		$(GPT2_TRAIN)
