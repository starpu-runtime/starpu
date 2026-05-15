# new_sched — SGOC graph scheduler

Loadable StarPU policy **`sgoc`** (`libgraph_sgoc_sched.so`): ready queue for the pinned CUDA worker, plus `starpu_graph_capture` hooks (`graph_sgoc_recording_begin` / `end` in `graph_sgoc.h`; legacy `starpu_graph_sched_graph_recording_*` aliases match the NNTile ctypes loader).

**Sources:** `graph_sgoc_types.hpp` (graph/MM/checkpoint data structures), `graph_sgoc_internal.hpp` (C++ API declarations; includes types), `graph_sgoc_common.hpp` and topic headers `graph_sgoc_bundle_{dag,parse,topo,mm,flush,checkpoint}.hpp` (split-TU glue; `graph_sgoc_bundle_detail.hpp` is an umbrella include), `libgraph_sgoc_sched.cpp` (StarPU policy), **graph algorithms:** `graph_sgoc_dag.cpp`, `graph_sgoc_parse.cpp`, `graph_sgoc_topo.cpp`, `graph_sgoc_mm_plan.cpp`, `graph_sgoc_checkpoint_wrr.cpp` (WRR graph surgery + clone/insert), `graph_sgoc_checkpoint.cpp` (checkpoint orchestration / apply-before-topo), `graph_sgoc_capture_linearize.cpp`, `graph_sgoc_flush.cpp`, `graph_sgoc_capture.cpp`, `graph_sgoc_runtime.cpp`, `graph_sgoc_env.cpp`, `graph_sgoc_pinned_worker.cpp`, `graph_sgoc_offload.cpp`, `graph_sgoc_policy_lookup.cpp`, `graph_sgoc_victim.cpp`. Public C API: `graph_sgoc.h`.

**Layout:** Types live in `graph_sgoc_types.hpp`. File-scope declarations shared across translation units are in `graph_sgoc_internal.hpp`. Cross-TU helpers used only by the split algorithm sources are grouped in namespace `graph_sgoc_bundle` (declarations in the `graph_sgoc_bundle_*.hpp` headers).

Build:
```bash
cd new_sched
make   # needs pkg-config starpu-1.4 + a StarPU build/install that includes starpu_graph_capture.h
```

Artifact: `libgraph_sgoc_sched.so`. Optional in-process checks: `make check` (builds `graph_sgoc_self_test`; needs the same `LD_LIBRARY_PATH` as a normal StarPU app—`make check` sets it from pkg-config).

**IDE / clangd:** Generate `compile_commands.json` from this directory, for example `bear -- make` (after `apt install bear` or equivalent) or `pip install compiledb && compiledb make`, then point your editor at `new_sched/compile_commands.json`.

Use this policy with any StarPU program that loads `libgraph_sgoc_sched.so`, for example:
```bash
export LD_LIBRARY_PATH="path/to/starpu/lib:${LD_LIBRARY_PATH}"
STARPU_SCHED=sgoc STARPU_SCHED_LIB=/absolute/path/to/libgraph_sgoc_sched.so your_program ...
```

Env: `STARPU_GRAPH_SCHED_VERBOSE=1` prints init/deinit lines.
