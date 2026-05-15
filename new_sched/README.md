# new_sched — SGOC graph scheduler

Loadable StarPU policy **`sgoc`** (`libgraph_sgoc_sched.so`): ready queue for the pinned CUDA worker, plus `starpu_graph_capture` hooks (`starpu_graph_sched_graph_recording_begin` / `end` in `graph_sched.h`).

**Sources:** `graph_sched_types.hpp` (graph/MM/checkpoint data structures), `graph_sched_internal.hpp` (C++ API declarations; includes types), `graph_sgoc_common.hpp` and topic headers `graph_sgoc_bundle_{dag,parse,topo,mm,flush,checkpoint}.hpp` (split-TU glue; `graph_sgoc_bundle_detail.hpp` is an umbrella include), `libgraph_sgoc_sched.cpp` (StarPU policy), **graph algorithms:** `graph_sgoc_dag.cpp`, `graph_sgoc_parse.cpp`, `graph_sgoc_topo.cpp`, `graph_sgoc_mm_plan.cpp`, `graph_sgoc_checkpoint_wrr.cpp` (WRR graph surgery + clone/insert), `graph_sgoc_checkpoint.cpp` (checkpoint orchestration / apply-before-topo), `graph_sgoc_capture_linearize.cpp`, `graph_sgoc_flush.cpp`, `graph_sgoc_capture.cpp`, `graph_sgoc_runtime.cpp`, `graph_sgoc_env.cpp`, `graph_sched_pinned_worker.cpp`, `graph_sched_offload.cpp`, `graph_sched_policy_lookup.cpp`, `graph_sched_sgoc_victim.cpp`. Public C API: `graph_sched.h`.

**Conventions:** Cross-TU helpers used only by the split algorithm sources live in namespace `graph_sgoc_bundle` with existing `graph_sched_*` / `graph_sgoc_*` names (linkage seam, not a full rename). Policy-wide hooks remain at file scope in `graph_sched_internal.hpp`.

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
