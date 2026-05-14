# new_sched — SGOC graph scheduler demo

Loadable StarPU policy **`sgoc`** (`libgraph_sgoc_sched.so`): ready queue for the pinned CUDA worker, plus `starpu_graph_capture` hooks (`starpu_graph_sched_graph_recording_begin` / `end` in `graph_sched.h`).

Build:
```bash
cd new_sched
make   # needs pkg-config starpu-1.4 + a StarPU build/install that includes starpu_graph_capture.h
```

Artifacts: `libgraph_sgoc_sched.so`, `demo_graph_sched`.

Run with this policy:
```bash
STARPU_SCHED=sgoc STARPU_SCHED_LIB=./libgraph_sgoc_sched.so ./demo_graph_sched
```
Or `make run`. `make run2` uses StarPU’s built-in scheduler (the demo binary still links the graph library).

Env: `STARPU_GRAPH_SCHED_VERBOSE=1` prints init/deinit lines.
