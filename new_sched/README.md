graph_recorder (StarPU skeleton)

Minimal loadable scheduler **`graph_recorder`**: FIFO queue of ready tasks, plus registration with StarPU’s **`starpu_graph_recorder`** extension (`starpu_graph_sched_graph_recording_begin` / `end` in `graph_sched.h`).

The previous checkpoint / TaskGraph implementation was removed on purpose; replace `libgraph_sched.cpp` with your own graph logic while keeping the hooks you need.

Build:
```bash
cd new_sched
make   # needs pkg-config starpu-1.4 + installed StarPU with starpu_graph_recorder.h
```

Artifacts: `libgraph_recorder_sched.so`, `demo_graph_sched`.

Run with this policy:
```bash
STARPU_SCHED=graph_recorder STARPU_SCHED_LIB=./libgraph_recorder_sched.so ./demo_graph_sched
```
Or `make run`. `make run2` uses StarPU’s built-in scheduler (library still linked for the demo binary).

Env: `STARPU_GRAPH_SCHED_VERBOSE=1` prints init/deinit lines.
