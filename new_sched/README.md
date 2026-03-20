Standalone Graph-Inspired Scheduler (StarPU)

This is a standalone project that implements a graph-based scheduling policy inspired by StarPU's in-tree `graph_test` policy, but usable as an external application-defined scheduler.

Key points:
- Uses public StarPU scheduler API (`starpu_sched_policy`) to plug a custom scheduler.
- Records ready tasks into a bag until `do_schedule`, then partitions tasks into CPU/GPU queues using a simple device-power heuristic.
- Fully self-contained: does not rely on StarPU internal headers.

Build prerequisites:
- StarPU installed and discoverable via `pkg-config` (package name: `starpu-1.4`).

Build:
```bash
cd new_sched
make
```

Artifacts:
- Shared library: `libgraph_standalone_sched.so` (loadable StarPU scheduler)
- Demo program: `demo_graph_sched`

Run the demo:
```bash
./demo_graph_sched
```

Environment variables:
- `STARPU_SCHED` should be unset or empty when running, since this program sets its own policy.
- `STARPU_SILENT=1` silences progress prints from this example.
- `STARPU_GRAPH_SCHED_VERBOSE` for the loadable policy: `0` none; `1` init/deinit; `2` push/pop/post-exec + checkpoint lines; `3` submit hook + `do_schedule` one-liner (total/ready tasks); `4` full `do_schedule` listing (see `graph_sched.h`).


