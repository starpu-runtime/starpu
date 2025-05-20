<!---
 StarPU --- Runtime system for heterogeneous multicore architectures.

 Copyright (C) 2009-2025    University of Bordeaux, CNRS (LaBRI UMR 5800), Inria

 StarPU is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or (at
 your option) any later version.

 StarPU is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 See the GNU Lesser General Public License in COPYING.LGPL for more details.
-->

## Directory structure

The directory structure is as follows:
- `src`        : internal source for StarPU
- `include`    : public API
- `tests`      : unitary tests
- `examples`   : examples using StarPU
- `doc`        : documentation for StarPU
- `tools`      : tools for StarPU

StarPU extensions have their own directory (`src`/`include`/`tests`/`examples`) structure:

- `mpi`            : The MPI support
- `socl`           : the StarPU OpenCL-compatible interface
- `sc_hypervisor`  : The Scheduling Context Hypervisor
- `starpufft`      : The FFT support
- `eclipse-plugin` : The Eclipse Plugin
- `starpupy`       : The StarPU Python Interface
- `starpurm`       : The StarPU Resource Manager

Some directories contain only build system details:
- `build-aux`
- `m4`
- `autom4te.cache`

## Developer Warnings

They are enabled only if the `STARPU_DEVEL` environment variable is
defined to a non-empty value, when calling `configure`.

## Tests

Please do try `make check`, at least with `./configure --enable-quick-check`

If a test fails, you can run it specifically again with

```shell
make check TESTS=the_test
```

You can also re-run only the failing tests with

```shell
make recheck
```

## Naming Conventions

- Prefix names of public objects (types, functions, etc.) with `starpu_`

- Prefix names of internal objects (types, functions, etc.) with `_starpu_`

- Names for qualified types (`struct`, `union`, `enum`) do not end with `_t`, `_s` or similar.
  Use `_t` only for typedef types, such as opaque public types, e.g

```C
typedef struct _starpu_data_state* starpu_data_handle_t;
```
or
```C
typedef uint64_t starpu_tag_t;
```

- When a variable can only take a finite set of values, use an `enum`
  type instead of defining macros for each of the values.


##  Coding Style

- Curly braces always go on a new line

## Error handling

- Use `STARPU_ABORT()` for catastrophic errors, from which StarPU will never
  recover.

```C
switch (node_kind)
{
	case STARPU_CPU_RAM:
		do_stg();
		break;
	...
	default:
		/* We cannot be here */
		STARPU_ABORT();
}
```

- Use `STARPU_ASSERT()` to run checks that are very likely to succeed, but still
  are useful for debugging purposes. It should be OK to disable them with
  `--enable-fast`.

```C
STARPU_ASSERT(j->terminated != 0)
```

- Use `STARPU_ASSERT_MSG()` to run checks that might not succeed, and notably due
  to application programming error. The additional message parameter should
  guide the programmer into fixing their error.

## Documentation

When adding a feature, we want four kinds of documentation:

- Announcing the feature in `ChangeLog`.

- At least one working example in `examples/`, or at least a working test in
  `tests/`. Ideally enough examples and tests to cover all the various features.

  The test should include a comment (after `#includes`) to explain what it tests.

- A section in the Doxygen documentation, that explains in which case the
  feature is useful and how to use it, and points to the abovementioned
  example/test.

  It should cover all aspects of the feature, so programmers don't have to look
  into the .h file or reference documentation to discover features. It however
  does not need to dive into all details, that can be provided in the next
  documentation.

- Doxygen comments along the declarations in the .h file. These should document
  each macro, enum, function, function parameter, flag, etc. And refer to the
  abovementioned section so that somebody who finds some function/macro/etc. can
  easily know what that is all about.

## Makefile.am

Dependency libraries are appended to `LIBS`.
Only real `LDFLAGS` such as `-no-undefined` go to `LDFLAGS`.

If a program foo needs more libraries, it can put then in `foo_LDADD`.

(No, `AM_LDADD` does not exist)

All install rules must use `$(DESTDIR)` so that

```shell
./configure --prefix=/usr && make && make install DESTDIR=/tmp/foobar
```

can properly work, as it is used by distributions. That can easily be checked by
*not* running it as root.

## Writing a new driver

Writing a new driver is essentially:

- Creating an `src/drivers/yourdriver/` and adding it to `src/Makefile.am`

  You can pick up `src/drivers/cuda/driver_cuda0.c` as an example of very basic driver which
  should be relatively easy to get working. Once you have it working you can
  try to get inspiration from `src/drivers/cuda/driver_cuda1.c` to implement
  asynchronous data and kernel execution.

- Adding fields in `struct starpu_conf` and `struct starpu_codelet`.

- Adding cases in `src/core/task.c`, look for `_CUDA` for an example.

- Adding initialization calls in `src/core/topology.c`, look for `_CUDA` for an example.

- Adding cases in `src/core/worker.c`, look for `_CUDA` for an example.

- Adding the case in `src/datawizard/reduction.c`, look for `_CUDA` for an example.

- There are a few "Driver porters" notes in the code.

- TODO: task & bus performance model

  For now the simplest is not to implement performance models. We'll rework the
  support to make it very generic.

- Other places can be extended to add features: asynchronous data transfers,
  energy measurement, multiformat, memory mapping.

## Adding a new trace event

This consists in:

- Adding a new function in `src/profiling/starpu_tracing.h` and in `src/profiling/starpu_tracing.c`

- Calling this function in the wanted place in the runtime

Implementing this event with FxT consists in

- Adding a code number in `src/profiling/fxt/fxt.h`

- Adding the callable runtime macro in `src/profiling/fxt/fxt.h`

- Adding a paje state in `states_list` `src/debug/traces/starpu_fxt.c` and in
  `src/debug/traces/starpu_paje.c`

- Adding the management of the code in `_starpu_fxt_parse_new_file`, usually
  calling a function that does the actual paje state addition (a push/pop pair
  or two state sets)

A simple example can be found in [28740e7a91a2 ("Add a Parallel sync state")](https://gitlab.inria.fr/starpu/starpu/-/commit/28740e7a91a2d6b4879861734905db086674f5e3).

## Continuous Integration

Different tests are available with the [gitlab pipelines](https://gitlab.inria.fr/starpu/starpu/-/pipelines)

- The master branch is tested on each commit with a minimal set of
  tests (mainly compilation with different profiles, execution on a
  few specific configurations, and a chameleon test).
  This is the same for all branches having the commit [48f7aceae0](https://gitlab.inria.fr/starpu/starpu/-/commit/48f7aceae0916820432d01b25b54f74d45da3271)

- Once a day, benchmarks and coverage tests are automatically run
  against the master branch, though a scheduled pipeline. See the
  [schedule](https://gitlab.inria.fr/starpu/starpu/-/pipeline_schedules)

- Once a day, extended tests are automatically run against the master
  branch. See the
  [schedule](https://gitlab.inria.fr/starpu/starpu/-/pipeline_schedules)

- All merge requests are automatically tested at each update. A
  minimal set of tests is automatically started. Extended tests must
  be run manually, and *SHOULD BE* run at least once before the MR is
  approved.
