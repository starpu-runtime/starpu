---
name: Bug report
about: Create a report to help us improve starpu
title: ''
labels: ''
assignees: ''

---

### Steps to reproduce

Please describe how you make the issue happen, so we can reproduce it.

### Obtained behavior

Please describe the result of your actions, and notably what you got that you didn't expect.

If you get a segfault or assertion failure, please run the program in a debugger and send a backtrace (`bt full`). If you are trying a starpu example, you may need to run `gdb` through `libtool --mode=execute gdb` so it can find libraries etc.

### Expected behavior

Please describe the result that you expected instead.

### Configuration

The `configure` line you used.

### Configuration result

Please attach the `config.log` file from the build tree.

### Distribution

Its type and version

### Version of StarPU

The tarball version, or the git branch hash.

### Version of GPU drivers

If you are using CUDA/OpenCL/HIP, the version being used.
