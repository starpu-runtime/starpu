# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2019-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#


######################
##### Examples ######
######################

4 examples are provided to show the use of the different SLiC interfaces:

- max_fpga_basic_static.c lets SLiC initialize the maxeler stack itself. This
  is a very simple interface but does not allow for multiple fpga support.

- max_fpga_advanced_static.c loads the maxeler design itself. This is a bit
  more complex to call, but allows for multiple fpga support.

- max_fpga_dynamic.c achieves the same as max_fpga_advanced_static.c, but using
  the dynamic interface.

- max_fpga_mux.c goes one step further by making input/output on the CPU or
  local memory at will.


######################
##### Maxeler  ######
######################
$ export XILINXD_LICENSE_FILE=2100@jumax
$ module load vivado maxcompiler
$ module load devtoolset/8


The Makefiles then build the program automatically. They do the equivalent of
the following, written here only for information:

Building the JAVA program: (for kernel and Manager (.maxj))

$ cd starpu/tests/
$ maxjc -1.7 -cp $MAXCLASSPATH fpga

Running the Java program to generate a DFE implementation (a .max file)
that can be called from a StarPU/FPGA application and slic headers
(.h) for simulation:

$ java -XX:+UseSerialGC -Xmx2048m -cp $MAXCLASSPATH:. fpga.MyTasksManager DFEModel=MAIA maxFileName=MyTasks target=DFE_SIM

$ cp MyTasks_MAX5C_DFE_SIM/results/*{.max,.h} fpga

$ cd fpga

Building the slic object file (simulation):

$ sliccompile MyTasks.max



Once built, to start the simulation:

$ maxcompilersim -c LIMA -n $USER-MyTasks restart
$ export LD_LIBRARY_PATH=$MAXELEROSDIR/lib:$LD_LIBRARY_PATH
$ export SLIC_CONF="use_simulation=$USER-MyTasks"

PS: To stop simulation

$ maxcompilersim -c LIMA -n $USER-MyTasks stop


#################################
##### StarPU with Maxeler  ######
#################################

$ ./autogen.sh
$ ../configure --prefix=$PWD/../install
$ make

By default they are built for simulation (target DFE_SIM). To build for native
execution, use instead:

make MAX_TARGET=DFE

To test the code (.c):
$ ./tests/fpga/max_fpga
