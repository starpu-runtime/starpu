#!/bin/sh
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
<<<<<<< HEAD
# Copyright (C) 2010-2011,2013,2015                      CNRS
# Copyright (C) 2009-2010,2012,2014                      Université de Bordeaux
=======
# Copyright (C) 2010,2011,2013,2015,2017                 CNRS
# Copyright (C) 2009,2010,2012,2014,2016,2020            Université de Bordeaux
>>>>>>> f2bea4ce9... Use $STARPU_LAUNCH in scripts
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
#!/bin/sh
ROOT=${0%.sh}
$STARPU_LAUNCH $ROOT "$@" > tasks_size_overhead.output
$ROOT.gp
gv tasks_size_overhead.eps
