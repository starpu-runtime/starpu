# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

listX = [10, 100, 1000, 10000, 100000, 1000000]
#listX = [10, 100]
list_size = []
for x in listX:
	for X in range(x, x*10, x):
		list_size.append(X)
list_size.append(10000000)
list_size.append(20000000)
list_size.append(30000000)
list_size.append(40000000)
list_size.append(50000000)
#print("list of size is",list_size)

