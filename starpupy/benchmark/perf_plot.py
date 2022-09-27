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
import json
import matplotlib.pyplot as plt
import test_handle_bench

file1 = open('handle_perf1.txt', 'r')
js1 = file1.read()
retfut_dict = json.loads(js1)
#print(retfut_dict)
program_submit1 = [x*1000 for x in retfut_dict['program_submit']]
program_await1 = [x*1000 for x in retfut_dict['program_await']]

file3 = open('handle_perf3.txt', 'r')
js3 = file3.read()
nostarpu_dict = json.loads(js3)
#print(nostarpu_dict)
program_submit3 = [x*1000 for x in nostarpu_dict['program_submit']]

file_std = open('handle_perf_std.txt', 'r')
js_std = file_std.read()
dict_std = json.loads(js_std)

file1.close()
file3.close()
file_std.close()

std11 = dict_std['list_std11']
std12 = dict_std['list_std12']
std3 = dict_std['list_std3']

plt.subplot(2, 1, 1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xscale("log")
plt.yscale("log")
plt.errorbar([i for i in test_handle_bench.list_size], program_submit1, yerr=std11, fmt='+-', ecolor='r', color='r', elinewidth=1, capsize=3, linewidth=1, label='using StarPU')
plt.errorbar([i for i in test_handle_bench.list_size], program_submit3, yerr=std3, fmt='+-',ecolor='y', color='y', elinewidth=1, capsize=3, linewidth=1, label='using numpy.add function')

plt.legend(loc='upper left', fontsize=15)
plt.xlabel("Numpy array size (# of elements)", fontsize=15)
plt.ylabel("Program execution time (ms)", fontsize=15)

plt.subplot(2, 1, 2)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xscale("log")
plt.yscale("log")
plt.errorbar([i for i in test_handle_bench.list_size], program_await1, yerr=std12, fmt='+-',ecolor='r', color='r', elinewidth=1, capsize=3, linewidth=1, label='using StarPU')

plt.legend(loc='upper left', fontsize=15)
plt.xlabel("Numpy array size (# of elements)", fontsize=15)
plt.ylabel("Program await time (ms)", fontsize=15)

plt.show()
#plt.savefig("starpupy_handle_perf.png")
#plt.savefig("starpupy_handle_perf.eps")

