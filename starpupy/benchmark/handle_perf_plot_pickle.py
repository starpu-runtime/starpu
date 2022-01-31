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

num = 1000000
listX = [10, 100, 1000, 10000, 100000, 1000000]
list_size = []
for x in listX:
	for X in range(x, x*10, x):
		list_size.append(X)
list_size.append(10000000)
list_size.append(20000000)
list_size.append(30000000)
list_size.append(40000000)
list_size.append(50000000)
#print(list_size)

file1 = open('handle_perf1.txt', 'r') 
js1 = file1.read()
withhandle_dict = json.loads(js1)   
#print(withhandle_dict)
program_submit1 = withhandle_dict['program_submit']
program_await1 = withhandle_dict['program_await']

file2 = open('handle_perf2.txt', 'r') 
js2 = file2.read()
nohandle_dict = json.loads(js2)   
#print(nohandle_dict)
program_submit2 = nohandle_dict['program_submit']
program_await2 = nohandle_dict['program_await']

file3 = open('handle_perf3.txt', 'r') 
js3 = file3.read()
nostarpu_dict = json.loads(js3)   
#print(nostarpu_dict)
program_submit3 = nostarpu_dict['program_submit']

file_std = open('handle_perf_std.txt', 'r') 
js_std = file_std.read()
dict_std = json.loads(js_std)

std11 = dict_std['list_std11']
std12 = dict_std['list_std12']
std21 = dict_std['list_std21']
std22 = dict_std['list_std22']
std3 = dict_std['list_std3']

plt.subplot(2, 1, 1)
plt.xscale("log")
plt.yscale("log")
plt.errorbar([i/num for i in list_size], program_submit1, yerr=std11, fmt='+-', ecolor='r', color='r', elinewidth=1, capsize=3, linewidth=1, label='using virtually shared memory manager')
plt.errorbar([i/num for i in list_size], program_submit2, yerr=std21, fmt='+-', ecolor='b', color='b', elinewidth=1, capsize=3, linewidth=1, label='without using virtually shared memory manager')
plt.errorbar([i/num for i in list_size], program_submit3, yerr=std3, fmt='+-',ecolor='y', color='y', elinewidth=1, capsize=3, linewidth=1, label='without using StarPU task submitting')

plt.legend(loc='upper left')
plt.xlabel("Numpy array size (MB)")
plt.ylabel("Program execution time (s)")


plt.subplot(2, 1, 2)
plt.xscale("log")
plt.yscale("log")
plt.errorbar([i/num for i in list_size], program_await1, yerr=std12, fmt='+-',ecolor='r', color='r', elinewidth=1, capsize=3, linewidth=1, label='using virtually shared memory manager')
plt.errorbar([i/num for i in list_size], program_await2, yerr=std22, fmt='+-',ecolor='b', color='b', elinewidth=1, capsize=3, linewidth=1, label='without using virtually shared memory manager')

plt.legend(loc='upper left')
plt.xlabel("Numpy array size (MB)")
plt.ylabel("Program await time (s)")

plt.show()

file1.close()
file2.close()
file3.close()
file_std.close()