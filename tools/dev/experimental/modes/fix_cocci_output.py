#!/usr/bin/env python3
# coding=utf-8
#
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2012-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import sys
import re

# XXX Could be something else, like a file.
output = sys.stdout

def get_indentation_level(s):
	cnt = 0
	for c in s:
		if c != '\t':
			return -1
		cnt = cnt + 1

	return cnt

def fix(f):
	modes = {}
	trailing_comma = False
	indentation_level = -1
	for line in f.readlines():
		# This regexp could be more precise, but it should be good
		# enough
		regexp = "((\s)*)\.modes\[(\d)+\](\s)*=(\s)*(.*)"
		m = re.search(regexp, line)
		if not m:
			if modes:
				output.write("".join(["\t" for i in range(indentation_level)]))
				output.write(".modes = { ")
				idx = 0
				while modes.get(str(idx)):
					if idx != 0:
						output.write(", ")
					output.write(modes[str(idx)])
					idx = idx+1
				if trailing_comma:
					output.write(" },\n")
				else:
					output.write(" }\n")

				# Resetting these..
				modes.clear()
				trailing_comma = False
				indentation_level = -1
			output.write(line)
		else:
			idx = m.group(3)
			mode = m.group(6)

			# Remove traling comma
			if mode[-1] == ',':
				mode = mode[:-1]
				# This is the last mode for this 
				# codelet. Was this also the last
				# field ?
				if int(idx) == 0:
					trailing_comma = True

			# Try and guess the level of indentation
			if int(idx) == 0:
				s = m.group(1)
				indentation_level = get_indentation_level(s)

			modes[idx] = mode

def fix_file(filename):
	with open(filename, 'r') as f:
		fix(f)


def usage():
	print "%s <filename>" % sys.argv[0]
	sys.exit(1)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage()
		sys.exit(1)

	fix_file(sys.argv[1])
