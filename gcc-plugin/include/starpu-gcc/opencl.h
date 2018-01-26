/* GCC-StarPU
   Copyright (C) 2012 Inria

   GCC-StarPU is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GCC-StarPU is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GCC-StarPU.  If not, see <http://www.gnu.org/licenses/>.  */

#pragma once

#include <gcc-plugin.h>
#include <tree.h>
#include <cpplib.h>

#include <starpu-gcc/utils.h>

extern tree opencl_include_dirs;

extern void handle_pragma_opencl (struct cpp_reader *reader);
extern void validate_opencl_argument_type (location_t loc, const_tree type);
