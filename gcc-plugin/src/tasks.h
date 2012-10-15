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

/* Various utilities.  */

#pragma once

#include <starpu-gcc-config.h>

#include <utils.h>
#include <starpu.h>


extern const char task_attribute_name[];
extern const char task_implementation_attribute_name[];
extern const char output_attribute_name[];

extern const char task_implementation_wrapper_attribute_name[];
extern const char task_implementation_list_attribute_name[];

extern bool task_p (const_tree decl);
extern bool task_implementation_p (const_tree decl);
extern int task_implementation_where (const_tree task_impl);
extern int task_implementation_target_to_int (const_tree target);
extern tree task_implementation_task (const_tree task_impl);
extern tree task_codelet_declaration (const_tree task_decl);
extern tree task_implementation_list (const_tree task_decl);
extern tree task_pointer_parameter_types (const_tree task_decl);
extern int task_where (const_tree task_decl);
extern tree task_implementation_wrapper (const_tree task_impl);
extern enum starpu_access_mode access_mode (const_tree type);
extern bool output_type_p (const_tree type);

extern tree codelet_type (void);
extern void taskify_function (tree fn);
extern tree build_codelet_identifier (tree task_decl);
extern tree build_codelet_declaration (tree task_decl);
extern tree build_codelet_initializer (tree task_decl);
extern tree declare_codelet (tree task_decl);
extern void define_task (tree task_decl);
extern void add_task_implementation (tree task_decl, tree fn,
				     const_tree where);
