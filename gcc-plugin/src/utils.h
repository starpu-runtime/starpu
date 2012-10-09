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

/* GCC 4.7 requires compilation with `g++', and C++ lacks a number of GNU C
   features, so work around that.  */

#ifdef __cplusplus

/* G++ doesn't implement nested functions, so use C++11 lambdas instead.  */

# include <functional>

# define local_define(ret, name, parms)     auto name = [=]parms
# define function_parm(ret, name, parms)    std::function<ret parms> name

/* G++ lacks designated initializers.  */
# define designated_field_init(name, value) value /* XXX: cross fingers */

#else  /* !__cplusplus */

/* GNU C nested functions.  */

# define local_define(ret, name, parms)	    ret name parms
# define function_parm(ret, name, parms)    ret (*name) parms

/* Designated field initializer.  */

# define designated_field_init(name, value) .name = value

#endif /* !__cplusplus */


extern bool verbose_output_p;
extern const char task_attribute_name[];
extern bool task_p (const_tree decl);


#include <tree.h>

/* This declaration is from `c-tree.h', but that header doesn't get
   installed.  */

extern tree xref_tag (enum tree_code, tree);


/* Don't warn about the unused `gcc_version' variable, from
   <plugin-version.h>.  */

static const struct plugin_gcc_version *starpu_gcc_version
  __attribute__ ((__unused__)) = &gcc_version;
