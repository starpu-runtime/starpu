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

#include <starpu-gcc/config.h>

#include <unistd.h>

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


/* List and vector utilities, à la SRFI-1.  */

extern tree chain_trees (tree t, ...)
  __attribute__ ((sentinel));

extern tree filter (function_parm (bool, pred, (const_tree)), tree t);
extern tree list_remove (function_parm (bool, pred, (const_tree)), tree t);
extern tree map (function_parm (tree, func, (const_tree)), tree t);
extern void for_each (function_parm (void, func, (tree)), tree t);
extern size_t count (function_parm (bool, pred, (const_tree)), const_tree t);


/* Compatibility tricks & workarounds.  */

#include <tree.h>
#include <vec.h>

/* This declaration is from `c-tree.h', but that header doesn't get
   installed.  */

extern tree xref_tag (enum tree_code, tree);

#if !HAVE_DECL_BUILTIN_DECL_EXPLICIT

/* This function was introduced in GCC 4.7 as a replacement for the
   `built_in_decls' array.  */

static inline tree
builtin_decl_explicit (enum built_in_function fncode)
{
  return built_in_decls[fncode];
}

#endif

#if !HAVE_DECL_BUILD_CALL_EXPR_LOC_ARRAY

extern tree build_call_expr_loc_array (location_t loc, tree fndecl, int n,
				       tree *argarray);

#endif

#if !HAVE_DECL_BUILD_CALL_EXPR_LOC_VEC

extern tree build_call_expr_loc_vec (location_t loc, tree fndecl,
				     VEC(tree,gc) *vec);

#endif

#if !HAVE_DECL_BUILD_ZERO_CST

extern tree build_zero_cst (tree type);

#endif

#ifndef VEC_qsort

/* This macro is missing in GCC 4.5.  */

# define VEC_qsort(T,V,CMP) qsort(VEC_address (T,V), VEC_length(T,V),	\
				  sizeof (T), CMP)

#endif


/* Helpers.  */

extern bool verbose_output_p;

extern tree build_pointer_lookup (tree pointer);
extern tree build_starpu_error_string (tree error_var);
extern tree build_constructor_from_unsorted_list (tree type, tree vals);
extern tree read_pragma_expressions (const char *pragma, location_t loc);
extern tree type_decl_for_struct_tag (const char *tag);
extern tree build_function_arguments (tree fn);
extern tree build_error_statements (location_t, tree,
				    function_parm (tree, f, (tree)),
				    const char *, ...)
  __attribute__ ((format (printf, 4, 5)));

extern bool void_type_p (const_tree lst);
extern bool pointer_type_p (const_tree lst);

/* Lookup the StarPU function NAME in the global scope and store the result
   in VAR (this can't be done from `lower_starpu'.)  */

#define LOOKUP_STARPU_FUNCTION(var, name)				\
  if ((var) == NULL_TREE)						\
    {									\
      (var) = lookup_name (get_identifier (name));			\
      gcc_assert ((var) != NULL_TREE && TREE_CODE (var) == FUNCTION_DECL); \
    }


/* Don't warn about the unused `gcc_version' variable, from
   <plugin-version.h>.  */

static const struct plugin_gcc_version *starpu_gcc_version
  __attribute__ ((__unused__)) = &gcc_version;
