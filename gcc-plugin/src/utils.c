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

#include <starpu-gcc/config.h>

/* We must include starpu.h here, otherwise gcc will complain about a poisoned
   malloc in xmmintrin.h.  */
#include <starpu.h>

#include <gcc-plugin.h>
#include <plugin-version.h>

#include <plugin.h>
#include <cpplib.h>
#include <tree.h>
#include <tree-iterator.h>
#include <gimple.h>

#ifdef HAVE_C_FAMILY_C_COMMON_H
# include <c-family/c-common.h>
#elif HAVE_C_COMMON_H
# include <c-common.h>
#endif

#include <starpu-gcc/utils.h>

/* Whether to enable verbose output.  */
bool verbose_output_p = false;


/* Various helpers.  */

/* Return a TYPE_DECL for the RECORD_TYPE with tag name TAG.  */

tree
type_decl_for_struct_tag (const char *tag)
{
  tree type_decl = xref_tag (RECORD_TYPE, get_identifier (tag));
  gcc_assert (type_decl != NULL_TREE
	      && TREE_CODE (type_decl) == RECORD_TYPE);

  /* `build_decl' expects a TYPE_DECL, so give it what it wants.  */

  type_decl = TYPE_STUB_DECL (type_decl);
  gcc_assert (type_decl != NULL && TREE_CODE (type_decl) == TYPE_DECL);

  return type_decl;
}

/* Given ERROR_VAR, an integer variable holding a StarPU error code, return
   statements that print out the error message returned by
   BUILD_ERROR_MESSAGE (ERROR_VAR) and abort.  */

tree
build_error_statements (location_t loc, tree error_var,
			function_parm (tree, build_error_message, (tree)),
			const char *fmt, ...)
{
  expanded_location xloc = expand_location (loc);

  tree print;
  char *str, *fmt_long;
  va_list args;

  va_start (args, fmt);

  /* Build a longer format.  Since FMT itself contains % escapes, this needs
     to be done in two steps.  */

  vasprintf (&str, fmt, args);

  if (error_var != NULL_TREE)
    {
      /* ERROR_VAR is an error code.  */
      gcc_assert (TREE_CODE (error_var) == VAR_DECL
		  && TREE_TYPE (error_var) == integer_type_node);

      asprintf (&fmt_long, "%s:%d: error: %s: %%s\n",
		xloc.file, xloc.line, str);

      print =
	build_call_expr (builtin_decl_explicit (BUILT_IN_PRINTF), 2,
			 build_string_literal (strlen (fmt_long) + 1,
					       fmt_long),
			 build_error_message (error_var));
    }
  else
    {
      /* No error code provided.  */

      asprintf (&fmt_long, "%s:%d: error: %s\n",
		xloc.file, xloc.line, str);

      print =
	build_call_expr (builtin_decl_explicit (BUILT_IN_PUTS), 1,
			 build_string_literal (strlen (fmt_long) + 1,
					       fmt_long));
    }

  free (fmt_long);
  free (str);
  va_end (args);

  tree stmts = NULL;
  append_to_statement_list (print, &stmts);
  append_to_statement_list (build_call_expr
			    (builtin_decl_explicit (BUILT_IN_ABORT), 0),
			    &stmts);

  return stmts;
}

/* Return a fresh argument list for FN.  */

tree
build_function_arguments (tree fn)
{
  gcc_assert (TREE_CODE (fn) == FUNCTION_DECL
	      && DECL_ARGUMENTS (fn) == NULL_TREE);

  local_define (tree, build_argument, (const_tree lst))
    {
      tree param, type;

      type = TREE_VALUE (lst);
      param = build_decl (DECL_SOURCE_LOCATION (fn), PARM_DECL,
			  create_tmp_var_name ("argument"),
			  type);
      DECL_ARG_TYPE (param) = type;
      DECL_CONTEXT (param) = fn;

      return param;
    };

  return map (build_argument,
	      list_remove (void_type_p,
			   TYPE_ARG_TYPES (TREE_TYPE (fn))));
}

/* Return true if LST holds the void type.  */

bool
void_type_p (const_tree lst)
{
  gcc_assert (TREE_CODE (lst) == TREE_LIST);
  return VOID_TYPE_P (TREE_VALUE (lst));
}

/* Return true if LST holds a pointer type.  */

bool
pointer_type_p (const_tree lst)
{
  gcc_assert (TREE_CODE (lst) == TREE_LIST);
  return POINTER_TYPE_P (TREE_VALUE (lst));
}


/* C expression parser, possibly with C++ linkage.  */

extern int yyparse (location_t, const char *, tree *);
extern int yydebug;

/* Parse expressions from the CPP reader for PRAGMA, which is located at LOC.
   Return a TREE_LIST of C expressions.  */

tree
read_pragma_expressions (const char *pragma, location_t loc)
{
  tree expr = NULL_TREE;

  if (yyparse (loc, pragma, &expr))
    /* Parse error or memory exhaustion.  */
    expr = NULL_TREE;

  return expr;
}


/* List and vector utilities, à la SRFI-1.  */

tree
chain_trees (tree t, ...)
{
  va_list args;

  va_start (args, t);

  tree next, prev = t;
  for (prev = t, next = va_arg (args, tree);
       next != NULL_TREE;
       prev = next, next = va_arg (args, tree))
    TREE_CHAIN (prev) = next;

  va_end (args);

  return t;
}

tree
filter (function_parm (bool, pred, (const_tree)), tree t)
{
  tree result, lst;

  gcc_assert (TREE_CODE (t) == TREE_LIST);

  result = NULL_TREE;
  for (lst = t; lst != NULL_TREE; lst = TREE_CHAIN (lst))
    {
      if (pred (lst))
	result = tree_cons (TREE_PURPOSE (lst), TREE_VALUE (lst),
			    result);
    }

  return nreverse (result);
}

tree
list_remove (function_parm (bool, pred, (const_tree)), tree t)
{
  local_define (bool, opposite, (const_tree t))
  {
    return !pred (t);
  };

  return filter (opposite, t);
}

/* Map FUNC over chain T.  T does not have to be `TREE_LIST'; it can be a
   chain of arbitrary tree objects.  */

tree
map (function_parm (tree, func, (const_tree)), tree t)
{
  tree result, tail, lst;

  result = tail = NULL_TREE;
  for (lst = t; lst != NULL_TREE; lst = TREE_CHAIN (lst))
    {
      tree r = func (lst);
      if (tail != NULL_TREE)
	TREE_CHAIN (tail) = r;
      else
	result = r;

      tail = r;
    }

  return result;
}

void
for_each (function_parm (void, func, (tree)), tree t)
{
  tree lst;

  gcc_assert (TREE_CODE (t) == TREE_LIST);

  for (lst = t; lst != NULL_TREE; lst = TREE_CHAIN (lst))
    func (TREE_VALUE (lst));
}

size_t
count (function_parm (bool, pred, (const_tree)), const_tree t)
{
  size_t result;
  const_tree lst;

  for (lst = t, result = 0; lst != NULL_TREE; lst = TREE_CHAIN (lst))
    if (pred (lst))
      result++;

  return result;
}


/* Useful code backported from GCC 4.6.  */

#if !HAVE_DECL_BUILD_CALL_EXPR_LOC_ARRAY

tree
build_call_expr_loc_array (location_t loc, tree fndecl, int n, tree *argarray)
{
  tree fntype = TREE_TYPE (fndecl);
  tree fn = build1 (ADDR_EXPR, build_pointer_type (fntype), fndecl);

  return fold_builtin_call_array (loc, TREE_TYPE (fntype), fn, n, argarray);
}

#endif

#if !HAVE_DECL_BUILD_CALL_EXPR_LOC_VEC

tree
build_call_expr_loc_vec (location_t loc, tree fndecl, VEC(tree,gc) *vec)
{
  return build_call_expr_loc_array (loc, fndecl, VEC_length (tree, vec),
				    VEC_address (tree, vec));
}

#endif

#if !HAVE_DECL_BUILD_ZERO_CST

tree
build_zero_cst (tree type)
{
  switch (TREE_CODE (type))
    {
    case INTEGER_TYPE: case ENUMERAL_TYPE: case BOOLEAN_TYPE:
    case POINTER_TYPE: case REFERENCE_TYPE:
    case OFFSET_TYPE:
      return build_int_cst (type, 0);

    default:
      abort ();
    }
}

#endif

/* Build a "conversion" from a raw C pointer to its data handle.  The
   assumption is that the programmer should have already registered the
   pointer by themselves.  */

tree
build_pointer_lookup (tree pointer)
{
  static tree data_lookup_fn;

  /* Make sure DATA_LOOKUP_FN is valid.  */
  LOOKUP_STARPU_FUNCTION (data_lookup_fn, "starpu_data_lookup");

  location_t loc;

  if (DECL_P (pointer))
    loc = DECL_SOURCE_LOCATION (pointer);
  else
    loc = UNKNOWN_LOCATION;

  /* Introduce a local variable to hold the handle.  */

  tree result_var = build_decl (loc, VAR_DECL,
  				create_tmp_var_name (".data_lookup_result"),
  				ptr_type_node);
  DECL_CONTEXT (result_var) = current_function_decl;
  DECL_ARTIFICIAL (result_var) = true;
  DECL_SOURCE_LOCATION (result_var) = loc;

  tree call = build_call_expr (data_lookup_fn, 1, pointer);
  tree assignment = build2 (INIT_EXPR, TREE_TYPE (result_var),
  			    result_var, call);

  /* Build `if (RESULT_VAR == NULL) error ();'.  */

  tree cond = build3 (COND_EXPR, void_type_node,
		      build2 (EQ_EXPR, boolean_type_node,
			      result_var, null_pointer_node),
		      build_error_statements (loc, NULL_TREE,
					      build_starpu_error_string,
					      "attempt to use unregistered "
					      "pointer"),
		      NULL_TREE);

  tree stmts = NULL;
  append_to_statement_list (assignment, &stmts);
  append_to_statement_list (cond, &stmts);
  append_to_statement_list (result_var, &stmts);

  return build4 (TARGET_EXPR, ptr_type_node, result_var, stmts, NULL_TREE, NULL_TREE);
}

/* Build an error string for the StarPU return value in ERROR_VAR.  */

tree
build_starpu_error_string (tree error_var)
{
  static tree strerror_fn;
  LOOKUP_STARPU_FUNCTION (strerror_fn, "strerror");

  tree error_code =
    build1 (NEGATE_EXPR, TREE_TYPE (error_var), error_var);

  return build_call_expr (strerror_fn, 1, error_code);
}

/* Like `build_constructor_from_list', but sort VALS according to their
   offset in struct TYPE.  Inspired by `gnat_build_constructor'.  */

tree
build_constructor_from_unsorted_list (tree type, tree vals)
{
  local_define (int, compare_elmt_bitpos, (const void *rt1, const void *rt2))
  {
    const constructor_elt *elmt1 = (constructor_elt *) rt1;
    const constructor_elt *elmt2 = (constructor_elt *) rt2;
    const_tree field1 = elmt1->index;
    const_tree field2 = elmt2->index;
    int ret
      = tree_int_cst_compare (bit_position (field1), bit_position (field2));

    return ret ? ret : (int) (DECL_UID (field1) - DECL_UID (field2));
  };

  tree t;
  VEC(constructor_elt,gc) *v = NULL;

  if (vals)
    {
      v = VEC_alloc (constructor_elt, gc, list_length (vals));
      for (t = vals; t; t = TREE_CHAIN (t))
	CONSTRUCTOR_APPEND_ELT (v, TREE_PURPOSE (t), TREE_VALUE (t));
    }

  /* Sort field initializers by field offset.  */
  VEC_qsort (constructor_elt, v, compare_elmt_bitpos);

  return build_constructor (type, v);
}
