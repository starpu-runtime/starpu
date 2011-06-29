/* GCC-StarPU
   Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique

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

/* Use extensions of the GNU C Library.  */
#define _GNU_SOURCE 1

#include <starpu-gcc-config.h>

int plugin_is_GPL_compatible;

/* #define ENABLE_TREE_CHECKING 1 */

#include <gcc-plugin.h>
#include <plugin-version.h>

#include <plugin.h>
#include <cpplib.h>
#include <tree.h>
#include <tree-iterator.h>
#include <c-common.h>
#include <c-pragma.h>
#include <tm.h>
#include <gimple.h>
#include <tree-pass.h>
#include <tree-flow.h>
#include <cgraph.h>
#include <gimple.h>
#include <toplev.h>

#include <stdio.h>

#include <starpu.h>  /* for `STARPU_CPU' & co.  */


/* The name of this plug-in.  */
static const char plugin_name[] = "starpu";

/* Names of public attributes.  */
static const char task_attribute_name[] = "task";
static const char task_implementation_attribute_name[] = "task_implementation";

/* Names of attributes used internally.  */
static const char task_codelet_attribute_name[] = ".codelet";
static const char task_implementation_list_attribute_name[] =
  ".task_implementation_list";
static const char task_implementation_wrapper_attribute_name[] =
  ".task_implementation_wrapper";

/* Names of data structures defined in <starpu.h>.  */
static const char codelet_struct_name[] = "starpu_codelet";
static const char task_struct_name[] = "starpu_task";

/* The `starpu_insert_task' and `starpu_data_lookup' FUNCTION_DECLs.  */
static tree insert_task_fn, data_lookup_fn;


/* Forward declarations.  */

static tree build_codelet_declaration (tree task_decl);



/* Useful code backported from GCC 4.6.  */

#if !HAVE_DECL_BUILD_CALL_EXPR_LOC_ARRAY

static tree
build_call_expr_loc_array (location_t loc, tree fndecl, int n, tree *argarray)
{
  tree fntype = TREE_TYPE (fndecl);
  tree fn = build1 (ADDR_EXPR, build_pointer_type (fntype), fndecl);

  return fold_builtin_call_array (loc, TREE_TYPE (fntype), fn, n, argarray);
}

#endif

#if !HAVE_DECL_BUILD_CALL_EXPR_LOC_VEC

static tree
build_call_expr_loc_vec (location_t loc, tree fndecl, VEC(tree,gc) *vec)
{
  return build_call_expr_loc_array (loc, fndecl, VEC_length (tree, vec),
				    VEC_address (tree, vec));
}

#endif


/* Helpers.  */


/* Build a reference to the INDEXth element of ARRAY.  `build_array_ref' is
   not exported, so we roll our own.  */

static tree
array_ref (tree array, size_t index)
{
  gcc_assert (POINTER_TYPE_P (TREE_TYPE (array)));

  tree pointer_plus_offset =
    index > 0
    ? build_binary_op (UNKNOWN_LOCATION, PLUS_EXPR,
		       array,
		       build_int_cstu (integer_type_node, index),
		       0)
    : array;

  gcc_assert (POINTER_TYPE_P (TREE_TYPE (pointer_plus_offset)));

  return build_indirect_ref (UNKNOWN_LOCATION,
			     pointer_plus_offset,
			     RO_ARRAY_INDEXING);
}

/* Like `build_constructor_from_list', but sort VALS according to their
   offset in struct TYPE.  Inspired by `gnat_build_constructor'.  */

static tree
build_constructor_from_unsorted_list (tree type, tree vals)
{
  int compare_elmt_bitpos (const void *rt1, const void *rt2)
  {
    const constructor_elt *elmt1 = (constructor_elt *) rt1;
    const constructor_elt *elmt2 = (constructor_elt *) rt2;
    const_tree field1 = elmt1->index;
    const_tree field2 = elmt2->index;
    int ret
      = tree_int_cst_compare (bit_position (field1), bit_position (field2));

    return ret ? ret : (int) (DECL_UID (field1) - DECL_UID (field2));
  }

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



/* Debugging helpers.  */

static tree build_printf (const char *, ...)
  __attribute__ ((format (printf, 1, 2)));

static tree
build_printf (const char *fmt, ...)
{
  tree call;
  char *str;
  va_list args;

  va_start (args, fmt);
  vasprintf (&str, fmt, args);
  call = build_call_expr (built_in_decls[BUILT_IN_PUTS], 1,
	 		  build_string_literal (strlen (str) + 1, str));
  free (str);
  va_end (args);

  return call;
}

static tree
build_hello_world (void)
{
  return build_printf ("Hello, StarPU!");
}


/* List and vector utilities, à la SRFI-1.  */

static tree chain_trees (tree t, ...)
  __attribute__ ((sentinel));

static tree
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

static tree
filter (bool (*pred) (const_tree), tree t)
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

/* Map FUNC over chain T.  T does not have to be `TREE_LIST'; it can be a
   chain of arbitrary tree objects.  */

static tree
map (tree (*func) (const_tree), tree t)
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

static void
for_each (void (*func) (tree), tree t)
{
  tree lst;

  gcc_assert (TREE_CODE (t) == TREE_LIST);

  for (lst = t; lst != NULL_TREE; lst = TREE_CHAIN (lst))
    func (TREE_VALUE (lst));
}


/* Pragmas.  */

#define STARPU_PRAGMA_NAME_SPACE "starpu"

static void
handle_pragma_hello (struct cpp_reader *reader)
{
  add_stmt (build_hello_world ());
}

static void
handle_pragma_wait (struct cpp_reader *reader)
{
  tree fndecl;

  fndecl = lookup_name (get_identifier ("starpu_task_wait_for_all"));
  gcc_assert (TREE_CODE (fndecl) == FUNCTION_DECL);

  add_stmt (build_call_expr (fndecl, 0));
}

/* Process `#pragma starpu register VAR [COUNT]' and emit the corresponding
   `starpu_vector_data_register' call.  */

static void
handle_pragma_register (struct cpp_reader *reader)
{
  tree token;
  location_t loc;
  enum cpp_ttype type;

  loc = cpp_peek_token (reader, 0)->src_loc;

  type = pragma_lex (&token);
  if (type == CPP_EOF)
    {
      error_at (loc, "unterminated %<starpu register%> pragma");
      return;
    }
  else if (type != CPP_NAME)
    {
      error_at (loc, "identifier expected");
      return;
    }

  /* Get the variable name.  */
  tree var_name = token;
  tree var = lookup_name (var_name);
  if (var == NULL_TREE || !DECL_P (var))
    {
      error_at (loc, "unbound variable %qE", var_name);
      return;
    }

  if (!POINTER_TYPE_P (TREE_TYPE (var))
      && TREE_CODE (TREE_TYPE (var)) != ARRAY_TYPE)
    {
      error_at (loc, "%qE is neither a pointer nor an array", var_name);
      return;
    }

  if (TREE_CODE (TREE_TYPE (var)) == ARRAY_TYPE
      && !DECL_EXTERNAL (var)
      && !TREE_STATIC (var)
      && !MAIN_NAME_P (DECL_NAME (current_function_decl)))
    warning_at (loc, 0, "using an on-stack array as a task input "
		"considered unsafe");

  /* Determine the number of elements in the vector.  */
  tree count = NULL_TREE;

  if (TREE_CODE (TREE_TYPE (var)) == ARRAY_TYPE)
    {
      tree domain = TYPE_DOMAIN (TREE_TYPE (var));

      if (domain != NULL_TREE)
	{
	  count = build_binary_op (loc, MINUS_EXPR,
				   TYPE_MAX_VALUE (domain),
				   TYPE_MIN_VALUE (domain),
				   false);
	  count = build_binary_op (loc, PLUS_EXPR,
				   count,
				   build_int_cstu (integer_type_node, 1),
				   false);
	  count = fold_convert (size_type_node, count);
	}
    }


  type = pragma_lex (&token);
  if (type == CPP_EOF)
    {
      /* End of line reached: don't consume TOKEN and check whether the array
	 size was determined.  */
      if (count == NULL_TREE)
	error_at (loc, "cannot determine size of array %qE", var_name);
    }
  else if (type == CPP_NUMBER)
    {
      /* TOKEN is a number, consume it.  */
      tree count_arg = token;

      if (count != NULL_TREE)
	{
	  /* The number of elements of this array was already determined.  */
	  inform (loc,
		  "element count can be omitted for bounded array %qE",
		  var_name);

	  if (!tree_int_cst_equal (count, count_arg))
	    error_at (loc,
		      "specified element count differs from actual size of array %qE",
		      var_name);
	}
      else
	count = count_arg;

      if (pragma_lex (&token) != CPP_EOF)
	error_at (loc, "junk after %<starpu register%> pragma");
    }
  else
    {
      error_at (loc, "integer expected");
      return;
    }

  /* If VAR is an array, take its address.  */
  tree pointer =
    POINTER_TYPE_P (TREE_TYPE (var))
    ? var
    : build_addr (var, current_function_decl);

  /* Introduce a local variable to hold the handle.  */
  tree handle_var = create_tmp_var (ptr_type_node, ".handle");

  tree register_fn =
    lookup_name (get_identifier ("starpu_vector_data_register"));

  /* Build `starpu_vector_data_register (&HANDLE_VAR, 0, POINTER,
                                         COUNT, sizeof *POINTER)'  */
  tree call =
    build_call_expr (register_fn, 5,
		     build_addr (handle_var, current_function_decl),
		     build_zero_cst (uintptr_type_node), /* home node */
		     pointer, count,
		     size_in_bytes (TREE_TYPE (TREE_TYPE (var))));

  add_stmt (call);
}

static void
register_pragmas (void *gcc_data, void *user_data)
{
  c_register_pragma (STARPU_PRAGMA_NAME_SPACE, "hello",
		     handle_pragma_hello);
  c_register_pragma (STARPU_PRAGMA_NAME_SPACE, "wait",
		     handle_pragma_wait);
  c_register_pragma_with_expansion (STARPU_PRAGMA_NAME_SPACE, "register",
				    handle_pragma_register);
}


/* Attributes.  */

/* Handle the `task' function attribute.  */

static tree
handle_task_attribute (tree *node, tree name, tree args,
		       int flags, bool *no_add_attrs)
{
  tree fn;

  fn = *node;
  gcc_assert (TREE_CODE (fn) == FUNCTION_DECL);

  /* This is a function declaration for something local to this
     translation unit, so add the `task' attribute to FN.  */
  *no_add_attrs = false;

  /* Add an empty `task_implementation_list' attribute.  */
  DECL_ATTRIBUTES (fn) =
    tree_cons (get_identifier (task_implementation_list_attribute_name),
	       NULL_TREE,
	       NULL_TREE);

  if (!TREE_PUBLIC (fn))
    {
      /* Set a dummy body to avoid "used but never defined" warnings when FN
	 has file scope.  */
      DECL_SAVED_TREE (fn) = build_printf ("hello from the task!");
      DECL_INITIAL (fn) = build_block (NULL_TREE, NULL_TREE, fn, NULL_TREE);
    }

  /* Push a declaration for the corresponding `starpu_codelet' object and add
     it as an attribute of FN.  */
  tree cl = build_codelet_declaration (fn);
  DECL_ATTRIBUTES (fn) =
    tree_cons (get_identifier (task_codelet_attribute_name), cl,
	       DECL_ATTRIBUTES (fn));
  pushdecl (cl);

  TREE_USED (fn) = true;

  /* Lookup the useful StarPU functions in the global scope (this can't be
     done from `lower_starpu'.)
     XXX: Move it in a pass of its own.  */

#define LOOKUP_STARPU_FUNCTION(var, name)				\
  if ((var) == NULL_TREE)						\
    {									\
      (var) = lookup_name (get_identifier (name));			\
      gcc_assert ((var) != NULL_TREE && TREE_CODE (var) == FUNCTION_DECL); \
    }

  LOOKUP_STARPU_FUNCTION (insert_task_fn, "starpu_insert_task");
  LOOKUP_STARPU_FUNCTION (data_lookup_fn, "starpu_data_lookup");

#undef LOOKUP_STARPU_FUNCTION

  return NULL_TREE;
}

/* Handle the `task_implementation (WHERE, TASK)' attribute.  WHERE is a
   string constant ("cpu", "cuda", etc.), and TASK is the identifier of a
   function declared with the `task' attribute.  */

static tree
handle_task_implementation_attribute (tree *node, tree name, tree args,
				      int flags, bool *no_add_attrs)
{
  tree fn, where, task_decl;

  /* FIXME: Use error nodes instead of `gcc_assert'.  */

  /* FIXME:TODO: To change the order to (TASK, WHERE):
	  tree cleanup_id = TREE_VALUE (TREE_VALUE (attr));
	  tree cleanup_decl = lookup_name (cleanup_id);
  */


  fn = *node;
  gcc_assert (TREE_CODE (fn) == FUNCTION_DECL);

  where = TREE_VALUE (args);
  gcc_assert (TREE_CODE (where) == STRING_CST);

  task_decl = TREE_VALUE (TREE_CHAIN (args));
  gcc_assert (TREE_CODE (task_decl) == FUNCTION_DECL);
  gcc_assert (lookup_attribute (task_attribute_name,
				DECL_ATTRIBUTES (task_decl)));
  gcc_assert (TYPE_CANONICAL (TREE_TYPE (fn))
	      == TYPE_CANONICAL (TREE_TYPE (task_decl))
	      && TYPE_CANONICAL (TREE_TYPE (fn)) != NULL_TREE);

  /* Add FN to the list of implementations of TASK_DECL.  */

  tree attr, impls;

  attr = lookup_attribute (task_implementation_list_attribute_name,
			   DECL_ATTRIBUTES (task_decl));
  impls = tree_cons (NULL_TREE, fn, TREE_VALUE (attr));

  DECL_ATTRIBUTES (task_decl) =
    tree_cons (get_identifier (task_implementation_list_attribute_name),
	       impls,
	       remove_attribute (task_implementation_list_attribute_name,
				 DECL_ATTRIBUTES (task_decl)));

  TREE_USED (fn) = true;

  /* Keep the attribute.  */
  *no_add_attrs = false;

  return NULL_TREE;
}

/* Return the declaration of the `starpu_codelet' variable associated with
   TASK_DECL.  */

static tree
task_codelet_declaration (const_tree task_decl)
{
  tree cl_attr;

  cl_attr = lookup_attribute (task_codelet_attribute_name,
			      DECL_ATTRIBUTES (task_decl));
  gcc_assert (cl_attr != NULL_TREE);

  return TREE_VALUE (cl_attr);
}

/* Return the list of implementations of TASK_DECL.  */

static tree
task_implementation_list (const_tree task_decl)
{
  tree attr;

  attr = lookup_attribute (task_implementation_list_attribute_name,
			   DECL_ATTRIBUTES (task_decl));
  return TREE_VALUE (attr);
}

/* Return the list of pointer parameter types of TASK_DECL.  */

static tree
task_pointer_parameter_types (const_tree task_decl)
{
  bool is_pointer (const_tree item)
  {
    return POINTER_TYPE_P (TREE_VALUE (item));
  }

  return filter (is_pointer, TYPE_ARG_TYPES (TREE_TYPE (task_decl)));
}

/* Return a value indicating where TASK_IMPL should execute (`STARPU_CPU',
   `STARPU_CUDA', etc.).  */

static int
task_implementation_where (const_tree task_impl)
{
  int where_int;
  tree impl_attr, args, where;

  gcc_assert (TREE_CODE (task_impl) == FUNCTION_DECL);

  impl_attr = lookup_attribute (task_implementation_attribute_name,
				DECL_ATTRIBUTES (task_impl));
  gcc_assert (impl_attr != NULL_TREE);

  args = TREE_VALUE (impl_attr);
  where = TREE_VALUE (args);

  if (!strncmp (TREE_STRING_POINTER (where), "cpu",
		TREE_STRING_LENGTH (where)))
    where_int = STARPU_CPU;
  else if (!strncmp (TREE_STRING_POINTER (where), "opencl",
		     TREE_STRING_LENGTH (where)))
    where_int = STARPU_OPENCL;
  else if (!strncmp (TREE_STRING_POINTER (where), "cuda",
		     TREE_STRING_LENGTH (where)))
    where_int = STARPU_CUDA;
  else
    /* FIXME: Error out?  */
    where_int = 0;

  return where_int;
}

/* Return the task implemented by TASK_IMPL.  */

static tree
task_implementation_task (const_tree task_impl)
{
  tree impl_attr, args;

  gcc_assert (TREE_CODE (task_impl) == FUNCTION_DECL);

  impl_attr = lookup_attribute (task_implementation_attribute_name,
				DECL_ATTRIBUTES (task_impl));
  gcc_assert (impl_attr != NULL_TREE);

  args = TREE_VALUE (impl_attr);

  return TREE_VALUE (TREE_CHAIN (args));
}

/* Return the FUNCTION_DECL of the wrapper generated for TASK_IMPL.  */

static tree
task_implementation_wrapper (const_tree task_impl)
{
  tree attr;

  gcc_assert (TREE_CODE (task_impl) == FUNCTION_DECL);

  attr = lookup_attribute (task_implementation_wrapper_attribute_name,
			   DECL_ATTRIBUTES (task_impl));
  gcc_assert (attr != NULL_TREE);

  return TREE_VALUE (attr);
}


static void
register_task_attributes (void *gcc_data, void *user_data)
{
  static const struct attribute_spec task_attr =
    {
      task_attribute_name, 0, 0, true, false, false,
      handle_task_attribute
    };

  static const struct attribute_spec task_implementation_attr =
    {
      task_implementation_attribute_name, 2, 2, true, false, false,
      handle_task_implementation_attribute
    };

  register_attribute (&task_attr);
  register_attribute (&task_implementation_attr);
}



/* Return the type of a codelet function, i.e.,
   `void (*) (void **, void *)'.  */

static tree
build_codelet_wrapper_type (void)
{
  tree void_ptr_ptr;

  void_ptr_ptr = build_pointer_type (ptr_type_node);

  return build_function_type_list (void_type_node,
				   void_ptr_ptr, ptr_type_node,
				   NULL_TREE);
}

/* Return an identifier for the wrapper of TASK_IMPL, a task
   implementation.  */

static tree
build_codelet_wrapper_identifier (tree task_impl)
{
  static const char suffix[] = ".task_implementation_wrapper";

  tree id;
  char *cl_name;
  const char *task_name;

  id = DECL_NAME (task_impl);
  task_name = IDENTIFIER_POINTER (id);

  cl_name = alloca (IDENTIFIER_LENGTH (id) + strlen (suffix) + 1);
  memcpy (cl_name, task_name, IDENTIFIER_LENGTH (id));
  strcpy (&cl_name[IDENTIFIER_LENGTH (id)], suffix);

  return get_identifier (cl_name);
}

/* Return a function of type `void (*) (void **, void *)' that calls function
   TASK_IMPL, the FUNCTION_DECL of a task implementation whose prototype may
   be arbitrary.  */

static tree
build_codelet_wrapper_definition (tree task_impl)
{
  location_t loc;
  tree task_decl, decl;

  loc = DECL_SOURCE_LOCATION (task_impl);
  task_decl = task_implementation_task (task_impl);

  bool not_void_type_p (const_tree type)
  {
    return !VOID_TYPE_P (TREE_VALUE (type));
  }

  tree build_local_var (const_tree type)
  {
    tree var, t;
    const char *seed;

    t = TREE_VALUE (type);
    seed = POINTER_TYPE_P (t) ? "pointer_arg" : "scalar_arg";

    var = build_decl (loc, VAR_DECL, create_tmp_var_name (seed), t);
    DECL_CONTEXT (var) = decl;
    DECL_ARTIFICIAL (var) = true;

    return var;
  }

  /* Return the body of the wrapper, which unpacks `cl_args' and calls the
     user-defined task implementation.  */

  tree build_body (tree wrapper_decl, tree vars)
  {
    tree stmts = NULL, call, unpack_fndecl, v;
    VEC(tree, gc) *args;

    unpack_fndecl = lookup_name (get_identifier ("starpu_unpack_cl_args"));
    gcc_assert (unpack_fndecl != NULL_TREE
    		&& TREE_CODE (unpack_fndecl) == FUNCTION_DECL);

    /* Build `var0 = STARPU_VECTOR_GET_PTR (buffers[0]); ...'.  */

    size_t index = 0;
    for (v = vars; v != NULL_TREE; v = TREE_CHAIN (v))
      {
	if (POINTER_TYPE_P (TREE_TYPE (v)))
	  {
	    /* Compute `void *VDESC = buffers[0];'.  */
	    tree vdesc = array_ref (DECL_ARGUMENTS (wrapper_decl), index);

	    /* Below we assume (1) that pointer arguments are registered as
	       StarPU vector handles, and (2) that the `ptr' field is at
	       offset 0 of `starpu_vector_interface_s'.  The latter allows us
	       to use a simple pointer dereference instead of expanding
	       `STARPU_VECTOR_GET_PTR'.  */
	    assert (offsetof (struct starpu_vector_interface_s, ptr) == 0);

	    /* Compute `type *PTR = *(type **) VDESC;'.  */
	    tree ptr = build1 (INDIRECT_REF,
			       build_pointer_type (TREE_TYPE (v)),
			       vdesc);

	    append_to_statement_list (build2 (MODIFY_EXPR, TREE_TYPE (v),
					      v, ptr),
				      &stmts);

	    index++;
	  }
      }

    /* Build `starpu_unpack_cl_args (cl_args, &var1, &var2, ...)'.  */

    args = NULL;
    VEC_safe_push (tree, gc, args, TREE_CHAIN (DECL_ARGUMENTS (wrapper_decl)));
    for (v = vars; v != NULL_TREE; v = TREE_CHAIN (v))
      {
	if (!POINTER_TYPE_P (TREE_TYPE (v)))
	  VEC_safe_push (tree, gc, args, build_addr (v, wrapper_decl));
      }

    if (VEC_length (tree, args) > 1)
      {
	call = build_call_expr_loc_vec (UNKNOWN_LOCATION, unpack_fndecl, args);
	TREE_SIDE_EFFECTS (call) = 1;
	append_to_statement_list (call, &stmts);
      }

    /* Build `my_task_impl (var1, var2, ...)'.  */

    args = NULL;
    for (v = vars; v != NULL_TREE; v = TREE_CHAIN (v))
      VEC_safe_push (tree, gc, args, v);

    call = build_call_expr_loc_vec (UNKNOWN_LOCATION, task_impl, args);
    TREE_SIDE_EFFECTS (call) = 1;
    append_to_statement_list (call, &stmts);

    tree bind;
    bind = build3 (BIND_EXPR, void_type_node, vars, stmts,
		   DECL_INITIAL (wrapper_decl));
    TREE_TYPE (bind) = TREE_TYPE (TREE_TYPE (wrapper_decl));

    return bind;
  }

  /* Return the parameter list of the wrapper:
     `(void **BUFFERS, void *CL_ARGS)'.  */

  tree build_parameters (tree wrapper_decl)
  {
    tree param1, param2;

    param1 = build_decl (loc, PARM_DECL,
			 create_tmp_var_name ("buffers"),
			 build_pointer_type (ptr_type_node));
    DECL_ARG_TYPE (param1) = ptr_type_node;
    DECL_CONTEXT (param1) = wrapper_decl;
    TREE_USED (param1) = true;

    param2 = build_decl (loc, PARM_DECL,
			 create_tmp_var_name ("cl_args"),
			 ptr_type_node);
    DECL_ARG_TYPE (param2) = ptr_type_node;
    DECL_CONTEXT (param2) = wrapper_decl;
    TREE_USED (param2) = true;

    return chainon (param1, param2);
  }

  tree wrapper_name, vars, result;

  wrapper_name = build_codelet_wrapper_identifier (task_impl);
  decl = build_decl (loc, FUNCTION_DECL, wrapper_name,
		     build_codelet_wrapper_type ());

  vars = map (build_local_var,
	      filter (not_void_type_p,
		      TYPE_ARG_TYPES (TREE_TYPE (task_decl))));

  DECL_CONTEXT (decl) = NULL_TREE;
  DECL_ARGUMENTS (decl) = build_parameters (decl);

  result = build_decl (loc, RESULT_DECL, NULL_TREE, void_type_node);
  DECL_CONTEXT (result) = decl;
  DECL_ARTIFICIAL (result) = true;
  DECL_IGNORED_P (result) = true;
  DECL_RESULT (decl) = result;

  DECL_INITIAL (decl) = build_block (vars, NULL_TREE, decl, NULL_TREE);

  DECL_SAVED_TREE (decl) = build_body (decl, vars);

  TREE_PUBLIC (decl) = TREE_PUBLIC (task_impl);
  TREE_STATIC (decl) = true;
  TREE_USED (decl) = true;
  DECL_ARTIFICIAL (decl) = true;
  DECL_EXTERNAL (decl) = false;
  DECL_UNINLINABLE (decl) = true;

  rest_of_decl_compilation (decl, true, 0);

  struct function *prev_cfun = cfun;
  set_cfun (NULL);
  allocate_struct_function (decl, false);
  cfun->function_end_locus = DECL_SOURCE_LOCATION (task_impl);

  cgraph_finalize_function (decl, false);

  /* Mark DECL as needed so that it doesn't get removed by
     `cgraph_remove_unreachable_nodes' when it's not public.  */
  cgraph_mark_needed_node (cgraph_get_node (decl));

  set_cfun (prev_cfun);

  return decl;
}

/* Define one wrapper function for each implementation of TASK.  TASK should
   be the FUNCTION_DECL of a task.  */

static void
define_codelet_wrappers (tree task)
{
  void define (tree task_impl)
  {
    tree wrapper_def;

    wrapper_def = build_codelet_wrapper_definition (task_impl);

    DECL_ATTRIBUTES (task_impl) =
      tree_cons (get_identifier (task_implementation_wrapper_attribute_name),
		 wrapper_def,
		 DECL_ATTRIBUTES (task_impl));

    pushdecl (wrapper_def);
  }

  for_each (define, task_implementation_list (task));
}

/* Return a NODE_IDENTIFIER for the variable holding the `starpu_codelet'
   structure associated with TASK_DECL.  */

static tree
build_codelet_identifier (tree task_decl)
{
  static const char suffix[] = ".codelet";

  tree id;
  char *cl_name;
  const char *task_name;

  id = DECL_NAME (task_decl);
  task_name = IDENTIFIER_POINTER (id);

  cl_name = alloca (IDENTIFIER_LENGTH (id) + strlen (suffix) + 1);
  memcpy (cl_name, task_name, IDENTIFIER_LENGTH (id));
  strcpy (&cl_name[IDENTIFIER_LENGTH (id)], suffix);

  return get_identifier (cl_name);
}

static tree
codelet_type (void)
{
  tree type_decl;

  /* Lookup the `starpu_codelet' struct type.  This should succeed since we
     push <starpu.h> early on.  */

  type_decl = lookup_name (get_identifier (codelet_struct_name));
  gcc_assert (type_decl != NULL_TREE && TREE_CODE (type_decl) == TYPE_DECL);

  return TREE_TYPE (type_decl);
}

/* Return a VAR_DECL that declares a `starpu_codelet' structure for
   TASK_DECL.  */

static tree
build_codelet_declaration (tree task_decl)
{
  tree name, cl_decl;

  name = build_codelet_identifier (task_decl);

  cl_decl = build_decl (DECL_SOURCE_LOCATION (task_decl),
			VAR_DECL, name,
			/* c_build_qualified_type (type, TYPE_QUAL_CONST) */
			codelet_type ());

  DECL_ARTIFICIAL (cl_decl) = true;
  TREE_PUBLIC (cl_decl) = TREE_PUBLIC (task_decl);
  TREE_STATIC (cl_decl) = false;
  TREE_USED (cl_decl) = true;
  DECL_EXTERNAL (cl_decl) = true;
  DECL_CONTEXT (cl_decl) = NULL_TREE;

  return cl_decl;
}

/* Return a `starpu_codelet' initializer for TASK_DECL.  */

static tree
build_codelet_initializer (tree task_decl)
{
  tree fields;

  fields = TYPE_FIELDS (codelet_type ());
  gcc_assert (TREE_CODE (fields) == FIELD_DECL);

  tree lookup_field (const char *name)
  {
    tree fdecl, fname;

    fname = get_identifier (name);
    for (fdecl = fields;
	 fdecl != NULL_TREE;
	 fdecl = TREE_CHAIN (fdecl))
      {
	if (DECL_NAME (fdecl) == fname)
	  return fdecl;
      }

    /* Field NAME wasn't found.  */
    gcc_assert (false);
  }

  tree field_initializer (const char *name, tree value)
  {
    tree field, init;

    field = lookup_field (name);
    init = make_node (TREE_LIST);
    TREE_PURPOSE (init) = field;
    TREE_VALUE (init) = fold_convert (TREE_TYPE (field), value);
    TREE_CHAIN (init) = NULL_TREE;

    return init;
  }

  tree where_init (tree impls)
  {
    tree impl;
    int where_int = 0;

    for (impl = impls;
	 impl != NULL_TREE;
	 impl = TREE_CHAIN (impl))
      {
	tree impl_decl;

	impl_decl = TREE_VALUE (impl);
	gcc_assert (TREE_CODE (impl_decl) == FUNCTION_DECL);

	printf ("   `%s'\n", IDENTIFIER_POINTER (DECL_NAME (impl_decl)));

	where_int |= task_implementation_where (impl_decl);
      }

    return build_int_cstu (integer_type_node, where_int);
  }

  tree implementation_pointer (tree impls, int where)
  {
    tree impl;

    for (impl = impls;
	 impl != NULL_TREE;
	 impl = TREE_CHAIN (impl))
      {
	tree impl_decl;

	impl_decl = TREE_VALUE (impl);
	if (task_implementation_where (impl_decl) == where)
	  {
	    /* Return a pointer to the wrapper of IMPL_DECL.  */
	    tree addr = build_addr (task_implementation_wrapper (impl_decl),
				    NULL_TREE);
	    return addr;
	  }
      }

    /* Default to a NULL pointer.  */
    return build_int_cstu (build_pointer_type (void_type_node), 0);
  }

  tree pointer_arg_count (void)
  {
    size_t len;

    len = list_length (task_pointer_parameter_types (task_decl));
    return build_int_cstu (integer_type_node, len);
  }

  printf ("implementations for `%s':\n",
	  IDENTIFIER_POINTER (DECL_NAME (task_decl)));

  tree impls, inits;

  impls = task_implementation_list (task_decl);

  inits =
    chain_trees (field_initializer ("where", where_init (impls)),
		 field_initializer ("nbuffers", pointer_arg_count ()),
		 field_initializer ("cpu_func",
				    implementation_pointer (impls, STARPU_CPU)),
		 field_initializer ("opencl_func",
		 		    implementation_pointer (impls,
		 					    STARPU_OPENCL)),
		 field_initializer ("cuda_func",
		 		    implementation_pointer (impls,
		 					    STARPU_CUDA)),
		 NULL_TREE);

  return build_constructor_from_unsorted_list (codelet_type (), inits);
}

/* Return the VAR_DECL that defines a `starpu_codelet' structure for
   TASK_DECL.  The VAR_DECL is assumed to already exists, so it must not be
   pushed again.  */

static tree
define_codelet (tree task_decl)
{
  /* Generate a wrapper function for each implementation of TASK_DECL that
     does all the packing/unpacking.  */
  define_codelet_wrappers (task_decl);

  /* Retrieve the declaration of the `starpu_codelet' object.  */
  tree cl_def;
  cl_def = lookup_name (build_codelet_identifier (task_decl));
  gcc_assert (cl_def != NULL_TREE && TREE_CODE (cl_def) == VAR_DECL);

  /* Turn the codelet declaration into a definition.  */
  TREE_PUBLIC (cl_def) = TREE_PUBLIC (task_decl);
  TREE_STATIC (cl_def) = true;
  DECL_EXTERNAL (cl_def) = false;
  DECL_INITIAL (cl_def) = build_codelet_initializer (task_decl);

  return cl_def;
}

/* Define the `starpu_codelet' structure for the task implemented by
   IMPL_DECL if we're in the right compilation unit, i.e., is IMPL_DECL is a
   "cpu" task implementation.  */

static void
maybe_define_codelet (tree impl_decl)
{
  if (task_implementation_where (impl_decl) == STARPU_CPU)
    /* IMPL_DECL is a "cpu" implementation of some task, so define the
       codelet structure in this compilation unit.  */
    define_codelet (task_implementation_task (impl_decl));
}

static void
handle_pre_genericize (void *gcc_data, void *user_data)
{
  tree fndecl = (tree) gcc_data;

  gcc_assert (fndecl != NULL_TREE && TREE_CODE (fndecl) == FUNCTION_DECL);
  gcc_assert (lookup_name (DECL_NAME (fndecl)) == fndecl);

  if (lookup_attribute (task_implementation_attribute_name,
			DECL_ATTRIBUTES (fndecl)))
    maybe_define_codelet (fndecl);
}

/* Build a "conversion" from a raw C pointer to its data handle.  The
   assumption is that the programmer should have already registered the
   pointer by themselves.  */

static tree
build_pointer_lookup (tree pointer, gimple_seq *body)
{
  gimple emit_error_message (void)
  {
    static const char msg[] =
      "starpu: task called with unregistered pointer, aborting\n";

    return gimple_build_call (built_in_decls[BUILT_IN_PUTS], 1,
			      build_string_literal (strlen (msg) + 1, msg));
  }

  tree var;

  var = create_tmp_var (ptr_type_node, ".handle-arg");
  mark_addressable (var);

  /* Initialize VAR with `starpu_data_lookup (POINTER)'.  */
  gimple_seq init = NULL;
  tree modify = build2 (MODIFY_EXPR, ptr_type_node, var,
			build_call_expr (data_lookup_fn, 1, pointer));
  force_gimple_operand (modify, &init, true, var);

  gimple_seq_add_seq (body, init);

  /* FIXME: Add `if (VAR == NULL) abort ();'.  */
#if 0
  tree abort_label = create_artificial_label (UNKNOWN_LOCATION);
  tree success_label = create_artificial_label (UNKNOWN_LOCATION);

  gimple cond = gimple_build_cond (EQ_EXPR,
				   var, null_pointer_node,
				   abort_label, success_label);
  gimple_seq_add_stmt (body, cond);

  gimplify_seq_add_stmt (body, gimple_build_label (abort_label));
  gimple_seq_add_stmt (body, emit_error_message ());
  gimple_seq_add_stmt (body,
		       gimple_build_call (built_in_decls[BUILT_IN_ABORT], 0));
  gimplify_seq_add_stmt (body, gimple_build_label (success_label));

  rebuild_cgraph_edges ();
#endif

  return var;
}

/* Build a call to `starpu_insert_task' for TASK_DECL, which will replace
   CALL.  */

static gimple_seq
build_task_submission (tree task_decl, gimple call)
{
  /* Return a chain of local variables that need to be introduced.  Variables
     are introduced for each argument that is either a scalar constant or a
     non-addressable variable---e.g., a `char' variable allocated in a
     register.  Populate BODY with the initial assignments to these
     variables.  */
  /* FIXME: We should also introduce local variables to mimic implicit
     integer type conversion---e.g., when the formal parameter type is `char'
     and the function is called with a `long' variable.  */

  tree local_vars (gimple_seq *body)
  {
    size_t n;
    tree vars = NULL_TREE;
    tree arg_types = TYPE_ARG_TYPES (TREE_TYPE (task_decl));

    for (n = 0;
	 n < gimple_call_num_args (call);
	 n++, arg_types = TREE_CHAIN (arg_types))
      {
	tree arg, type;

	/* Initially ARG_TYPES should be a list of N type nodes.  */
	gcc_assert (TREE_CODE (arg_types) == TREE_LIST);
	type = TREE_VALUE (arg_types);
	gcc_assert (type != NULL_TREE);

	arg = gimple_call_arg (call, n);

	if (!POINTER_TYPE_P (type)
	    && ((TREE_CONSTANT (arg)
		&& TREE_CODE (arg) != VAR_DECL
		&& TREE_CODE (arg) != ADDR_EXPR)
		|| is_gimple_non_addressable (arg)))
	  {
	    /* ARG is a scalar constant or a non-addressable variable.
	       Introduce a variable to hold it.  */
	    tree var =
	      create_tmp_var (type,
			      is_gimple_non_addressable (arg)
			      ? ".non-addressable-arg"
			      : ".literal-arg");
	    mark_addressable (var);

	    /* Initialize VAR.  */
	    tree init_value = fold_convert (type, arg);
	    gimple_seq init = NULL;
	    tree modify = build2 (MODIFY_EXPR, type, var, init_value);
	    force_gimple_operand (modify, &init, true, var);

	    gimple_seq_add_seq (body, init);

	    if (vars != NULL_TREE)
	      chainon (vars, var);
	    else
	      vars = var;
	  }
      }

    return vars;
  }

  size_t n;
  VEC(tree, heap) *args = NULL;
  tree vars;
  gimple_seq body = NULL;
  tree arg_types = TYPE_ARG_TYPES (TREE_TYPE (task_decl));

  vars = local_vars (&body);

  /* The first argument will be a pointer to the codelet.  */

  VEC_safe_push (tree, heap, args,
		 build_addr (task_codelet_declaration (task_decl),
			     current_function_decl));

  for (n = 0;
       n < gimple_call_num_args (call);
       n++, arg_types = TREE_CHAIN (arg_types))
    {
      tree arg, type;

      arg = gimple_call_arg (call, n);
      type = TREE_VALUE (arg_types);

      if (POINTER_TYPE_P (type))
	{
	  /* A pointer: the arguments will be:
	     `STARPU_RW, ptr' or similar.  */

	  gcc_assert (TREE_CODE (arg) == VAR_DECL
		      || TREE_CODE (arg) == ADDR_EXPR);

	  /* If TYPE points to a const-qualified type, then mark the data as
	     read-only; otherwise default to read-write.
	     FIXME: Add an attribute to specify write-only.  */
	  int mode =
	    (TYPE_QUALS (TREE_TYPE (type)) & TYPE_QUAL_CONST)
	    ? STARPU_R : STARPU_RW;

	  VEC_safe_push (tree, heap, args,
			 build_int_cst (integer_type_node, mode));
	  VEC_safe_push (tree, heap, args, build_pointer_lookup (arg, &body));
	}
      else
	{
	  /* A scalar: the arguments will be:
	     `STARPU_VALUE, &scalar, sizeof (scalar)'.  */

	  if (is_gimple_non_addressable (arg) || TREE_CONSTANT (arg))
	    {
	      /* Use the local variable we introduced to hold ARG's
		 value.  */
	      arg = vars;
	      vars = TREE_CHAIN (vars);
	    }
	  gcc_assert (TREE_CODE (arg) == VAR_DECL);
	  gcc_assert (TREE_ADDRESSABLE (arg));

	  VEC_safe_push (tree, heap, args,
			 build_int_cst (integer_type_node, STARPU_VALUE));
	  VEC_safe_push (tree, heap, args,
			 build_addr (arg, current_function_decl));
	  VEC_safe_push (tree, heap, args,
			 size_in_bytes (TREE_TYPE (arg)));
	}
    }

  /* Push the terminating zero.  */

  VEC_safe_push (tree, heap, args,
		 build_int_cst (integer_type_node, 0));

  gimple_seq_add_stmt (&body, gimple_build_call_vec (insert_task_fn, args));

  return body;
}

static unsigned int
lower_starpu (void)
{
  tree fndecl;
  const struct cgraph_node *cgraph;
  const struct cgraph_edge *callee;

  fndecl = current_function_decl;
  gcc_assert (TREE_CODE (fndecl) == FUNCTION_DECL);

  /* This pass should occur after `build_cgraph_edges'.  */
  cgraph = cgraph_get_node (fndecl);
  gcc_assert (cgraph != NULL);

  for (callee = cgraph->callees;
       callee != NULL;
       callee = callee->next_callee)
    {
      gcc_assert (callee->callee != NULL);

      tree callee_decl;

      callee_decl = callee->callee->decl;

      if (lookup_attribute (task_attribute_name,
			    DECL_ATTRIBUTES (callee_decl)))
	{
	  printf ("%s: `%s' calls task `%s'\n", __func__,
		  IDENTIFIER_POINTER (DECL_NAME (fndecl)),
		  IDENTIFIER_POINTER (DECL_NAME (callee_decl)));

	  gimple call_site;
	  gimple_seq submission;
	  gimple_stmt_iterator gsi;

	  call_site = callee->call_stmt;
	  gsi = gsi_for_stmt (call_site);

	  submission = build_task_submission (callee_decl, call_site);
	  gsi_remove (&gsi, true);
	  gsi_insert_seq_before (&gsi, submission, GSI_SAME_STMT);

	  rebuild_cgraph_edges ();
	}
    }

  return 0;
}

static struct opt_pass pass_lower_starpu =
  {
    .type = GIMPLE_PASS,
    .name = "pass_lower_starpu",
    .execute = lower_starpu,

    /* The rest is zeroed.  */
  };


/* Initialization.  */

static void
define_cpp_macros (void *gcc_data, void *user_data)
{
  cpp_define (parse_in, "STARPU_GCC_PLUGIN=0");
  cpp_push_include (parse_in, "starpu.h");
}

int
plugin_init (struct plugin_name_args *plugin_info,
	     struct plugin_gcc_version *version)
{
  if (!plugin_default_version_check (version, &gcc_version))
    return 1;

  register_callback (plugin_name, PLUGIN_START_UNIT,
		     define_cpp_macros, NULL);
  register_callback (plugin_name, PLUGIN_PRAGMAS,
		     register_pragmas, NULL);
  register_callback (plugin_name, PLUGIN_ATTRIBUTES,
		     register_task_attributes, NULL);
  register_callback (plugin_name, PLUGIN_PRE_GENERICIZE,
  		     handle_pre_genericize, NULL);

  /* Register our pass so that it happens after `build_cgraph_edges' has been
     done.  */

  struct register_pass_info pass_info =
    {
      .pass = &pass_lower_starpu,
      .reference_pass_name = "*build_cgraph_edges",
      .ref_pass_instance_number = 1,
      .pos_op = PASS_POS_INSERT_AFTER
    };

  register_callback (plugin_name, PLUGIN_PASS_MANAGER_SETUP,
		     NULL, &pass_info);

  return 0;
}
