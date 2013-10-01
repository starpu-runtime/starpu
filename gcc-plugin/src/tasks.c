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

#include <diagnostic.h>

#include <starpu-gcc/tasks.h>
#include <starpu-gcc/utils.h>
#include <starpu-gcc/opencl.h>


/* Task-related functions.  */

/* Name of public attributes.  */
const char task_attribute_name[] = "task";
const char task_implementation_attribute_name[] = "task_implementation";
const char output_attribute_name[] = "output";

/* Names of attributes used internally.  */
static const char task_codelet_attribute_name[] = ".codelet";
const char task_implementation_list_attribute_name[] =
  ".task_implementation_list";
const char task_implementation_wrapper_attribute_name[] =
  ".task_implementation_wrapper";

/* Names of data structures defined in <starpu.h>.  */
static const char codelet_struct_tag[] = "starpu_codelet";


/* Return true if DECL is a task.  */

bool
task_p (const_tree decl)
{
  return (TREE_CODE (decl) == FUNCTION_DECL &&
	  lookup_attribute (task_attribute_name,
			    DECL_ATTRIBUTES (decl)) != NULL_TREE);
}

/* Return true if DECL is a task implementation.  */

bool
task_implementation_p (const_tree decl)
{
  return (TREE_CODE (decl) == FUNCTION_DECL &&
	  lookup_attribute (task_implementation_attribute_name,
			    DECL_ATTRIBUTES (decl)) != NULL_TREE);
}

/* Return a value indicating where TASK_IMPL should execute (`STARPU_CPU',
   `STARPU_CUDA', etc.).  */

int
task_implementation_where (const_tree task_impl)
{
  tree impl_attr, args, where;

  gcc_assert (TREE_CODE (task_impl) == FUNCTION_DECL);

  impl_attr = lookup_attribute (task_implementation_attribute_name,
				DECL_ATTRIBUTES (task_impl));
  gcc_assert (impl_attr != NULL_TREE);

  args = TREE_VALUE (impl_attr);
  where = TREE_VALUE (args);

  return task_implementation_target_to_int (where);
}

/* Return the StarPU integer constant corresponding to string TARGET.  */

int
task_implementation_target_to_int (const_tree target)
{
  gcc_assert (TREE_CODE (target) == STRING_CST);

  int where_int;

  if (!strncmp (TREE_STRING_POINTER (target), "cpu",
		TREE_STRING_LENGTH (target)))
    where_int = STARPU_CPU;
  else if (!strncmp (TREE_STRING_POINTER (target), "opencl",
		     TREE_STRING_LENGTH (target)))
    where_int = STARPU_OPENCL;
  else if (!strncmp (TREE_STRING_POINTER (target), "cuda",
		     TREE_STRING_LENGTH (target)))
    where_int = STARPU_CUDA;
  else
    where_int = 0;

  return where_int;
}

/* Return the task implemented by TASK_IMPL.  */

tree
task_implementation_task (const_tree task_impl)
{
  tree impl_attr, args, task;

  gcc_assert (TREE_CODE (task_impl) == FUNCTION_DECL);

  impl_attr = lookup_attribute (task_implementation_attribute_name,
				DECL_ATTRIBUTES (task_impl));
  gcc_assert (impl_attr != NULL_TREE);

  args = TREE_VALUE (impl_attr);

  task = TREE_VALUE (TREE_CHAIN (args));
  if (task_implementation_p (task))
    /* TASK is an implicit CPU task implementation, so return its real
       task.  */
    return task_implementation_task (task);

  return task;
}

/* Return the declaration of the `struct starpu_codelet' variable associated with
   TASK_DECL.  */

tree
task_codelet_declaration (const_tree task_decl)
{
  tree cl_attr;

  cl_attr = lookup_attribute (task_codelet_attribute_name,
			      DECL_ATTRIBUTES (task_decl));
  gcc_assert (cl_attr != NULL_TREE);

  return TREE_VALUE (cl_attr);
}

/* Return the list of implementations of TASK_DECL.  */

tree
task_implementation_list (const_tree task_decl)
{
  tree attr;

  attr = lookup_attribute (task_implementation_list_attribute_name,
			   DECL_ATTRIBUTES (task_decl));
  return TREE_VALUE (attr);
}

/* Return the list of pointer parameter types of TASK_DECL.  */

tree
task_pointer_parameter_types (const_tree task_decl)
{
  return filter (pointer_type_p, TYPE_ARG_TYPES (TREE_TYPE (task_decl)));
}

/* Return a bitwise-or of the supported targets of TASK_DECL.  */

int
task_where (const_tree task_decl)
{
  gcc_assert (task_p (task_decl));

  int where;
  const_tree impl;

  for (impl = task_implementation_list (task_decl), where = 0;
       impl != NULL_TREE;
       impl = TREE_CHAIN (impl))
    where |= task_implementation_where (TREE_VALUE (impl));

  return where;
}

/* Return the FUNCTION_DECL of the wrapper generated for TASK_IMPL.  */

tree
task_implementation_wrapper (const_tree task_impl)
{
  tree attr;

  gcc_assert (TREE_CODE (task_impl) == FUNCTION_DECL);

  attr = lookup_attribute (task_implementation_wrapper_attribute_name,
			   DECL_ATTRIBUTES (task_impl));
  gcc_assert (attr != NULL_TREE);

  return TREE_VALUE (attr);
}

tree
codelet_type (void)
{
  /* XXX: Hack to allow the type declaration to be accessible at lower
     time.  */
  static tree type_decl = NULL_TREE;

  if (type_decl == NULL_TREE)
    /* Lookup the `struct starpu_codelet' struct type.  This should succeed since
       we push <starpu.h> early on.  */
    type_decl = type_decl_for_struct_tag (codelet_struct_tag);

  return TREE_TYPE (type_decl);
}

/* Return the access mode for POINTER, a PARM_DECL of a task.  */

enum starpu_data_access_mode
access_mode (const_tree type)
{
  gcc_assert (POINTER_TYPE_P (type));

  /* If TYPE points to a const-qualified type, then mark the data as
     read-only; if is has the `output' attribute, then mark it as write-only;
     otherwise default to read-write.  */
  return ((TYPE_QUALS (TREE_TYPE (type)) & TYPE_QUAL_CONST)
	  ? STARPU_R
	  : (output_type_p (type) ? STARPU_W : STARPU_RW));
}

/* Return true if TYPE is `output'-qualified.  */

bool
output_type_p (const_tree type)
{
  return (lookup_attribute (output_attribute_name,
			    TYPE_ATTRIBUTES (type)) != NULL_TREE);
}


/* Code generation.  */

/* Turn FN into a task, and push its associated codelet declaration.  */

void
taskify_function (tree fn)
{
  gcc_assert (TREE_CODE (fn) == FUNCTION_DECL);

  /* Add a `task' attribute and an empty `task_implementation_list'
     attribute.  */
  DECL_ATTRIBUTES (fn) =
    tree_cons (get_identifier (task_implementation_list_attribute_name),
	       NULL_TREE,
	       tree_cons (get_identifier (task_attribute_name), NULL_TREE,
			  DECL_ATTRIBUTES (fn)));

  /* Push a declaration for the corresponding `struct starpu_codelet' object and
     add it as an attribute of FN.  */
  tree cl = build_codelet_declaration (fn);
  DECL_ATTRIBUTES (fn) =
    tree_cons (get_identifier (task_codelet_attribute_name), cl,
	       DECL_ATTRIBUTES (fn));

  pushdecl (cl);
}


/* Return a NODE_IDENTIFIER for the variable holding the `struct starpu_codelet'
   structure associated with TASK_DECL.  */

tree
build_codelet_identifier (tree task_decl)
{
  static const char suffix[] = ".codelet";

  tree id;
  char *cl_name;
  const char *task_name;

  id = DECL_NAME (task_decl);
  task_name = IDENTIFIER_POINTER (id);

  cl_name = (char *) alloca (IDENTIFIER_LENGTH (id) + strlen (suffix) + 1);
  memcpy (cl_name, task_name, IDENTIFIER_LENGTH (id));
  strcpy (&cl_name[IDENTIFIER_LENGTH (id)], suffix);

  return get_identifier (cl_name);
}

/* Return a VAR_DECL that declares a `struct starpu_codelet' structure for
   TASK_DECL.  */

tree
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

/* Return a `struct starpu_codelet' initializer for TASK_DECL.  */

tree
build_codelet_initializer (tree task_decl)
{
  tree fields;

  fields = TYPE_FIELDS (codelet_type ());
  gcc_assert (TREE_CODE (fields) == FIELD_DECL);

  local_define (tree, lookup_field, (const char *name))
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
  };

  local_define (tree, field_initializer, (const char *name, tree value))
  {
    tree field, init;

    field = lookup_field (name);
    init = make_node (TREE_LIST);
    TREE_PURPOSE (init) = field;
    TREE_CHAIN (init) = NULL_TREE;

    if (TREE_CODE (TREE_TYPE (value)) != ARRAY_TYPE)
      TREE_VALUE (init) = fold_convert (TREE_TYPE (field), value);
    else
      TREE_VALUE (init) = value;

    return init;
  };

  local_define (tree, codelet_name, ())
  {
    const char *name = IDENTIFIER_POINTER (DECL_NAME (task_decl));
    return build_string_literal (strlen (name) + 1, name);
  };

  local_define (tree, where_init, (tree impls))
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

	if (verbose_output_p)
	  /* List the implementations of TASK_DECL.  */
	  inform (DECL_SOURCE_LOCATION (impl_decl),
		  "   %qE", DECL_NAME (impl_decl));

	where_int |= task_implementation_where (impl_decl);
      }

    return build_int_cstu (integer_type_node, where_int);
  };

  local_define (tree, implementation_pointers, (tree impls, int where))
  {
    size_t len;
    tree impl, pointers;

    for (impl = impls, pointers = NULL_TREE, len = 0;
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
	    pointers = tree_cons (size_int (len), addr, pointers);
	    len++;

	    if (len > STARPU_MAXIMPLEMENTATIONS)
	      error_at (DECL_SOURCE_LOCATION (impl_decl),
			"maximum number of per-target task implementations "
			"exceeded");
	  }
      }

    /* POINTERS must be null-terminated.  */
    pointers = tree_cons (size_int (len), build_zero_cst (ptr_type_node),
			  pointers);
    len++;

    /* Return an array initializer.  */
    tree index_type = build_index_type (size_int (list_length (pointers)));

    return build_constructor_from_list (build_array_type (ptr_type_node,
							  index_type),
					nreverse (pointers));
  };

  local_define (tree, pointer_arg_count, (void))
  {
    size_t len;

    len = list_length (task_pointer_parameter_types (task_decl));
    return build_int_cstu (integer_type_node, len);
  };

  local_define (tree, access_mode_array, (void))
  {
    const_tree type;
    tree modes;
    size_t index;

    for (type = task_pointer_parameter_types (task_decl),
	   modes = NULL_TREE, index = 0;
	 type != NULL_TREE && index < STARPU_NMAXBUFS;
	 type = TREE_CHAIN (type), index++)
      {
	tree value = build_int_cst (integer_type_node,
				    access_mode (TREE_VALUE (type)));

	modes = tree_cons (size_int (index), value, modes);
      }

    tree index_type = build_index_type (size_int (list_length (modes)));

    return build_constructor_from_list (build_array_type (integer_type_node,
							  index_type),
					nreverse (modes));
  };

  if (verbose_output_p)
    inform (DECL_SOURCE_LOCATION (task_decl),
	    "implementations for task %qE:", DECL_NAME (task_decl));

  tree impls, inits;

  impls = task_implementation_list (task_decl);

  inits =
    chain_trees (field_initializer ("name", codelet_name ()),
		 field_initializer ("where", where_init (impls)),
		 field_initializer ("nbuffers", pointer_arg_count ()),
		 field_initializer ("modes", access_mode_array ()),
		 field_initializer ("cpu_funcs",
				    implementation_pointers (impls,
							     STARPU_CPU)),
		 field_initializer ("opencl_funcs",
		 		    implementation_pointers (impls,
							     STARPU_OPENCL)),
		 field_initializer ("cuda_funcs",
		 		    implementation_pointers (impls,
							     STARPU_CUDA)),
		 NULL_TREE);

  return build_constructor_from_unsorted_list (codelet_type (), inits);
}

/* Return the VAR_DECL that defines a `struct starpu_codelet' structure for
   TASK_DECL.  The VAR_DECL is assumed to already exists, so it must not be
   pushed again.  */

tree
declare_codelet (tree task_decl)
{
  /* Retrieve the declaration of the `struct starpu_codelet' object.  */
  tree cl_decl;
  cl_decl = lookup_name (build_codelet_identifier (task_decl));
  gcc_assert (cl_decl != NULL_TREE && TREE_CODE (cl_decl) == VAR_DECL);

  /* Turn the codelet declaration into a definition.  */
  TREE_TYPE (cl_decl) = codelet_type ();
  TREE_PUBLIC (cl_decl) = TREE_PUBLIC (task_decl);

  return cl_decl;
}

/* Build the body of TASK_DECL, which will call `starpu_task_insert'.  */

void
define_task (tree task_decl)
{
  /* First of all, give TASK_DECL an argument list.  */
  DECL_ARGUMENTS (task_decl) = build_function_arguments (task_decl);

  VEC(tree, gc) *args = NULL;
  location_t loc = DECL_SOURCE_LOCATION (task_decl);
  tree p, params = DECL_ARGUMENTS (task_decl);

  /* The first argument will be a pointer to the codelet.  */

  VEC_safe_push (tree, gc, args,
		 build_addr (task_codelet_declaration (task_decl),
			     current_function_decl));

  for (p = params; p != NULL_TREE; p = TREE_CHAIN (p))
    {
      gcc_assert (TREE_CODE (p) == PARM_DECL);

      tree type = TREE_TYPE (p);

      if (POINTER_TYPE_P (type))
	{
	  /* A pointer: the arguments will be:
	     `STARPU_RW, ptr' or similar.  */

	  VEC_safe_push (tree, gc, args,
			 build_int_cst (integer_type_node,
					access_mode (type)));
	  VEC_safe_push (tree, gc, args, build_pointer_lookup (p));
	}
      else
	{
	  /* A scalar: the arguments will be:
	     `STARPU_VALUE, &scalar, sizeof (scalar)'.  */

	  mark_addressable (p);

	  VEC_safe_push (tree, gc, args,
			 build_int_cst (integer_type_node, STARPU_VALUE));
	  VEC_safe_push (tree, gc, args,
			 build_addr (p, current_function_decl));
	  VEC_safe_push (tree, gc, args,
			 size_in_bytes (type));
	}
    }

  /* Push the terminating zero.  */

  VEC_safe_push (tree, gc, args,
		 build_int_cst (integer_type_node, 0));

  /* Introduce a local variable to hold the error code.  */

  tree error_var = build_decl (loc, VAR_DECL,
  			       create_tmp_var_name (".task_insert_error"),
  			       integer_type_node);
  DECL_CONTEXT (error_var) = task_decl;
  DECL_ARTIFICIAL (error_var) = true;

  /* Build this:

       err = starpu_task_insert (...);
       if (err != 0)
         { printf ...; abort (); }
   */

  static tree task_insert_fn;
  LOOKUP_STARPU_FUNCTION (task_insert_fn, "starpu_task_insert");

  tree call = build_call_expr_loc_vec (loc, task_insert_fn, args);

  tree assignment = build2 (INIT_EXPR, TREE_TYPE (error_var),
  			    error_var, call);

  tree name = DECL_NAME (task_decl);
  tree cond = build3 (COND_EXPR, void_type_node,
		      build2 (NE_EXPR, boolean_type_node,
			      error_var, integer_zero_node),
		      build_error_statements (loc, error_var,
					      build_starpu_error_string,
					      "failed to insert task `%s'",
					      IDENTIFIER_POINTER (name)),
		      NULL_TREE);

  tree stmts = NULL;
  append_to_statement_list (assignment, &stmts);
  append_to_statement_list (cond, &stmts);

  tree bind = build3 (BIND_EXPR, void_type_node, error_var, stmts,
  		      NULL_TREE);

  /* Put it all together.  */

  DECL_SAVED_TREE (task_decl) = bind;
  TREE_STATIC (task_decl) = true;
  DECL_EXTERNAL (task_decl) = false;
  DECL_ARTIFICIAL (task_decl) = true;
  DECL_INITIAL (task_decl) =
    build_block (error_var, NULL_TREE, task_decl, NULL_TREE);
  DECL_RESULT (task_decl) =
    build_decl (loc, RESULT_DECL, NULL_TREE, void_type_node);
  DECL_CONTEXT (DECL_RESULT (task_decl)) = task_decl;
}

/* Add FN to the list of implementations of TASK_DECL.  */

void
add_task_implementation (tree task_decl, tree fn, const_tree where)
{
  location_t loc;
  tree attr, impls;

  attr = lookup_attribute (task_implementation_list_attribute_name,
			   DECL_ATTRIBUTES (task_decl));
  gcc_assert (attr != NULL_TREE);

  gcc_assert (TREE_CODE (where) == STRING_CST);

  loc = DECL_SOURCE_LOCATION (fn);

  impls = tree_cons (NULL_TREE, fn, TREE_VALUE (attr));
  TREE_VALUE (attr) = impls;

  TREE_USED (fn) = true;

  /* Check the `where' argument to raise a warning if needed.  */
  if (task_implementation_target_to_int (where) == 0)
    warning_at (loc, 0,
		"unsupported target %E; task implementation won't be used",
		where);
  else if (task_implementation_target_to_int (where) == STARPU_OPENCL)
    {
      local_define (void, validate, (tree t))
	{
	  validate_opencl_argument_type (loc, t);
	};

      for_each (validate, TYPE_ARG_TYPES (TREE_TYPE (fn)));
    }
}
