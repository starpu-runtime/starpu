@@
identifier c;
expression E;
@@
struct starpu_codelet c = {
-	.where = E
};

@@
struct starpu_codelet cl;
expression E;
@@
-cl.where = E;

@@
struct starpu_codelet*  pointer_to_cl;
expression E;
@@
- pointer_to_cl->where = E;
