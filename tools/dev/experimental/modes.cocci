@find_cl@
identifier c;
@@
struct starpu_codelet c =
{
};

@find_task depends on find_cl@
identifier t;
identifier find_cl.c;
@@
t->cl = &c; 


/*
 * Remove task->buffers[id].mode = STARPU_{R,W,RW}
 * Replace task->buffers[id].handle = handle; by  
 *      task->handles[id] = handle;
 */
@remove_task_mode depends on find_task@
identifier find_task.t;
expression E;
identifier h;
expression id; 
identifier find_cl.c;
@@
(
- t->buffers[id].handle = h;
++ t->handles[id] = h;
|
- t->buffers[id].mode = E;
)

@has_modes depends on remove_task_mode@
identifier find_cl.c;
expression remove_task_mode.id;
expression remove_task_mode.E;
@@
struct starpu_codelet c =
{
++	.modes[id] = E,
};
