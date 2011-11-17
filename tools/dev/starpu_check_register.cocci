@initialize:python@
handles = {}


@select@
position p;
identifier f =~ "^starpu_.*_data_register$";
identifier e;
@@
<...
f@p( &e, ... );
...>
@script:python@
p << select.p;
f << select.f;
e << select.e;
@@
s = "%s(%s),%s:%s" % (f,e,p[0].file,p[0].line)
# hack: 'clean' the string e from unwanted non printing characters, otherwise 'e' in select rule does not match 'e' in check rule
e = "%s" % e
handles[e]=s


@check@
position p;
identifier select.e;
@@
<...
starpu_data_unregister@p( e );
...>
@script:python@
e << select.e;
p << check.p;
@@
# hack: position p must be defined in the check rule even though it is not used, otherwise the 'check' python script is not run
e = "%s" % e
if e in handles:
        del handles[e]


@finalize:python@
for s in handles.values():
        print s

