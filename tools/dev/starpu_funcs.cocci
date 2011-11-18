@starpufunc@
position p;
type t;
identifier f ~= "^starpu_";
@@

t f@p( ... );

@ script:python @
p << starpufunc.p;
f << starpufunc.f;
@@
print "%s,%s:%s" % (f,p[0].file,p[0].line)
