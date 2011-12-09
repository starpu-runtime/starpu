// I have no idea in Hell why we need "<... ...>", but this will not work with
// "... ..."
@@
@@
main(...)
{
<...
-	return 0;
+	return EXIT_SUCCESS;
...>
}


//
// Is (..)*/common/helper.h included ?
// XXX : OK, that suxx, but it should work. Is there a way to use a regular
// expression to match a header ?
@helper_included@
@@
(
#include "common/helper.h"
|
#include "../common/helper.h"
|
#include "../../common/helper.h"
|
#include "../../../common/helper.h"
)


@depends on helper_included@
@@
main(...)
{
...
-	return 77;
+	return STARPU_TEST_SKIPPED;
...
}
