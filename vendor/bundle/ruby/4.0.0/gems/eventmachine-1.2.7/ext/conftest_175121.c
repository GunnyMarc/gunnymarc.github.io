#include "ruby.h"

#include <ruby/intern.h>

/*top*/
typedef rb_fdset_t conftest_type;
int conftestval[sizeof(conftest_type)?1:-1];
