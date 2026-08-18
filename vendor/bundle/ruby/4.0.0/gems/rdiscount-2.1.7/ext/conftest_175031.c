#include "ruby.h"

typedef unsigned int rbcv_typedef_;
rbcv_typedef_ *rbcv_ptr_;

#include <stdio.h>
/*top*/
typedef unsigned
#ifdef PRI_LL_PREFIX
#define PRI_CONFTEST_PREFIX PRI_LL_PREFIX
LONG_LONG
#else
#define PRI_CONFTEST_PREFIX "l"
long
#endif
conftest_type;
conftest_type conftest_const = (conftest_type)(sizeof((*rbcv_ptr_)));
int main() {printf("%"PRI_CONFTEST_PREFIX"u\n", conftest_const); return 0;}
