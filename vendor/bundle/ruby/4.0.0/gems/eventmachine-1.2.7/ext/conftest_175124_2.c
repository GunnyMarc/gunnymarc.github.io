#include "ruby.h"

#include <sys/syscall.h>
/*top*/
#ifndef __NR_inotify_init
# error
|:/ === __NR_inotify_init undefined === /:|
#endif
