#include "ruby.h"

#include <ruby.h>
#include <rubysig.h>

/*top*/
extern int t(void);
int main(int argc, char **argv)
{
  if (argc > 1000000) {
    int (* volatile tp)(void)=(int (*)(void))&t;
    printf("%d", (*tp)());
  }

  return !!argv[argc];
}
int t(void) { const volatile void *volatile p; p = &(&rb_trap_immediate)[0]; return !p; }
