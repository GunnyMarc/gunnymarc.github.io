#include "ruby.h"

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
extern void rb_enable_interrupt();
int t(void) { rb_enable_interrupt(); return 0; }
