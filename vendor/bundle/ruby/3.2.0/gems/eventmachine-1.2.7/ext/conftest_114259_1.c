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
extern void rb_wait_for_single_fd();
int t(void) { rb_wait_for_single_fd(); return 0; }
