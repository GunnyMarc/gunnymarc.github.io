#include "ruby.h"

#include <sys/inotify.h>

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
extern void inotify_init();
int t(void) { inotify_init(); return 0; }
