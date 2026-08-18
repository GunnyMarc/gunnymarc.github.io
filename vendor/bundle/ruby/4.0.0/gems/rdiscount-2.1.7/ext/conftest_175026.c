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
int t(void) { void ((*volatile p)()); p = (void ((*)()))rand; return !p; }
