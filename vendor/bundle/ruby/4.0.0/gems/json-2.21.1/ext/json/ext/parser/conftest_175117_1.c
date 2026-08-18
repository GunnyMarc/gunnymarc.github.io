#include "ruby.h"

#include <arm_neon.h>

int main(int argc, char **argv) {
  uint8x16_t test = vdupq_n_u8(32);

  if (argc > 100000) printf("%p", &test);
  return 0;
}
