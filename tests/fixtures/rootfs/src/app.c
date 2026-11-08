#include <stdio.h>
#include <string.h>
#include <stdlib.h>
const char *greet(const char *);
int vendor_sum(int, int);
/* Firmware can optionally supply a startup hook in another shared library. */
extern void firmware_startup_hook(void) __attribute__((weak));
int main(int argc, char **argv) {
  if (firmware_startup_hook)
    firmware_startup_hook();
  const char *who = argc > 1 ? argv[1] : "world";
  char *copy = strdup(who);
  printf("%s %d\n", greet(copy), vendor_sum(argc, (int)strlen(copy)));
  free(copy);
  return 0;
}
