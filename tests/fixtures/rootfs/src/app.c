#include <stdio.h>
#include <string.h>
#include <stdlib.h>
const char *greet(const char *);
int vendor_sum(int, int);
int main(int argc, char **argv) {
  const char *who = argc > 1 ? argv[1] : "world";
  char *copy = strdup(who);
  printf("%s %d\n", greet(copy), vendor_sum(argc, (int)strlen(copy)));
  free(copy);
  return 0;
}
