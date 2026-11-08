#include <stdio.h>
int greet_count = 0;
const char *greet(const char *who) { greet_count++; static char buf[64]; snprintf(buf, sizeof buf, "hello %s", who); return buf; }
int vendor_sum(int a, int b) { return a + b; }
