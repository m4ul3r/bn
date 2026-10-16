/* Rebuild bw_consumer_demand at -O0 without debug info (-g0): BN recovers the
 * stack objects itself, so no store proof completes a slice. */
#include "bw_consumer_demand.c"
