#ifndef TIMING_H
#define TIMING_H

/*
 * -- Initialiser la bibliothèque avec timing_init();
 * -- Mémoriser un timestamp :
 *  tick_t t;
 *  GET_TICK(t);
 * -- Calculer un intervalle en microsecondes :
 *  TIMING_DELAY(t1, t2);
 */

#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>

#ifndef min
#define min(a,b) \
	({__typeof__ ((a)) _a = (a); \
	  __typeof__ ((b)) _b = (b); \
	  _a < _b ? _a : _b; })
#endif

typedef union u_tick
{
  uint64_t tick;

  struct
  {
    uint32_t low;
    uint32_t high;
  }
  sub;
} tick_t;

#if defined(__i386__) || defined(__pentium__) || defined(__pentiumpro__) || defined(__i586__) || defined(__i686__) || defined(__k6__) || defined(__k7__) || defined(__x86_64__)
#  define GET_TICK(t) __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#else
//#  error "Processeur non-supporté par timing.h"
/* XXX */
//#warning "unsupported processor GET_TICK returns 0"
#  define GET_TICK(t) do {} while(0);
#endif

void __attribute__ ((unused)) timing_init(void);
double __attribute__ ((unused)) tick2usec(long long t);
double __attribute__ ((unused)) timing_delay(tick_t *t1, tick_t *t2);

double __attribute__ ((unused)) timing_now(void);

#endif /* TIMING_H */


