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

static double scale = 0.0;
static unsigned long long residual = 0;

#if defined(__i386__) || defined(__pentium__) || defined(__pentiumpro__) || defined(__i586__) || defined(__i686__) || defined(__k6__) || defined(__k7__) || defined(__x86_64__)
#  define GET_TICK(t) __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#else
#  error "Processeur non-supporté par timing.h"
#endif

#define TICK_RAW_DIFF(t1, t2) ((t2).tick - (t1).tick)
#define TICK_DIFF(t1, t2) (TICK_RAW_DIFF(t1, t2) - residual)
#define TIMING_DELAY(t1, t2) tick2usec(TICK_DIFF(t1, t2))

void timing_init(void)
{
  static tick_t t1, t2;
  int i;
      
  residual = (unsigned long long)1 << 63;
  
  for(i = 0; i < 20; i++)
    {
      GET_TICK(t1);
      GET_TICK(t2);
      residual = min(residual, TICK_RAW_DIFF(t1, t2));
    }
  
  {
    struct timeval tv1,tv2;
    
    GET_TICK(t1);
    gettimeofday(&tv1,0);
    usleep(500000);
    GET_TICK(t2);
    gettimeofday(&tv2,0);
    scale = ((tv2.tv_sec*1e6 + tv2.tv_usec) -
	     (tv1.tv_sec*1e6 + tv1.tv_usec)) / 
      (double)(TICK_DIFF(t1, t2));
  }

}

double tick2usec(long long t)
{
  return (double)(t)*scale;
}

#endif /* TIMING_H */


