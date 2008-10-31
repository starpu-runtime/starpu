#include "timing.h"

#define TICK_RAW_DIFF(t1, t2) ((t2).tick - (t1).tick)
#define TICK_DIFF(t1, t2) (TICK_RAW_DIFF(t1, t2) - residual)
#define TIMING_DELAY(t1, t2) tick2usec(TICK_DIFF(t1, t2))

static double scale = 0.0;
static unsigned long long residual = 0;

static int inited = 0;

void timing_init(void)
{
  static tick_t t1, t2;
  int i;

  if (inited) return;

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

  inited = 1;
}

double tick2usec(long long t)
{
  return (double)(t)*scale;
}

double timing_delay(tick_t *t1, tick_t *t2)
{
	return TIMING_DELAY(*t1, *t2);
}

double timing_now(void)
{
	tick_t tick_now;
	GET_TICK(tick_now);

	return tick2usec(tick_now.tick);
}
