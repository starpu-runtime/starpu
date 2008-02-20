#ifndef __PARAMETERS_H__

#define BLOCKDIMX       16
#define BLOCKDIMY       16
#define GRIDDIMX        4
#define GRIDDIMY        4


// to speed up speedup measurements ;)
#ifndef SEQFACTOR
#define SEQFACTOR 1
#endif

#ifndef SIZE
#define SIZE       2048
#endif

#ifndef GRAIN
#define GRAIN   128
#endif

#endif // __PARAMETERS_H__
