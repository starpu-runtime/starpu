#~ bash Scripts_maxime/Script_test_thomas.sh

start=`date +%s`

ulimit -S -s 5000000

# Pour indiquer le dossier des perfmodels
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

# Taille de la matrice
N=50

# Nombre de GPUs
NGPU=1

# Scheduler
#~ ORDO="dmdar"
#~ ORDO="modular-eager-prefetching"
ORDO="modular-heft"

# Limite de la mémoire en MB
CM=500
#~ CM=250

# Activation de l'heuristique ready. Elle est toujours activé pour DMDAR
#~ READY=0
READY=1

# Threshold du nombre de tâches dans que l'on peut prefetch
TH=10

# Nombres de tâches soumises au GPU à l'avance
CP=5

# Perfmodel
HOST="gemini-1-fgcs"
#~ HOST="attila"

# Pour générer ou non une trace. Attention sur ma branche il faut recompiler avec ./configure --enable-simgrid --disable-mpi --with-simgrid-dir=/path_simgrid/simgrid --with-fxt=/path_fxt/fxt
TRACE=0
#~ TRACE=1

# Différentes applications
# Multiplication de matrices 2D
#~ ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1
# Multiplication de matrices 3D
#~ ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1
# Cholesky
#~ ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))

STARPU_SCHED=${ORDO} STARPU_LIMIT_CUDA_MEM=$((CM)) STARPU_GENERATE_TRACE=$((TRACE)) STARPU_SCHED_READY=$((READY)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -xy $((960*N)) -nblocks $((N)) -iter 1

end=`date +%s` 
runtime=$((end-start))
echo "Fin du script, l'execution a durée" $((runtime/60))" min "$((runtime%60))" sec."
