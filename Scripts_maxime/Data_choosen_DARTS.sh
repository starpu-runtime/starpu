# bash Scripts_maxime/Data_choosen_DARTS.sh Matrice3D 5
# bash Scripts_maxime/Data_choosen_DARTS.sh Cholesky 5

DOSSIER=$1

make -j 6
ulimit -S -s 5000000
export STARPU_PERF_MODEL_DIR=tools/perfmodels/sampling

N=$2

NGPU=1

ORDO="dynamic-data-aware"

CM=500

#~ EVICTION=0
EVICTION=1

READY=0
#~ READY=1

TH=10

CP=5

HOST="gemini-1-fgcs"

SIMMEM=0

FROMMEM=1

if [ $DOSSIER = "Matrice3D" ]
then
	SIMULATE_MEMORY=$((SIMMEM)) CHOOSE_BEST_DATA_FROM=$((FROMMEM)) STARPU_SCHED=${ORDO} APP=1 STARPU_LIMIT_CUDA_MEM=$((CM)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/mult/sgemm -3d -xy $((960*N)) -nblocks $((N)) -nblocksz $((4)) -iter 1 
fi
if [ $DOSSIER = "Cholesky" ]
then
	SIMULATE_MEMORY=$((SIMMEM)) CHOOSE_BEST_DATA_FROM=$((FROMMEM)) STARPU_SCHED=${ORDO} APP=1 STARPU_LIMIT_CUDA_MEM=$((CM)) EVICTION_STRATEGY_DYNAMIC_DATA_AWARE=$((EVICTION)) STARPU_NTASKS_THRESHOLD=$((TH)) STARPU_CUDA_PIPELINE=$((CP)) STARPU_MINIMUM_CLEAN_BUFFERS=0 STARPU_TARGET_CLEAN_BUFFERS=0 STARPU_NCPU=0 STARPU_NCUDA=$((NGPU)) STARPU_NOPENCL=0 STARPU_HOSTNAME=${HOST} ./examples/cholesky/cholesky_implicit -size $((960*N)) -nblocks $((N))
fi

python3 /home/gonthier/these_gonthier_maxime/Code/Barplot_DARTS.py Output_maxime/DARTS_data_choosen_stats.csv

mv Output_maxime/DARTS_data_choosen_stats.csv /home/gonthier/these_gonthier_maxime/Starpu/R/Data/${DOSSIER}/DARTS_data_choosen_stats_N${N}_SIMMEM${SIMMEM}_FROMMEM${FROMMEM}.csv

mv plot.pdf /home/gonthier/these_gonthier_maxime/Starpu/R/Courbes/${DOSSIER}/DARTS_data_choosen_stats_N${N}_SIMMEM${SIMMEM}_FROMMEM${FROMMEM}.pdf
