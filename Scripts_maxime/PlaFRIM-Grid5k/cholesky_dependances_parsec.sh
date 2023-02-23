# bash cholesky_dependances_parsec.sh 1 1920 12 2000

# Attention si je veux changer la mémoire je dois tester en commantant la dernière ligne du fichier conf de Mathieu!!

if [ $# != 4 ]
then
	echo "Arguments must be: bash cholesky_dependances_parsec.sh NGPU TAILLE_TUILE NB_TAILLE_TESTE MEMOIRE"
	exit
fi

module load intel-oneapi-mkl/2022.0.2_gcc-10.2.0-openmpi
make -j 6
START_X=0
ECHELLE_X=5
NGPU=$1
TAILLE_TUILE=$2
NB_TAILLE_TESTE=$3
CM=$4
FICHIER_RAW=parsec_${TAILLE_TUILE}_${NGPU}GPU_${CM}Mo.csv
truncate -s 0 ${FICHIER_RAW}

echo "CM =" ${CM} "BLOCK SIZE =" ${TAILLE_TUILE} "NGPU =" ${NGPU}

for ((i=1 ; i<=(($NB_TAILLE_TESTE)); i++))
do
	N=$((START_X+i*ECHELLE_X))
	echo "N=${N}"
	mpiexec -n 1 dplasma/builddir/tests/testing_spotrf -t $((TAILLE_TUILE)) -T $((TAILLE_TUILE)) -N $((TAILLE_TUILE*N)) -g $((NGPU)) >> ${FICHIER_RAW}
done
