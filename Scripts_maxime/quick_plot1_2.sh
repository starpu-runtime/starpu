# bash Scripts_maxime/quick_plot1_2.sh 1 1920 12 2000 best_ones
# bash Scripts_maxime/quick_plot1_2.sh 2 1920 12 2000 best_ones
# bash Scripts_maxime/quick_plot1_2.sh 4 1920 12 2000 best_ones
# bash Scripts_maxime/quick_plot1_2.sh 8 1920 12 2000 best_ones

# bash Scripts_maxime/quick_plot1_2.sh 1 1920 12 2000 DataInMemAndNotUsedYet
# bash Scripts_maxime/quick_plot1_2.sh 2 1920 12 2000 DataInMemAndNotUsedYet
# bash Scripts_maxime/quick_plot1_2.sh 4 1920 12 2000 DataInMemAndNotUsedYet
# bash Scripts_maxime/quick_plot1_2.sh 8 1920 12 2000 DataInMemAndNotUsedYet

# bash Scripts_maxime/quick_plot1_2.sh 1 1920 12 2000 TransferTimeOrder
# bash Scripts_maxime/quick_plot1_2.sh 2 1920 12 2000 TransferTimeOrder
# bash Scripts_maxime/quick_plot1_2.sh 4 1920 12 2000 TransferTimeOrder
# bash Scripts_maxime/quick_plot1_2.sh 8 1920 12 2000 TransferTimeOrder
# bash Scripts_maxime/quick_plot1_2.sh 8 1920 18 2000 TransferTimeOrder

# bash Scripts_maxime/quick_plot1_2.sh 1 1920 12 2000 GpuChoiceFreeTask
# bash Scripts_maxime/quick_plot1_2.sh 2 1920 12 2000 GpuChoiceFreeTask
# bash Scripts_maxime/quick_plot1_2.sh 4 1920 12 2000 GpuChoiceFreeTask
# bash Scripts_maxime/quick_plot1_2.sh 8 1920 12 2000 GpuChoiceFreeTask

NGPU=$1
TAILLE_TUILE=$2
NB_TAILLE_TESTE=$3
CM=$4
MODEL=$5

bash Scripts_maxime/quick_plot1.sh $((NGPU)) $((TAILLE_TUILE)) $((NB_TAILLE_TESTE)) $((CM)) ${MODEL}
bash Scripts_maxime/quick_plot2.sh $((NGPU)) $((TAILLE_TUILE)) $((NB_TAILLE_TESTE)) $((CM)) ${MODEL}
