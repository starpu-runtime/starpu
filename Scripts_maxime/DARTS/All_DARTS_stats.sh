# 2 scripts car python3 crash sinon à cuase de trop de threads.
# bash Scripts_maxime/DARTS/All_DARTS_stats.sh

N=2
#~ N=3
#~ N=6

#~ echo "1 GPU DARTS N =" $((N))
#~ bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DARTS gemini-1-fgcs 1
#~ bash Scripts_maxime/DARTS/Draw_DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DARTS gemini-1-fgcs 1
#~ echo "2 GPU DARTS N =" $((N))
#~ bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DARTS gemini-1-fgcs 2
#~ bash Scripts_maxime/DARTS/Draw_DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DARTS gemini-1-fgcs 2

echo "1 GPU DATA TASK ORDER N =" $((N))
bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DATA_TASK_ORDER gemini-1-fgcs 1
bash Scripts_maxime/DARTS/Draw_DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DATA_TASK_ORDER gemini-1-fgcs 1
#~ echo "2 GPU DATA TASK ORDER N =" $((N))
#~ bash Scripts_maxime/DARTS/DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DATA_TASK_ORDER gemini-1-fgcs 2
#~ bash Scripts_maxime/DARTS/Draw_DARTS_stats.sh /home/gonthier/ /home/gonthier/these_gonthier_maxime/Starpu/ $((N)) Cholesky_dependances DATA_TASK_ORDER gemini-1-fgcs 2
