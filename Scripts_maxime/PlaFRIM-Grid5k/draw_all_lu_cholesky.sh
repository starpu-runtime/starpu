#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 5 2000 best_ones 4 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 12 5 2000 best_ones 4 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 12 5 2000 best_ones 4 Cholesky_dependances
bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 15 5 2000 best_ones 4 Cholesky_dependances

#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 12 5 32000 best_ones 4 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 12 5 32000 best_ones 4 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 12 5 32000 best_ones 4 Cholesky_dependances
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 15 5 32000 best_ones 4 Cholesky_dependances


#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 7 6 2000 best_ones 4 LU
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 7 6 2000 best_ones 4 LU
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 7 6 2000 best_ones 4 LU
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 10 6 2000 best_ones 4 LU

#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 1920 10 6 32000 best_ones 4 LU
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 2 1920 7 6 32000 best_ones 4 LU
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 4 1920 7 6 32000 best_ones 4 LU
#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 8 1920 7 6 32000 best_ones 4 LU

#~ bash Scripts_maxime/PlaFRIM-Grid5k/draw_cholesky_dependances.sh 1 4096 5 2 1000 best_ones 4 out_of_core_lu

# CHO
#~ mgonthier@gemini-1:~$ mpiexec -n 1 dplasma/builddir/tests/testing_spotrf -t 1920 -T 1920 -N $((1920*65)) -g 8 --nruns 11 --scheduler=AP
#~ [****] TIME(s)     30.69432 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   21109.100610 gflops - ENQ&PROG&DEST     37.23714 :   17400.087411 gflops - ENQ      5.60608 - DEST      0.93674
#~ [****] TIME(s)     26.61078 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   24348.379059 gflops - ENQ&PROG&DEST     33.93333 :   19094.189255 gflops - ENQ      6.38387 - DEST      0.93868
#~ [****] TIME(s)     26.56015 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   24394.794021 gflops - ENQ&PROG&DEST     32.92638 :   19678.123888 gflops - ENQ      5.43039 - DEST      0.93585
#~ [****] TIME(s)     26.69161 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   24274.647570 gflops - ENQ&PROG&DEST     33.23446 :   19495.710814 gflops - ENQ      5.59532 - DEST      0.94753
#~ [****] TIME(s)     27.64550 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   23437.064532 gflops - ENQ&PROG&DEST     34.08027 :   19011.862446 gflops - ENQ      5.49733 - DEST      0.93744
#~ [****] TIME(s)     25.86084 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   25054.458897 gflops - ENQ&PROG&DEST     32.44335 :   19971.102989 gflops - ENQ      5.64092 - DEST      0.94159
#~ [****] TIME(s)     26.66768 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   24296.432868 gflops - ENQ&PROG&DEST     33.10119 :   19574.204881 gflops - ENQ      5.46951 - DEST      0.96400
#~ [****] TIME(s)     25.71906 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   25192.579565 gflops - ENQ&PROG&DEST     32.49839 :   19937.276252 gflops - ENQ      5.83725 - DEST      0.94208
#~ [****] TIME(s)     25.99770 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   24922.565152 gflops - ENQ&PROG&DEST     33.02079 :   19621.862225 gflops - ENQ      6.07477 - DEST      0.94832
#~ [****] TIME(s)     25.82976 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   25084.609634 gflops - ENQ&PROG&DEST     32.37448 :   20013.584810 gflops - ENQ      5.60694 - DEST      0.93778
#~ [****] TIME(s)     28.35297 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   22852.261764 gflops - ENQ&PROG&DEST     34.70295 :   18670.728936 gflops - ENQ      5.41158 - DEST      0.93841
#~ +----------------------------------------------------------------------------------------------------------------------------+
#~ |         |                    |                       Data In                              |         Data Out               |
#~ |Rank   0 |  # KERNEL |    %   |  Required  |   Transfered H2D(%)   |   Transfered D2D(%)   |  Required  |   Transfered(%)   |
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |  Dev  0 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | default
#~ |  Dev  1 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | recursive
#~ |  Dev  2 |     65910 |  12.51 |  1215.57GB |    1215.57GB(100.00)   |       0.00 B( 0.00)   |   905.14GB |   409.60GB(45.25) | Tesla V100-SXM2-32GB: cuda(0)
#~ |  Dev  3 |     65781 |  12.48 |  1211.35GB |    1211.35GB(100.00)   |       0.00 B( 0.00)   |   903.36GB |   405.40GB(44.88) | Tesla V100-SXM2-32GB: cuda(1)
#~ |  Dev  4 |     65517 |  12.43 |  1256.03GB |    1256.03GB(100.00)   |       0.00 B( 0.00)   |   899.74GB |   393.52GB(43.74) | Tesla V100-SXM2-32GB: cuda(2)
#~ |  Dev  5 |     65533 |  12.44 |  1284.63GB |    1284.63GB(100.00)   |       0.00 B( 0.00)   |   899.96GB |   404.13GB(44.91) | Tesla V100-SXM2-32GB: cuda(3)
#~ |  Dev  6 |     66020 |  12.53 |  1224.41GB |    1224.41GB(100.00)   |       0.00 B( 0.00)   |   906.65GB |   406.91GB(44.88) | Tesla V100-SXM2-32GB: cuda(4)
#~ |  Dev  7 |     66630 |  12.64 |  1219.67GB |    1219.67GB(100.00)   |       0.00 B( 0.00)   |   915.02GB |   402.87GB(44.03) | Tesla V100-SXM2-32GB: cuda(5)
#~ |  Dev  8 |     66039 |  12.53 |  1235.03GB |    1235.03GB(100.00)   |       0.00 B( 0.00)   |   906.91GB |   403.32GB(44.47) | Tesla V100-SXM2-32GB: cuda(6)
#~ |  Dev  9 |     65525 |  12.43 |  1268.76GB |    1268.76GB(100.00)   |       0.00 B( 0.00)   |   899.85GB |   396.28GB(44.04) | Tesla V100-SXM2-32GB: cuda(7)
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |All Devs |    526955 | 100.00 |  9915.45GB |    9915.45GB(100.00)   |       0.00 B( 0.00)   |  7236.63GB |  3222.02GB(44.52) |
#~ +----------------------------------------------------------------------------------------------------------------------------+

#~ Full transfer matrix:
#~ dst\src          0          1          2          3          4          5          6          7          8          9 
   #~ 0        -          0.00 B   409.60GB   405.40GB   393.52GB   404.13GB   406.91GB   402.87GB   403.32GB   396.28GB
   #~ 1        0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 2     1215.57GB     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 3     1211.35GB     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 4     1256.03GB     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 5     1284.63GB     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B
   #~ 6     1224.41GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B
   #~ 7     1219.67GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B
   #~ 8     1235.03GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B
   #~ 9     1268.76GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -     
#~ mgonthier@gemini-1:~$ mpiexec -n 1 dplasma/builddir/tests/testing_spotrf -t 1920 -T 1920 -N $((1920*70)) -g 8 --nruns 11 --scheduler=AP
#~ [****] TIME(s)     37.09740 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   21814.130895 gflops - ENQ&PROG&DEST     44.64267 :   18127.219994 gflops - ENQ      6.44601 - DEST      1.09927
#~ [****] TIME(s)     34.40438 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   23521.644389 gflops - ENQ&PROG&DEST     41.97228 :   19280.524688 gflops - ENQ      6.48009 - DEST      1.08781
#~ [****] TIME(s)     34.56211 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   23414.299262 gflops - ENQ&PROG&DEST     42.02445 :   19256.587445 gflops - ENQ      6.37117 - DEST      1.09117
#~ [****] TIME(s)     34.60372 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   23386.144380 gflops - ENQ&PROG&DEST     42.12632 :   19210.022002 gflops - ENQ      6.43544 - DEST      1.08717
#~ [****] TIME(s)     35.08235 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   23067.085397 gflops - ENQ&PROG&DEST     42.54383 :   19021.502085 gflops - ENQ      6.33216 - DEST      1.12932
#~ [****] TIME(s)     33.79534 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   23945.535659 gflops - ENQ&PROG&DEST     41.81720 :   19352.025247 gflops - ENQ      6.92739 - DEST      1.09447
#~ [****] TIME(s)     35.48479 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   22805.477794 gflops - ENQ&PROG&DEST     42.90695 :   18860.522889 gflops - ENQ      6.32973 - DEST      1.09243
#~ [****] TIME(s)     33.43913 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   24200.619236 gflops - ENQ&PROG&DEST     41.08959 :   19694.708769 gflops - ENQ      6.55186 - DEST      1.09860
#~ [****] TIME(s)     33.77156 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   23962.396118 gflops - ENQ&PROG&DEST     41.17157 :   19655.495969 gflops - ENQ      6.31086 - DEST      1.08915
#~ [****] TIME(s)     34.90012 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   23187.531079 gflops - ENQ&PROG&DEST     42.30209 :   19130.202491 gflops - ENQ      6.31114 - DEST      1.09083
#~ [****] TIME(s)     33.01278 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   24513.161460 gflops - ENQ&PROG&DEST     40.50282 :   19980.028838 gflops - ENQ      6.40056 - DEST      1.08949
#~ +----------------------------------------------------------------------------------------------------------------------------+
#~ |         |                    |                       Data In                              |         Data Out               |
#~ |Rank   0 |  # KERNEL |    %   |  Required  |   Transfered H2D(%)   |   Transfered D2D(%)   |  Required  |   Transfered(%)   |
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |  Dev  0 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | default
#~ |  Dev  1 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | recursive
#~ |  Dev  2 |     81926 |  12.49 |  1544.65GB |    1544.65GB(100.00)   |       0.00 B( 0.00)   |  1125.08GB |   549.49GB(48.84) | Tesla V100-SXM2-32GB: cuda(0)
#~ |  Dev  3 |     81975 |  12.50 |  1619.74GB |    1619.74GB(100.00)   |       0.00 B( 0.00)   |  1125.76GB |   565.11GB(50.20) | Tesla V100-SXM2-32GB: cuda(1)
#~ |  Dev  4 |     81507 |  12.42 |  1603.98GB |    1603.98GB(100.00)   |       0.00 B( 0.00)   |  1119.33GB |   545.38GB(48.72) | Tesla V100-SXM2-32GB: cuda(2)
#~ |  Dev  5 |     81396 |  12.41 |  1570.50GB |    1570.50GB(100.00)   |       0.00 B( 0.00)   |  1117.80GB |   554.43GB(49.60) | Tesla V100-SXM2-32GB: cuda(3)
#~ |  Dev  6 |     83404 |  12.71 |  1704.56GB |    1704.56GB(100.00)   |       0.00 B( 0.00)   |  1145.38GB |   571.36GB(49.88) | Tesla V100-SXM2-32GB: cuda(4)
#~ |  Dev  7 |     81835 |  12.47 |  1593.22GB |    1593.22GB(100.00)   |       0.00 B( 0.00)   |  1123.83GB |   545.36GB(48.53) | Tesla V100-SXM2-32GB: cuda(5)
#~ |  Dev  8 |     82316 |  12.55 |  1660.19GB |    1660.19GB(100.00)   |       0.00 B( 0.00)   |  1130.44GB |   553.96GB(49.00) | Tesla V100-SXM2-32GB: cuda(6)
#~ |  Dev  9 |     81681 |  12.45 |  1611.85GB |    1611.85GB(100.00)   |       0.00 B( 0.00)   |  1121.72GB |   539.52GB(48.10) | Tesla V100-SXM2-32GB: cuda(7)
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |All Devs |    656040 | 100.00 | 12908.67GB |   12908.67GB(100.00)   |       0.00 B( 0.00)   |  9009.34GB |  4424.61GB(49.11) |
#~ +----------------------------------------------------------------------------------------------------------------------------+

#~ Full transfer matrix:
#~ dst\src          0          1          2          3          4          5          6          7          8          9 
   #~ 0        -          0.00 B   549.49GB   565.11GB   545.38GB   554.43GB   571.36GB   545.36GB   553.96GB   539.52GB
   #~ 1        0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 2     1544.65GB     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 3     1619.74GB     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 4     1603.98GB     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 5     1570.50GB     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B
   #~ 6     1704.56GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B
   #~ 7     1593.22GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B
   #~ 8     1660.19GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B
   #~ 9     1611.85GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -     
#~ mgonthier@gemini-1:~$ mpiexec -n 1 dplasma/builddir/tests/testing_spotrf -t 1920 -T 1920 -N $((1920*75)) -g 8 --nruns 11 --scheduler=AP
#~ [****] TIME(s)     49.59782 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   20068.186524 gflops - ENQ&PROG&DEST     58.27555 :   17079.863426 gflops - ENQ      7.42388 - DEST      1.25385
#~ [****] TIME(s)     46.39170 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   21455.095508 gflops - ENQ&PROG&DEST     55.07229 :   18073.304812 gflops - ENQ      7.42943 - DEST      1.25116
#~ [****] TIME(s)     44.61786 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22308.071303 gflops - ENQ&PROG&DEST     53.23556 :   18696.870806 gflops - ENQ      7.36598 - DEST      1.25172
#~ [****] TIME(s)     44.93785 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22149.222958 gflops - ENQ&PROG&DEST     53.63282 :   18558.383620 gflops - ENQ      7.43456 - DEST      1.26041
#~ [****] TIME(s)     44.05654 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22592.295206 gflops - ENQ&PROG&DEST     52.73882 :   18872.973293 gflops - ENQ      7.42911 - DEST      1.25317
#~ [****] TIME(s)     44.13929 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22549.940705 gflops - ENQ&PROG&DEST     53.64595 :   18553.839341 gflops - ENQ      8.25581 - DEST      1.25085
#~ [****] TIME(s)     43.79558 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22726.914352 gflops - ENQ&PROG&DEST     52.63078 :   18911.715339 gflops - ENQ      7.58503 - DEST      1.25017
#~ [****] TIME(s)     45.44545 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   21901.825245 gflops - ENQ&PROG&DEST     54.07547 :   18406.468173 gflops - ENQ      7.37001 - DEST      1.26001
#~ [****] TIME(s)     44.43448 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22400.133896 gflops - ENQ&PROG&DEST     53.02856 :   18769.855177 gflops - ENQ      7.34195 - DEST      1.25212
#~ [****] TIME(s)     43.54799 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22856.126975 gflops - ENQ&PROG&DEST     52.29507 :   19033.119105 gflops - ENQ      7.49154 - DEST      1.25554
#~ [****] TIME(s)     44.33499 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   22450.403914 gflops - ENQ&PROG&DEST     52.97663 :   18788.253427 gflops - ENQ      7.36033 - DEST      1.28131
#~ +----------------------------------------------------------------------------------------------------------------------------+
#~ |         |                    |                       Data In                              |         Data Out               |
#~ |Rank   0 |  # KERNEL |    %   |  Required  |   Transfered H2D(%)   |   Transfered D2D(%)   |  Required  |   Transfered(%)   |
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |  Dev  0 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | default
#~ |  Dev  1 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | recursive
#~ |  Dev  2 |    100490 |  12.49 |  2103.10GB |    2103.10GB(100.00)   |       0.00 B( 0.00)   |  1380.02GB |   780.14GB(56.53) | Tesla V100-SXM2-32GB: cuda(0)
#~ |  Dev  3 |    100054 |  12.43 |  2078.57GB |    2078.57GB(100.00)   |       0.00 B( 0.00)   |  1374.03GB |   778.53GB(56.66) | Tesla V100-SXM2-32GB: cuda(1)
#~ |  Dev  4 |    100229 |  12.46 |  2147.57GB |    2147.57GB(100.00)   |       0.00 B( 0.00)   |  1376.44GB |   771.32GB(56.04) | Tesla V100-SXM2-32GB: cuda(2)
#~ |  Dev  5 |    100622 |  12.51 |  2153.50GB |    2153.50GB(100.00)   |       0.00 B( 0.00)   |  1381.83GB |   785.32GB(56.83) | Tesla V100-SXM2-32GB: cuda(3)
#~ |  Dev  6 |    100462 |  12.49 |  2162.77GB |    2162.77GB(100.00)   |       0.00 B( 0.00)   |  1379.64GB |   779.88GB(56.53) | Tesla V100-SXM2-32GB: cuda(4)
#~ |  Dev  7 |    101411 |  12.60 |  2177.94GB |    2177.94GB(100.00)   |       0.00 B( 0.00)   |  1392.67GB |   791.96GB(56.87) | Tesla V100-SXM2-32GB: cuda(5)
#~ |  Dev  8 |    101367 |  12.60 |  2173.10GB |    2173.10GB(100.00)   |       0.00 B( 0.00)   |  1392.06GB |   779.47GB(55.99) | Tesla V100-SXM2-32GB: cuda(6)
#~ |  Dev  9 |    100015 |  12.43 |  2137.73GB |    2137.73GB(100.00)   |       0.00 B( 0.00)   |  1373.50GB |   771.27GB(56.15) | Tesla V100-SXM2-32GB: cuda(7)
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |All Devs |    804650 | 100.00 | 17134.28GB |   17134.28GB(100.00)   |       0.00 B( 0.00)   | 11050.19GB |  6237.89GB(56.45) |
#~ +----------------------------------------------------------------------------------------------------------------------------+

#~ Full transfer matrix:
#~ dst\src          0          1          2          3          4          5          6          7          8          9 
   #~ 0        -          0.00 B   780.14GB   778.53GB   771.32GB   785.32GB   779.88GB   791.96GB   779.47GB   771.27GB
   #~ 1        0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 2     2103.10GB     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 3     2078.57GB     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 4     2147.57GB     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 5     2153.50GB     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B
   #~ 6     2162.77GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B
   #~ 7     2177.94GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B
   #~ 8     2173.10GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B
   #~ 9     2137.73GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -     

# CHO no mem limit
#~ mgonthier@gemini-1:~$ mpiexec -n 1 dplasma/builddir/tests/testing_spotrf -t 1920 -T 1920 -N $((1920*65)) -g 8 --nruns 11 --scheduler=AP
#~ [****] TIME(s)     15.06929 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   42996.690926 gflops - ENQ&PROG&DEST     21.51884 :   30109.865077 gflops - ENQ      5.48606 - DEST      0.96350
#~ [****] TIME(s)     10.09368 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   64191.595953 gflops - ENQ&PROG&DEST     17.00708 :   38097.632717 gflops - ENQ      5.96926 - DEST      0.94414
#~ [****] TIME(s)     10.11845 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   64034.426969 gflops - ENQ&PROG&DEST     16.59066 :   39053.870837 gflops - ENQ      5.52023 - DEST      0.95198
#~ [****] TIME(s)     10.24640 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   63234.865999 gflops - ENQ&PROG&DEST     16.85166 :   38449.009795 gflops - ENQ      5.66656 - DEST      0.93870
#~ [****] TIME(s)     10.38675 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   62380.365132 gflops - ENQ&PROG&DEST     16.83884 :   38478.266062 gflops - ENQ      5.51054 - DEST      0.94154
#~ [****] TIME(s)     10.23263 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   63319.906566 gflops - ENQ&PROG&DEST     16.84384 :   38466.843393 gflops - ENQ      5.65074 - DEST      0.96047
#~ [****] TIME(s)     10.31742 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   62799.568005 gflops - ENQ&PROG&DEST     17.01460 :   38080.798270 gflops - ENQ      5.75950 - DEST      0.93768
#~ [****] TIME(s)     10.30886 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   62851.721156 gflops - ENQ&PROG&DEST     16.71013 :   38774.647134 gflops - ENQ      5.44783 - DEST      0.95345
#~ [****] TIME(s)     10.18280 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   63629.776831 gflops - ENQ&PROG&DEST     16.60437 :   39021.627437 gflops - ENQ      5.47816 - DEST      0.94340
#~ [****] TIME(s)     10.22195 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   63386.116056 gflops - ENQ&PROG&DEST     16.70867 :   38778.040747 gflops - ENQ      5.54502 - DEST      0.94170
#~ [****] TIME(s)     10.25831 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  124800 :   63161.410685 gflops - ENQ&PROG&DEST     16.66556 :   38878.352179 gflops - ENQ      5.42169 - DEST      0.98556
#~ +----------------------------------------------------------------------------------------------------------------------------+
#~ |         |                    |                       Data In                              |         Data Out               |
#~ |Rank   0 |  # KERNEL |    %   |  Required  |   Transfered H2D(%)   |   Transfered D2D(%)   |  Required  |   Transfered(%)   |
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |  Dev  0 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | default
#~ |  Dev  1 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | recursive
#~ |  Dev  2 |     65510 |  12.43 |   318.53GB |     318.53GB(100.00)   |       0.00 B( 0.00)   |   899.64GB |    40.83GB( 4.54) | Tesla V100-SXM2-32GB: cuda(0)
#~ |  Dev  3 |     65577 |  12.44 |   319.08GB |     319.08GB(100.00)   |       0.00 B( 0.00)   |   900.56GB |    40.28GB( 4.47) | Tesla V100-SXM2-32GB: cuda(1)
#~ |  Dev  4 |     65756 |  12.48 |   318.69GB |     318.69GB(100.00)   |       0.00 B( 0.00)   |   903.02GB |    40.39GB( 4.47) | Tesla V100-SXM2-32GB: cuda(2)
#~ |  Dev  5 |     65888 |  12.50 |   318.62GB |     318.62GB(100.00)   |       0.00 B( 0.00)   |   904.83GB |    40.87GB( 4.52) | Tesla V100-SXM2-32GB: cuda(3)
#~ |  Dev  6 |     65313 |  12.39 |   317.18GB |     317.18GB(100.00)   |       0.00 B( 0.00)   |   896.94GB |    39.67GB( 4.42) | Tesla V100-SXM2-32GB: cuda(4)
#~ |  Dev  7 |     66680 |  12.65 |   319.63GB |     319.63GB(100.00)   |       0.00 B( 0.00)   |   915.71GB |    41.31GB( 4.51) | Tesla V100-SXM2-32GB: cuda(5)
#~ |  Dev  8 |     66839 |  12.68 |   319.28GB |     319.28GB(100.00)   |       0.00 B( 0.00)   |   917.89GB |    41.73GB( 4.55) | Tesla V100-SXM2-32GB: cuda(6)
#~ |  Dev  9 |     65392 |  12.41 |   320.28GB |     320.28GB(100.00)   |       0.00 B( 0.00)   |   898.02GB |    38.95GB( 4.34) | Tesla V100-SXM2-32GB: cuda(7)
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |All Devs |    526955 | 100.00 |  2551.29GB |    2551.29GB(100.00)   |       0.00 B( 0.00)   |  7236.63GB |   324.03GB( 4.48) |
#~ +----------------------------------------------------------------------------------------------------------------------------+

#~ Full transfer matrix:
#~ dst\src          0          1          2          3          4          5          6          7          8          9 
   #~ 0        -          0.00 B    40.83GB    40.28GB    40.39GB    40.87GB    39.67GB    41.31GB    41.73GB    38.95GB
   #~ 1        0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 2      318.53GB     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 3      319.08GB     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 4      318.69GB     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 5      318.62GB     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B
   #~ 6      317.18GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B
   #~ 7      319.63GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B
   #~ 8      319.28GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B
   #~ 9      320.28GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -     
#~ mgonthier@gemini-1:~$ mpiexec -n 1 dplasma/builddir/tests/testing_spotrf -t 1920 -T 1920 -N $((1920*70)) -g 8 --nruns 11 --scheduler=AP
#~ [****] TIME(s)     17.11300 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   47288.458252 gflops - ENQ&PROG&DEST     24.48463 :   33051.245338 gflops - ENQ      6.28098 - DEST      1.09065
#~ [****] TIME(s)     12.16393 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   66528.436765 gflops - ENQ&PROG&DEST     19.62056 :   41244.871919 gflops - ENQ      6.37032 - DEST      1.08631
#~ [****] TIME(s)     12.08324 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   66972.743244 gflops - ENQ&PROG&DEST     19.54432 :   41405.766635 gflops - ENQ      6.37295 - DEST      1.08813
#~ [****] TIME(s)     12.28273 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   65885.009107 gflops - ENQ&PROG&DEST     19.74545 :   40983.999418 gflops - ENQ      6.37389 - DEST      1.08883
#~ [****] TIME(s)     12.23740 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   66129.030601 gflops - ENQ&PROG&DEST     19.70968 :   41058.375447 gflops - ENQ      6.38407 - DEST      1.08821
#~ [****] TIME(s)     12.18126 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   66433.827948 gflops - ENQ&PROG&DEST     19.47440 :   41554.420777 gflops - ENQ      6.20134 - DEST      1.09181
#~ [****] TIME(s)     12.04864 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   67165.050397 gflops - ENQ&PROG&DEST     19.39304 :   41728.753773 gflops - ENQ      6.24615 - DEST      1.09825
#~ [****] TIME(s)     12.13664 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   66678.082217 gflops - ENQ&PROG&DEST     19.53228 :   41431.292559 gflops - ENQ      6.30885 - DEST      1.08680
#~ [****] TIME(s)     12.17025 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   66493.916802 gflops - ENQ&PROG&DEST     19.48122 :   41539.878787 gflops - ENQ      6.22231 - DEST      1.08866
#~ [****] TIME(s)     12.26874 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   65960.100481 gflops - ENQ&PROG&DEST     19.70608 :   41065.876519 gflops - ENQ      6.32122 - DEST      1.11612
#~ [****] TIME(s)     12.22827 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  134400 :   66178.425522 gflops - ENQ&PROG&DEST     20.19177 :   40078.096205 gflops - ENQ      6.86794 - DEST      1.09556
#~ +----------------------------------------------------------------------------------------------------------------------------+
#~ |         |                    |                       Data In                              |         Data Out               |
#~ |Rank   0 |  # KERNEL |    %   |  Required  |   Transfered H2D(%)   |   Transfered D2D(%)   |  Required  |   Transfered(%)   |
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |  Dev  0 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | default
#~ |  Dev  1 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | recursive
#~ |  Dev  2 |     81992 |  12.50 |   371.50GB |     371.50GB(100.00)   |       0.00 B( 0.00)   |  1125.99GB |    46.42GB( 4.12) | Tesla V100-SXM2-32GB: cuda(0)
#~ |  Dev  3 |     81600 |  12.44 |   368.76GB |     368.76GB(100.00)   |       0.00 B( 0.00)   |  1120.61GB |    47.05GB( 4.20) | Tesla V100-SXM2-32GB: cuda(1)
#~ |  Dev  4 |     82197 |  12.53 |   370.34GB |     370.34GB(100.00)   |       0.00 B( 0.00)   |  1128.80GB |    46.10GB( 4.08) | Tesla V100-SXM2-32GB: cuda(2)
#~ |  Dev  5 |     81787 |  12.47 |   370.99GB |     370.99GB(100.00)   |       0.00 B( 0.00)   |  1123.17GB |    46.62GB( 4.15) | Tesla V100-SXM2-32GB: cuda(3)
#~ |  Dev  6 |     81742 |  12.46 |   370.64GB |     370.64GB(100.00)   |       0.00 B( 0.00)   |  1122.56GB |    47.04GB( 4.19) | Tesla V100-SXM2-32GB: cuda(4)
#~ |  Dev  7 |     82133 |  12.52 |   368.99GB |     368.99GB(100.00)   |       0.00 B( 0.00)   |  1127.93GB |    47.17GB( 4.18) | Tesla V100-SXM2-32GB: cuda(5)
#~ |  Dev  8 |     82922 |  12.64 |   370.54GB |     370.54GB(100.00)   |       0.00 B( 0.00)   |  1138.76GB |    47.98GB( 4.21) | Tesla V100-SXM2-32GB: cuda(6)
#~ |  Dev  9 |     81667 |  12.45 |   370.64GB |     370.64GB(100.00)   |       0.00 B( 0.00)   |  1121.53GB |    47.01GB( 4.19) | Tesla V100-SXM2-32GB: cuda(7)
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |All Devs |    656040 | 100.00 |  2962.39GB |    2962.39GB(100.00)   |       0.00 B( 0.00)   |  9009.34GB |   375.39GB( 4.17) |
#~ +----------------------------------------------------------------------------------------------------------------------------+

#~ Full transfer matrix:
#~ dst\src          0          1          2          3          4          5          6          7          8          9 
   #~ 0        -          0.00 B    46.42GB    47.05GB    46.10GB    46.62GB    47.04GB    47.17GB    47.98GB    47.01GB
   #~ 1        0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 2      371.50GB     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 3      368.76GB     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 4      370.34GB     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 5      370.99GB     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B
   #~ 6      370.64GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B
   #~ 7      368.99GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B
   #~ 8      370.54GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B
   #~ 9      370.64GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -     
#~ mgonthier@gemini-1:~$ mpiexec -n 1 dplasma/builddir/tests/testing_spotrf -t 1920 -T 1920 -N $((1920*75)) -g 8 --nruns 11 --scheduler=AP
#~ [****] TIME(s)     19.94787 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   49896.981790 gflops - ENQ&PROG&DEST     28.51924 :   34900.589209 gflops - ENQ      7.32659 - DEST      1.24479
#~ [****] TIME(s)     14.50015 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   68643.301123 gflops - ENQ&PROG&DEST     23.25056 :   42809.229557 gflops - ENQ      7.50160 - DEST      1.24880
#~ [****] TIME(s)     14.64336 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   67971.985516 gflops - ENQ&PROG&DEST     23.16213 :   42972.659724 gflops - ENQ      7.26701 - DEST      1.25176
#~ [****] TIME(s)     14.50500 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   68620.381367 gflops - ENQ&PROG&DEST     23.10938 :   43070.752603 gflops - ENQ      7.35504 - DEST      1.24934
#~ [****] TIME(s)     14.57189 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   68305.376566 gflops - ENQ&PROG&DEST     23.08202 :   43121.808887 gflops - ENQ      7.25867 - DEST      1.25146
#~ [****] TIME(s)     14.82596 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   67134.854592 gflops - ENQ&PROG&DEST     24.00908 :   41456.745085 gflops - ENQ      7.92986 - DEST      1.25327
#~ [****] TIME(s)     14.79272 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   67285.671625 gflops - ENQ&PROG&DEST     23.65551 :   42076.383707 gflops - ENQ      7.61563 - DEST      1.24716
#~ [****] TIME(s)     14.62024 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   68079.500356 gflops - ENQ&PROG&DEST     23.14553 :   43003.476539 gflops - ENQ      7.27553 - DEST      1.24976
#~ [****] TIME(s)     14.63048 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   68031.812054 gflops - ENQ&PROG&DEST     23.02134 :   43235.471399 gflops - ENQ      7.13892 - DEST      1.25193
#~ [****] TIME(s)     14.73875 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   67532.073409 gflops - ENQ&PROG&DEST     23.25732 :   42796.778462 gflops - ENQ      7.26828 - DEST      1.25029
#~ [****] TIME(s)     14.60233 : spotrf	PxQxg=   1 1   8 NB= 1920 N=  144000 :   68162.989275 gflops - ENQ&PROG&DEST     23.06286 :   43157.635882 gflops - ENQ      7.20150 - DEST      1.25903
#~ +----------------------------------------------------------------------------------------------------------------------------+
#~ |         |                    |                       Data In                              |         Data Out               |
#~ |Rank   0 |  # KERNEL |    %   |  Required  |   Transfered H2D(%)   |   Transfered D2D(%)   |  Required  |   Transfered(%)   |
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |  Dev  0 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | default
#~ |  Dev  1 |         0 |   0.00 |     0.00 B |       0.00 B( -nan)   |       0.00 B( -nan)   |     0.00 B |     0.00 B( -nan) | recursive
#~ |  Dev  2 |    100042 |  12.43 |   426.78GB |     426.78GB(100.00)   |       0.00 B( 0.00)   |  1373.87GB |    52.78GB( 3.84) | Tesla V100-SXM2-32GB: cuda(0)
#~ |  Dev  3 |    100477 |  12.49 |   424.39GB |     424.39GB(100.00)   |       0.00 B( 0.00)   |  1379.84GB |    51.92GB( 3.76) | Tesla V100-SXM2-32GB: cuda(1)
#~ |  Dev  4 |    100021 |  12.43 |   424.69GB |     424.69GB(100.00)   |       0.00 B( 0.00)   |  1373.58GB |    54.44GB( 3.96) | Tesla V100-SXM2-32GB: cuda(2)
#~ |  Dev  5 |     99897 |  12.41 |   424.42GB |     424.42GB(100.00)   |       0.00 B( 0.00)   |  1371.88GB |    54.53GB( 3.98) | Tesla V100-SXM2-32GB: cuda(3)
#~ |  Dev  6 |    100491 |  12.49 |   424.65GB |     424.65GB(100.00)   |       0.00 B( 0.00)   |  1380.03GB |    54.15GB( 3.92) | Tesla V100-SXM2-32GB: cuda(4)
#~ |  Dev  7 |    101135 |  12.57 |   424.62GB |     424.62GB(100.00)   |       0.00 B( 0.00)   |  1388.88GB |    54.44GB( 3.92) | Tesla V100-SXM2-32GB: cuda(5)
#~ |  Dev  8 |    103257 |  12.83 |   425.32GB |     425.32GB(100.00)   |       0.00 B( 0.00)   |  1418.02GB |    55.56GB( 3.92) | Tesla V100-SXM2-32GB: cuda(6)
#~ |  Dev  9 |     99330 |  12.34 |   424.10GB |     424.10GB(100.00)   |       0.00 B( 0.00)   |  1364.09GB |    52.71GB( 3.86) | Tesla V100-SXM2-32GB: cuda(7)
#~ |---------|-----------|--------|------------|-----------------------|-----------------------|------------|-------------------|
#~ |All Devs |    804650 | 100.00 |  3398.96GB |    3398.96GB(100.00)   |       0.00 B( 0.00)   | 11050.19GB |   430.53GB( 3.90) |
#~ +----------------------------------------------------------------------------------------------------------------------------+

#~ Full transfer matrix:
#~ dst\src          0          1          2          3          4          5          6          7          8          9 
   #~ 0        -          0.00 B    52.78GB    51.92GB    54.44GB    54.53GB    54.15GB    54.44GB    55.56GB    52.71GB
   #~ 1        0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 2      426.78GB     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 3      424.39GB     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 4      424.69GB     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B     0.00 B
   #~ 5      424.42GB     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B     0.00 B
   #~ 6      424.65GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B     0.00 B
   #~ 7      424.62GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B     0.00 B
   #~ 8      425.32GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -          0.00 B
   #~ 9      424.10GB     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     0.00 B     -     
