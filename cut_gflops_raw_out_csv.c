//~ gcc -o cut_gflops_raw_out_csv cut_gflops_raw_out_csv.c
//~ Pour cholesky la manière d'afficher les gflops n'est pas la même! Il affiche comme cela : 4800	112	328.9
//~ ./cut_gflops_raw_out_csv $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW} ${PATH_R}/R/Data/${DOSSIER}/GF_${MODEL}_${GPU}_${NGPU}GPU.csv
//~ ./cut_gflops_raw_out_csv 1 21 5 0 Output_maxime/GFlops_raw_out_1.txt test.csv
//~ Puis aller faire dans python3 Plot.py /home/gonthier/starpu/test.csv

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
	int i = 0; 
	
	//~ if (argc != 7)
	//~ {
		//~ fprintf(stderr, "Error: mauvais nb d'argument à ./cut_gflops_raw_out_csv\n");
		//~ exit(EXIT_FAILURE);
	//~ }
	int NOMBRE_DE_TAILLES_DE_MATRICES = atoi(argv[1]);
	int NOMBRE_ALGO_TESTE = atoi(argv[2]);
	int ECHELLE_X = atoi(argv[3]);
	int START_X = atoi(argv[4]);
	
	//~ int strsize = 0;
    //~ for (i=5; i<argc; i++) {
        //~ strsize += strlen(argv[i]);
        //~ if (argc > i+1)
            //~ strsize++;
    //~ }
    //~ printf("strsize: %d\n", strsize);
    //~ char *cmdstring;
    //~ cmdstring = malloc(strsize);
    //~ cmdstring[0] = '\0';
    //~ for (i=5; i<argc; i++) {
        //~ strcat(cmdstring, argv[i]);
        //~ if (argc > i+1)
            //~ strcat(cmdstring, " ");
    //~ }
    //~ printf("cmdstring: %s\n", cmdstring);
    //~ exit(0);
	
	//~ int NGPU = atoi(argv[7]);
	long where_to_write = 0; int k = 0;
	long ligne[10] = {0};
	int index = 0;
	int first_loop = 0;
	int j = 0;
	char str1[10];
	int count = 0;
	char GFlops[15];
	char deviance[15];
	char c;
	int conflits = 0;
    FILE* fichier_in = NULL;
    FILE* fichier_out = NULL;
    fichier_in = fopen(argv[5], "r");
    fichier_out = fopen(argv[6], "w+");
    int ecart_type = 1; /* A bool basically to separate gemm for cholesky for now */
    
    if (fichier_in != NULL)
    {
		c = fgetc(fichier_in);
		//~ while (c != '\n') { c = fgetc(fichier_in); }
		while (c != '\n') {
			c = fgetc(fichier_in);
			if (c == '	') { count++; }
		}
		if (count <= 2)
		{
			//We are in cholesky mode
			ecart_type = 0;
			
			if (count <= 1) // Cas conflits
			{
				conflits = 1;
			}
		}
		else
		{
			ecart_type = 1;
		}
		rewind(fichier_in);
		
		//~ if (NGPU == 1)
		//~ {
			//~ fprintf(fichier_out,"N,EAGER,DMDAR,DARTS,DARTS+LUF,DARTS+LUF+R,DARTS+LUF+TH2,DARTS+LUF+TH2+FM,DARTS+LUF+TH2+FM+SM,DARTS+LUF+TH2+FM+SM+NO1,DARTS+LUF+TH2+FM+SM+NO2,DARTS+LUF+TH2+FM+SM+NO2+R,DARTS+3D,DARTS+3D+LUF,DARTS+3D+LUF+R,DARTS+3D+LUF+TH2,DARTS+3D+LUF+TH2+FM,DARTS+3D+LUF+TH2+FM+NO1,DARTS+3D+LUF+TH2+FM+NO2,DARTS+3D+LUF+TH2+FM+SM,DARTS+3D+LUF+TH2+FM+SM+NO2,DARTS+3D+LUF+TH2+FM+SM+NO2+R\n");
			if (conflits == 0)
			{
				/* M2D mutexs */
				//~ fprintf(fichier_out,"N,DARTS+LUF,DARTS+LUF refined mutex\n");
				/* M3D mutexs */
				//~ fprintf(fichier_out,"N,DARTS+LUF+3D,DARTS+LUF+3D+TH2+FM+SM,DARTS+LUF+3D refined mutex,DARTS+LUF+3D+TH2+FM+SM refined mutex\n");
				/* CHO mutexs */
				fprintf(fichier_out,"N,DARTS+LUF,DARTS+LUF+3D,DARTS+LUF+3D+TH2+FM+SM,DARTS+LUF refined mutex,DARTS+LUF+3D refined mutex,DARTS+LUF+3D+TH2+FM+SM refined mutex\n");
			}
			else
			{
				/* M2D mutexs */
				//~ fprintf(fichier_out,"N,DARTS+LUF,DARTS+LUF refined mutex\n");
				/* M3D mutexs */
				//~ fprintf(fichier_out,"N,DARTS+LUF+3D conflits,DARTS+LUF+3D conflits critiques,DARTS+LUF+3D+TH2+FM+SM conflits,DARTS+LUF+3D+TH2+FM+SM conflits critiques\n");
				/* CHO mutexs */
				fprintf(fichier_out,"N,DARTS+LUF conflits,DARTS+LUF conflits critiques,DARTS+LUF+3D conflits,DARTS+LUF+3D conflits critiques,DARTS+LUF+3D+TH2+FM+SM conflits,DARTS+LUF+3D+TH2+FM+SM conflits critiques\n");
			}
		//~ }
		//~ else
		//~ {
			//~ fprintf(fichier_out,"EAGER,DMDAR,DARTS,DARTS+LUF,DARTS+LUF+R,DARTS+LUF+TH2,DARTS+LUF+TH2+FM,DARTS+LUF+TH2+FM+SM,DARTS+LUF+TH2+FM+SM+NO1,DARTS+LUF+TH2+FM+SM+NO2,DARTS+LUF+TH2+FM+SM+NO2+R,DARTS+3D,DARTS+3D+LUF,DARTS+3D+LUF+TH2,DARTS+3D+LUF+TH2+FM,DARTS+3D+LUF+TH2+FM+NO1,DARTS+3D+LUF+TH2+FM+NO2,DARTS+3D+LUF+TH2+FM+SM,DARTS+3D+LUF+TH2+FM+SM+NO2,DARTS+3D+LUF+TH2+FM+SM+NO2+R\n");
		//~ }
			
		for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++)
		{
			fprintf(fichier_out,"%d",ECHELLE_X*(j+1)+START_X);
			if (ecart_type == 0)
			{
				for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++)
				{
					if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j)
					{
						for (k = 0; k < count; k++)
						{
							fscanf(fichier_in,"%s",str1);
						}
						fscanf(fichier_in, "%s",GFlops);
						fprintf(fichier_out,",%s",GFlops);
					}
					else
					{
						for (k = 0; k < count + 1; k++)
						{
							fscanf(fichier_in,"%s",str1);
						}
					}
				}
			}
			else
			{
				for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++)
				{
					if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j)
					{
						for (k = 0; k < count - 1; k++)
						{
							fscanf(fichier_in,"%s", str1);
						}
						fscanf(fichier_in, "%s	%s",GFlops, deviance);
						fprintf(fichier_out,",%s",GFlops);
					}
					else
					{
						for (k = 0; k < count + 1; k++)
						{
							fscanf(fichier_in,"%s",str1);
						}
					}
				}
			}
			
			fprintf(fichier_out,"\n"); 
			rewind(fichier_in);
		}	
		fclose(fichier_in);
		fclose(fichier_out);
    }
    else
    {
        printf("Impossible d'ouvrir le fichier RAW.txt");
    }
    return 0;
}
