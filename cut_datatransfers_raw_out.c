/** La tête du fichier de sortie sera comme suit :
 * N NUMA->CUDA(ALGO1) NUMA->CUDA(ALGO2) CUDA->CUDA(ALGO1) CUDA->CUDA(ALGO2) NUMA->CUDA(ALGO1)+CUDA->CUDA(ALGO1) NUMA->CUDA(ALGO1)+CUDA->CUDA(ALGO2)
 */
 
//~ Pour me lancer : ./cut_datatransfers_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X $NGPU ${FICHIER_RAW_DT:0} Output_maxime/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt


#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>

int main(int argc, char *argv[])
{
	if (argc != 8)
	{
		printf("Erreur : mauvais nombre d'argument dans cut_data_transfers.c.\n");
		exit(0);
	}
	int NOMBRE_DE_TAILLES_DE_MATRICES = atoi(argv[1]);
	int NOMBRE_ALGO_TESTE = atoi(argv[2]);
	int ECHELLE_X = atoi(argv[3]);
	int START_X = atoi(argv[4]);
	int NGPU = atoi(argv[5]);
	long where_to_write = 0; int k = 0;
	long ligne[10] = {0};
	int index = 0;
	int first_loop = 0;
	int i = 0; int j = 0;
	char str1[10];
	int count = 0;
	char str2[10];
	char str3[10];
	char str4[10];
	char str5[10];
	char str6[10];
	char Datatransfers[10];
	char c;
    FILE* fichier_in = NULL;
    FILE* fichier_out = NULL;
    fichier_in = fopen(argv[6], "r");
    fichier_out = fopen(argv[7], "w+");
    if (fichier_in != NULL)
    {
		//~ if (NGPU == 1) 
		//~ {
			//~ count = 5;
			//~ for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++) 
			//~ {
				//~ fprintf(fichier_out,"%d",ECHELLE_X*(j+1)+START_X);
				//~ for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++) 
				//~ {
					//~ if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j) 
					//~ {
						//~ for (k = 0; k < count; k++) 
						//~ {
							//~ fscanf(fichier_in, "%s", str1);
						//~ }
						//~ fscanf(fichier_in, "%s", Datatransfers);
						//~ fprintf(fichier_out,"	%s", Datatransfers);
						//~ for (k = 0; k < 10; k++) {
							//~ fscanf(fichier_in,"%s", str1);
						//~ }
					//~ }
					//~ else 
					//~ {
						//~ for (k = 0; k < count + 11; k++) 
						//~ {
							//~ fscanf(fichier_in,"%s",str1);
						//~ }
					//~ }
				//~ }
				//~ fprintf(fichier_out,"\n"); 
				//~ rewind(fichier_in);
			//~ }
			//~ fclose(fichier_in);
			//~ fclose(fichier_out);
		//~ }
		//~ else
		//~ {
			int NCOMBINAISONS = NGPU*2+(NGPU-1)*NGPU;
			
			float NUMA_CUDA [NOMBRE_ALGO_TESTE][NOMBRE_DE_TAILLES_DE_MATRICES];
			float CUDA_CUDA [NOMBRE_ALGO_TESTE][NOMBRE_DE_TAILLES_DE_MATRICES];
			float TOTAL [NOMBRE_ALGO_TESTE][NOMBRE_DE_TAILLES_DE_MATRICES];
			int l = 0; int m = 0;
			for (i = 0; i < NOMBRE_ALGO_TESTE; i++)
			{
				for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++)
				{
					NUMA_CUDA[i][j] = 0;
					CUDA_CUDA[i][j] = 0;
					TOTAL[i][j] = 0;
				}
			}
			
			for (i = 0; i < NOMBRE_ALGO_TESTE; i++)
			{
				for (l = 0; l < NOMBRE_DE_TAILLES_DE_MATRICES; l++)
				{
					for (j = 0; j < NCOMBINAISONS; j++)
					{
						/* str1 et str 4 = acteurs, str6 = GB */
						fscanf(fichier_in, "%s	%s	%s	%s	%s	%s", str1, str2, str3, str4, str5, str6);
					
						if (strcmp(str1, "NUMA") == 0)
						{
							//~ printf("+= %f\n", atof(str6));
							NUMA_CUDA[i][l] += atof(str6);
						}
						else if (strcmp(str1, "CUDA") == 0 && strcmp(str4, "CUDA") == 0)
						{
							CUDA_CUDA[i][l] += atof(str6);
						}
						
						/* Skip to next line */
						for (k = 0; k < 10; k++) 
						{
							fscanf(fichier_in, "%s", str1);
						}
					}
				}
			}
			
			/* Calcul total */
			for (i = 0; i < NOMBRE_ALGO_TESTE; i++)
			{
				for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++)
				{
					TOTAL[i][j] += NUMA_CUDA[i][j] + CUDA_CUDA[i][j];
				}
			}
			//~ for (i = 0; i < NOMBRE_ALGO_TESTE; i++)
			//~ {
				//~ for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++){
					//~ printf("TOTAL %d = %f.\n", i, TOTAL[i][j]);
				//~ }
			//~ }
			
			/* Printing in the file */
			
			for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES; i++) 
			{
				fprintf(fichier_out, "%d", ECHELLE_X*(i+1)+START_X);
				for (j = 0; j < NOMBRE_ALGO_TESTE; j++) 
				{
					fprintf(fichier_out, "	%f", NUMA_CUDA[j][i]);
				}
				for (j = 0; j < NOMBRE_ALGO_TESTE; j++) 
				{
					fprintf(fichier_out, "	%f", CUDA_CUDA[j][i]);
				}
				for (j = 0; j < NOMBRE_ALGO_TESTE; j++) 
				{
					fprintf(fichier_out, "	%f", TOTAL[j][i]);
				}
				fprintf(fichier_out, "\n");
			}
			
			fclose(fichier_in);
			fclose(fichier_out);
		//~ }
    }
    else
    {
        printf("Impossible d'ouvrir le fichier raw.txt");
    }
    return 0;
}
