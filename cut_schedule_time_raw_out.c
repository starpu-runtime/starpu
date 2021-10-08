/** La tête du fichier de sortie sera comme suit :
 * Global time split schedule time split schedule time all
 */
 
//~ Pour me lancer : 
//~ gcc -o cut_schedule_time_raw_out cut_schedule_time_raw_out.c
//~ ./cut_schedule_time_raw_out $NB_TAILLE_TESTE $NB_ALGO_TESTE $ECHELLE_X $START_X ${FICHIER_RAW_DT:0} Output_maxime/Data/${DOSSIER}/DT_${MODEL}_${GPU}_${NGPU}GPU.txt

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>

int main(int argc, char *argv[])
{
	if (argc != 7)
	{
		printf("Erreur : mauvais nombre d'argument dans cut_schedule_time_transfers.c.\n"); fflush(stdout);
		exit(0);
	}
	int NOMBRE_DE_TAILLES_DE_MATRICES = atoi(argv[1]);
	int NOMBRE_ALGO_TESTE = atoi(argv[2]);
	int ECHELLE_X = atoi(argv[3]);
	int START_X = atoi(argv[4]);
	long where_to_write = 0; int k = 0;
	long ligne[10] = {0};
	int index = 0;
	int first_loop = 0;
	int i = 0; int j = 0;
	char str1[20];
	int count = 0;
	char str2[10];
	char str3[10];
	char str4[10];
	char str5[10];
	char str6[10];
	char global[10];
	char c;
    FILE* fichier_in = NULL;
    FILE* fichier_out = NULL;
    fichier_in = fopen(argv[5], "r");
    fichier_out = fopen(argv[6], "w+");
    if (fichier_in != NULL)
    {
		//~ if (NGPU == 1)
		//~ {
			int count_before_global = 5;
			int count_before_schedule_split = 24;
			int count_before_schedule_all = 13;
			for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++)
			{
				fprintf(fichier_out,"%d",ECHELLE_X*(j+1)+START_X);
				for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++) 
				{
					if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j) 
					{
						for (k = 0; k < count_before_global; k++) 
						{
							fscanf(fichier_in, "%s", str1);
						}
						fprintf(fichier_out,"	%s", str1);
						for (k = 0; k < count_before_schedule_split; k++) 
						{
							fscanf(fichier_in, "%s", str1);
						}
						fprintf(fichier_out,"	%s", str1);
						for (k = 0; k < count_before_schedule_all; k++) 
						{
							fscanf(fichier_in, "%s", str1);
						}
						fprintf(fichier_out,"	%s", str1);
						
						/* Sortir de la ligne */
						for (k = 0; k < 2; k++) 
						{
							fscanf(fichier_in, "%s", str1);
						}
					}
					else 
					{
						for (k = 0; k < 44; k++) 
						{
							fscanf(fichier_in,"%s",str1);
						}
					}
				}
				fprintf(fichier_out,"\n"); 
				rewind(fichier_in);
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
