//~ gcc -o cut_gflops_raw_out_mean_deviation cut_gflops_raw_out_mean_deviation.c -lm
//~ ./cut_gflops_raw_out_mean_deviation 2 2 5 0 Output_maxime/GFlops_raw_out_3.txt Output_maxime/GFlops.txt 5

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
	int NOMBRE_DE_TAILLES_DE_MATRICES = atoi(argv[1]);
	int NOMBRE_ALGO_TESTE = atoi(argv[2]);
	int ECHELLE_X = atoi(argv[3]);
	int START_X = atoi(argv[4]);
	int NOMBRE_TEST_MEME_SOMMET = atoi(argv[7]);
	long where_to_write = 0; int k = 0;
	long ligne[10] = {0};
	int index = 0;
	int first_loop = 0;
	int i = 0; int j = 0; int l = 0; int m = 0;
	double moyenne = 0;
	double ecart_type = 0;
	double valeurs[NOMBRE_TEST_MEME_SOMMET];
	for (m = 0; m < NOMBRE_TEST_MEME_SOMMET; m++) { valeurs[m] = 0; }
	char str1[10];
	int count = 0;
	char str2[10];
	char str3[10];
	char str4[10];
	char GFlops[10];
	char c;
    FILE* fichier_in = NULL;
    FILE* fichier_out = NULL;
    fichier_in = fopen(argv[5], "r");
    fichier_out = fopen(argv[6], "w+");
    if (fichier_in != NULL)
    {
		c = fgetc(fichier_in);
		while (c != '\n') {
			c = fgetc(fichier_in);
			if (c == '	') { count++; }
		}
		rewind(fichier_in);
		
		for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++) {
			fprintf(fichier_out,"%d",ECHELLE_X*(j+1)+START_X);
			for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++) {
				
				//~ if (i%NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_TEST_MEME_SOMMET == j) {
				if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j) {
					//~ for (k = 0; k < count; k++) {
						//~ fscanf(fichier_in,"%s",str1);
					//~ }
					//~ fscanf(fichier_in, "%s",GFlops);
					//~ fprintf(fichier_out,"	%s",GFlops);
					for (l = 0; l < NOMBRE_TEST_MEME_SOMMET; l++) {
						for (k = 0; k < count; k++) {
							fscanf(fichier_in,"%s",str1);
						}
						fscanf(fichier_in, "%s",GFlops);
						//~ printf("Add %s\n", GFlops);
						moyenne += atoi(GFlops);
						valeurs[l] = atoi(GFlops);
					}
					moyenne = moyenne/NOMBRE_TEST_MEME_SOMMET;
					for (l = 0; l < NOMBRE_TEST_MEME_SOMMET; l++) {
						ecart_type += (valeurs[l] - moyenne)*(valeurs[l] - moyenne);
					}
					ecart_type = sqrt(ecart_type/NOMBRE_TEST_MEME_SOMMET);
					fprintf(fichier_out,"	%f", moyenne);
					//~ printf("Moyenne : %f\n", moyenne);
					//~ printf("Ecart_type : %f\n", ecart_type);
					fprintf(fichier_out,"	%f", ecart_type);
					moyenne = 0;
					ecart_type = 0;
					for (m = 0; m < NOMBRE_TEST_MEME_SOMMET; m++) { valeurs[m] = 0; }
				}
				else {
					for (m = 0; m < NOMBRE_TEST_MEME_SOMMET; m++) {
						for (k = 0; k < count; k++) {
							fscanf(fichier_in, "%s", str1);
						}
					}
					for (k = 0; k < count; k++) {
						fscanf(fichier_in, "%s", str1);
					}
					fscanf(fichier_in, "%s", str1);
					//~ for (k = 0; k < count + 1; k++) {
						//~ fscanf(fichier_in,"%s",str1);
					//~ }
				}
				
				//~ for (l = 0; l < NOMBRE_TEST_MEME_SOMMET; l++) {
					//~ for (k = 0; k < count; k++) {
						//~ fscanf(fichier_in,"%s",str1);
					//~ }
					//~ fscanf(fichier_in, "%s",GFlops);
					//~ moyenne += atoi(GFlops);
					//~ valeurs[l] = atoi(GFlops);
					//~ printf("%f\n", moyenne);
				//~ }
				//~ moyenne = moyenne/NOMBRE_TEST_MEME_SOMMET;
				//~ for (l = 0; l < NOMBRE_TEST_MEME_SOMMET; l++) {
					//~ ecart_type += (valeurs[l] - moyenne)*(valeurs[l] - moyenne);
				//~ }
				//~ ecart_type = sqrt(ecart_type/NOMBRE_TEST_MEME_SOMMET);
				//~ fprintf(fichier_out,"	%f", moyenne);
				//~ fprintf(fichier_out,"	%f", ecart_type);
				//~ moyenne = 0;
				//~ ecart_type = 0;
				//~ for (m = 0; m < NOMBRE_TEST_MEME_SOMMET; m++) { valeurs[m] = 0; }
			}
			//~ printf("Retour à la ligne\n");
			fprintf(fichier_out, "\n");
			rewind(fichier_in);
		}
		
		//~ for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++) {
			//~ fprintf(fichier_out,"%d",ECHELLE_X*(j+1)+START_X);
			//~ for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++) {
				//~ if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j) {
					//~ for (k = 0; k < count; k++) {
						//~ fscanf(fichier_in,"%s",str1);
					//~ }
					//~ fscanf(fichier_in, "%s",GFlops);
					//~ fprintf(fichier_out,"	%s",GFlops);
				//~ }
				//~ else {
					//~ for (k = 0; k < count + 1; k++) {
						//~ fscanf(fichier_in,"%s",str1);
					//~ }
				//~ }
			//~ }
			//~ fprintf(fichier_out,"\n"); 
			//~ rewind(fichier_in);
		//~ }	
		fclose(fichier_in);
		fclose(fichier_out);
    }
    else
    {
        printf("Impossible d'ouvrir le fichier RAW.txt");
    }
    return 0;
}
