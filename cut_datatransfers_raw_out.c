#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	int NOMBRE_DE_TAILLES_DE_MATRICES = atoi(argv[1]);
	int NOMBRE_ALGO_TESTE = atoi(argv[2]);
	int ECHELLE_X = atoi(argv[3]);
	int START_X = atoi(argv[4]);
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
	char Datatransfers[10];
	char c;
    FILE* fichier_in = NULL;
    FILE* fichier_out = NULL;
    fichier_in = fopen(argv[5], "r");
    fichier_out = fopen(argv[6], "w+");
    if (fichier_in != NULL)
    {
		
		count = 5;
		
		
		for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++) {
			fprintf(fichier_out,"%d",ECHELLE_X*(j+1)+START_X);
			for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++) {
				if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j) {
					for (k = 0; k < count; k++) {
						fscanf(fichier_in,"%s",str1);
					}
					fscanf(fichier_in, "%s",Datatransfers);
					fprintf(fichier_out,"	%s",Datatransfers);
					for (k = 0; k < 10; k++) {
						fscanf(fichier_in,"%s",str1);
					}
				}
				else {
					for (k = 0; k < count + 11; k++) {
						fscanf(fichier_in,"%s",str1);
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
        printf("Impossible d'ouvrir le fichier raw.txt");
    }
    return 0;
}
