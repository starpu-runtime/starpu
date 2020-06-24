#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	int NOMBRE_DE_TAILLES_DE_MATRICES = atoi(argv[1]);
	int NOMBRE_ALGO_TESTE = atoi(argv[2]);
	
	long where_to_write = 0; int k =0;
	long ligne[10] = {0};
	int index = 0;
	int first_loop = 0;
	int i = 0; int j = 0;
	char str1[10];
	char str2[10];
	char str3[10];
	char str4[10];
	char GFlops[10];
    FILE* fichier_in = NULL;
    FILE* fichier_out = NULL;
    fichier_in = fopen("GFlops_raw_out.txt", "r");
    fichier_out = fopen("GFlops_data_out.txt", "w+");
    if (fichier_in != NULL)
    {
		
		
		for (j = 0; j < NOMBRE_DE_TAILLES_DE_MATRICES; j++) {
			k = j;
			while (k != 0) { fscanf(fichier_in, "%s	%s %s %s %s",str1,str2,str3,str4,GFlops); k--; }
			
			fscanf(fichier_in, "%s	%s %s %s %s",str1,str2,str3,str4,GFlops);
			fprintf(fichier_out,"%s",str1);
			rewind(fichier_in);	
			for (i = 0; i < NOMBRE_DE_TAILLES_DE_MATRICES*NOMBRE_ALGO_TESTE; i++) {
				if (i%NOMBRE_DE_TAILLES_DE_MATRICES == j) {
					fscanf(fichier_in, "%s	%s %s %s %s",str1,str2,str3,str4,GFlops);
					fprintf(fichier_out,"	%s",GFlops);
				}
				else {
					fscanf(fichier_in, "%s	%s %s %s %s",str1,str2,str3,str4,GFlops);
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
        printf("Impossible d'ouvrir le fichier GFlops_raw_out.txt");
    }
    return 0;
}
