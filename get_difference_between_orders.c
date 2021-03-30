#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	char str1[15];
	int x = 0; int y = 0;
	char str2[15];
    FILE *fichier_hfp = fopen(argv[1], "r");
    FILE *fichier_effectif = fopen(argv[2], "r");
    FILE *fichier_out = fopen(argv[3], "w");
    if (fichier_hfp != NULL && fichier_effectif != NULL)
    {
		/* Print beggining */
		fprintf(fichier_out, "	x	y\n");
		while (fscanf(fichier_hfp, "%s", str1) == 1) //expect 1 successful conversion
		{
		  //process buffer
		  fprintf(fichier_out, "%s	%d", str1, x);
		  x++;
		  rewind(fichier_effectif);
		  fscanf(fichier_effectif, "%s", str2);
		  y = 0;
		  while (strcmp(str1, str2) != 0)
		  {
			 fscanf(fichier_effectif, "%s", str2);
			 if (feof(fichier_effectif)) { y = 0; break; }
			 y++;
		  }
		 fprintf(fichier_out, "	%d\n", y); 
		}
		if (feof(fichier_hfp)) 
		{
		  //hit end of file
		  fclose(fichier_hfp);
		  fclose(fichier_effectif);
		  fclose(fichier_out);
		  return 0;
		}
		else
		{
		  //some other error interrupted the read
		  printf("Error while reading\n"); exit(0);
		}
    }
    else
    {
        printf("Impossible d'ouvrir au moins 1 fichier d'ordre\n"); 
        exit(0);
    }
    return 0;
}

