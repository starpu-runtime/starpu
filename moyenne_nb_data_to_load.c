//~ gcc -o moyenne_nb_data_to_load moyenne_nb_data_to_load.c
//~ ./moyenne_nb_data_to_load NGPU Output_maxime/Data_to_load_GPU_0 Output_maxime/Data_to_load_GPU_1 Output_maxime/Data_to_load_GPU_2

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	int i = 0; int j = 0; int k = 0;
	int NGPU = atoi(argv[1]);
	int moyenne_nb_tache = 10;
	FILE* fichier_in = NULL;
	char chr1[10];
	char chr2[10];
	int count = 0;
	char c;
	//~ char* chr1;
	//~ char *chr2;
	
	for (i = 0; i < NGPU; i++)
	{	
		count = 0;
		fichier_in = fopen(argv[i + 2], "r");
		c = fgetc(fichier_in);
		while (c != EOF) {
			c = fgetc(fichier_in);
			if (c == '\n') { count++; }
		}
		
		//~ printf("Il y a %d lignes dans le fichier %d\n", count, i);
		
		int moyenne[count/moyenne_nb_tache]; for (j = 0; j < count/moyenne_nb_tache; j++) { moyenne[j] = 0; }
		rewind(fichier_in);
		if (fichier_in != NULL)
		{
			for (j = 0; j < count/moyenne_nb_tache; j++)
			{
				fscanf(fichier_in, "%s	%s", chr1, chr2);
				moyenne[j] = atoi(chr2);
				//~ printf("%d\n", atoi(chr2));
				for (k = 0; k < moyenne_nb_tache - 1; k++)
				{
					fscanf(fichier_in, "%s	%s", chr1, chr2);
					//~ printf("%d\n", atoi(chr2));
					moyenne[j] += atoi(chr2);
				}
				//~ printf("moyenne = %d\n", moyenne[j]);
				//~ moyenne[j] = moyenne[j]/moyenne_nb_tache;
			}
			fclose(fichier_in);
			fichier_in = fopen(argv[i + 2], "w");
			for (j = 0; j < count/moyenne_nb_tache; j++)
			{ 
				fprintf(fichier_in, "%d	%d\n", j + 1, moyenne[j]);
			}
		}
		else
		{
			printf("Impossible d'ouvrir un fichier de data to load dans moyenne_nb_data_to_load.c");
		}
		fclose(fichier_in);
	}
    return 0;
}
