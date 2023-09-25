# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

# Use the output of an execution using dmdar or darts with --enable-darts-stats --enable-darts-verbose enabled to build a visualization in svg, saved in the starpu/ repository

import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import matplotlib.patheffects as PathEffects
from color_darts import gradiant_color
from color_darts import gradiant_multiple_color

# Give a color to a gpu
def gpu_color(gpu):
	r = 0
	g = 0
	b = 0
	if (gpu == 0):
		r = 1
	elif (gpu == 1):
		g = 1
	elif (gpu == 2):
		r = 73/255
		g = 116/255
		b = 1
	elif (gpu == 3):
		r = 1
		b = 1
	elif (gpu == 4):
		g = 1
		b = 1
	elif (gpu == 5):
		r = 1
		b = 1
	elif (gpu == 6):
		r = 1
		g = 0.5
		b = 0.5
	elif (gpu == 7):
		r = 0.5
		g = 1
		b = 0.5
	else:
		r = 1/gpu
		g = 1/gpu
		b = 1/gpu

	return (r, g, b)

# Print a line on the side of the matrix depending on wheter or not a data load was necessary
def lignes_sur_le_cote(case, axe, epaisseur, gpu, line_type):
	epaisseur = epaisseur/2

	if (gpu == 0):
		decalage = 3
	elif (gpu == 1):
		decalage = 4
	elif (gpu == 2):
		decalage = 5
	elif (gpu == 3):
		decalage = 6
	elif (gpu == 4):
		decalage = 7
	elif (gpu == 5):
		decalage = 8
	elif (gpu == 6):
		decalage = 9
	elif (gpu == 7):
		decalage = 10
	else:
		decalage = 3 + gpu

	if (axe == "x"):
		trans = ax.get_xaxis_transform()
		ax.annotate('', xy = (case, -0.1), xycoords=trans, ha = "center", va = "top")
		return ax.plot([case - 0.49, case + 0.49],[-.02 * decalage, -.02 * decalage], color = gpu_color(gpu), linewidth = epaisseur, transform = trans, clip_on = False, linestyle = line_type)
	elif (axe == "y"):
		trans = ax.get_yaxis_transform()
		return ax.plot([-.02 * decalage, -.02 * decalage], [case - 0.49, case + 0.49], color = gpu_color(gpu), linewidth = epaisseur, transform = trans, clip_on = False, linestyle = line_type)

def custom_lignes_sur_le_cote(case, axe, epaisseur, gpu, line_type, decalage):
	if (axe == "x"):
		trans = ax.get_xaxis_transform()
		ax.annotate('', xy = (case, -0.1), xycoords=trans, ha = "center", va = "top")
		return ax.plot([case - 0.45, case + 0.45],[-.02 * decalage, -.02 * decalage], color = gpu_color(gpu), linewidth = epaisseur, transform = trans, clip_on = False, linestyle = line_type)
	elif (axe == "y"):
		trans = ax.get_yaxis_transform()
		return ax.plot([-.02 * decalage, -.02 * decalage], [case - 0.45, case + 0.45], color = gpu_color(gpu), linewidth = epaisseur, transform = trans, clip_on = False, linestyle = line_type)

def data_sur_le_cote_3D(x, y, axe, line_width, gpu, line_type, i, j, z):
	if (gpu == 0):
		decalage = 0
	elif (gpu == 1):
		decalage = 0.2
	elif (gpu == 2):
		decalage = 0.4
	elif (gpu == 3):
		decalage = 0.6
	elif (gpu == 4):
		decalage = 0.8
	elif (gpu == 5):
		decalage = 1
	elif (gpu == 6):
		decalage = 1.2
	elif (gpu == 7):
		decalage = 1.4
	else:
		decalage = 1.6 + 0.2*(gpu%8)

	if (axe == "x"):
		return ax[i, j].plot([x - 0.49, x + 0.49], [z + decalage, z + decalage], color = gpu_color(gpu), linewidth = line_width, linestyle = line_type)
	elif (axe == "y"):
		return ax[i, j].plot([z + decalage, z + decalage], [y - 0.49, y + 0.49], color = gpu_color(gpu), linewidth = line_width, linestyle = line_type)
	elif (axe == "z"):
		return ax[i, j].plot([x - 0.49, x + 0.49 - decalage], [y - 0.49 + decalage, y + 0.49], color = gpu_color(gpu), linewidth = line_width, linestyle = line_type)
	else:
		sys.exit("Axe must be x, y or z in def data_sur_le_cote_3D(x, y, axe, epaisseur, gpu, line_type, i, j)")

# Printing in a separate white matrix the data loaded with a heat map
def data_sur_le_cote_3D_heat_map(x, y, axe, heat, gpu, i, j, z):
	r = 0
	g = 0
	b = 0
	if (heat == 1):
		g = 1
	elif (heat == 2):
		r = 1
		g = 127/255
	elif (heat == 3):
		r = 1
	elif (heat == 4):
		r = 0.5
		b = 0.5
	else:
		sys.exit("Heat must be 1, 2, 3 or 4 in data_sur_le_cote_3D_heat_map")

	if (axe == "z"):
		m[i, j][x, y, :] = (r, g, b)
	else:
		sys.exit("Axe must be z in data_sur_le_cote_3D_heat_map")

def tache_load_balanced(x, y, gpu):
	ax.plot([x - 0.49, x + 0.49],[y - 0.49, y - 0.49], color = gpu_color(gpu), clip_on = False)
	ax.plot([x - 0.49, x + 0.49],[y + 0.49, y + 0.49], color = gpu_color(gpu), clip_on = False)
	ax.plot([x - 0.49, x - 0.49],[y - 0.49, y + 0.49], color = gpu_color(gpu), clip_on = False)
	ax.plot([x + 0.49, x + 0.49],[y - 0.49, y + 0.49], color = gpu_color(gpu), clip_on = False)
	return

def tache_load_balanced_3D(x, y, gpu, i, j):
	ax[i, j].plot([x - 0.49, x + 0.49],[y - 0.49, y - 0.49], color = gpu_color(gpu), clip_on = False)
	ax[i, j].plot([x - 0.49, x + 0.49],[y + 0.49, y + 0.49], color = gpu_color(gpu), clip_on = False)
	ax[i, j].plot([x - 0.49, x - 0.49],[y - 0.49, y + 0.49], color = gpu_color(gpu), clip_on = False)
	ax[i, j].plot([x + 0.49, x + 0.49],[y - 0.49, y + 0.49], color = gpu_color(gpu), clip_on = False)
	return

def line_to_load(x, y):
    return plt.plot([x - 0.44, x + 0.44], [y, y], color='#FFBB96', lw = 4.2, zorder = 5)

def column_to_load(x, y):
	return plt.plot([x, x], [y - 0.44, y + 0.44], '#FFBB96', lw = 4.2, zorder = 5)

def line_to_load_prefetch(x, y):
    # ~ return plt.plot([x - 0.48, x + 0.48], [y, y], 'white', lw = 4.2, zorder = 5, linestyle='dotted')
    return plt.plot([x - 0.48, x + 0.48], [y, y], '#FFBB96', lw = 4.2, zorder = 5, linestyle='dotted')

def column_to_load_prefetch(x, y):
	return plt.plot([x, x], [y - 0.48, y + 0.48], '#FFBB96', lw = 4.2, zorder = 5, linestyle='dotted')

def line_to_load_3D(x, y, i, j):
    return ax[i, j].plot([x - 0.44, x + 0.44], [y, y], '#FFBB96', lw = 4.2, zorder = 5)

def column_to_load_3D(x, y, i, j):
	return ax[i, j].plot([x, x], [y - 0.44, y + 0.44], '#FFBB96', lw = 4.2, zorder = 5)

def Z_to_load_3D(x, y, i, j):
	return ax[i, j].plot([x - 0.44, x + 0.44], [y - 0.44, y + 0.44], '#FFBB96', lw = 4.2, zorder = 5)

def line_to_load_3D_prefetch(x, y, i, j):
    return ax[i, j].plot([x - 0.48, x + 0.48], [y, y], '#FFBB96', lw = 4.2, zorder = 5, linestyle='dotted')

def column_to_load_3D_prefetch(x, y, i, j):
	return ax[i, j].plot([x, x], [y - 0.48, y + 0.48], '#FFBB96', lw = 4.2, zorder = 5, linestyle='dotted')

def Z_to_load_3D_prefetch(x, y, i, j):
	return ax[i, j].plot([x - 0.48, x + 0.48], [y - 0.48, y + 0.48], '#FFBB96', lw = 4.2, zorder = 5, linestyle='dotted')

def sous_paquets(x, y, sous_paquet):
		return ax.annotate(sous_paquet, xy = (x, y), ha = "center")

def separation_sous_paquets(x, y, x_bis, y_bis):
	if (x == 0 and x_bis == N - 1):
		return ax.plot([x - 0.49, x - 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (x == N - 1 and x_bis == 0):
		return ax.plot([x + 0.49, x + 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y == 0 and y_bis == N - 1):
		return ax.plot([x - 0.49, x + 0.49],[y - 0.49, y - 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y == N - 1 and y_bis == 0):
		return ax.plot([x - 0.49, x + 0.49],[y + 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (x < x_bis):
		return ax.plot([x + 0.49, x + 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (x > x_bis):
		return ax.plot([x - 0.49, x - 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y < y_bis):
		return ax.plot([x - 0.49, x + 0.49],[y + 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y > y_bis):
		return ax.plot([x - 0.49, x + 0.49],[y - 0.49, y - 0.49], color = "black", linewidth = 4, clip_on = False)

def separation_sous_paquets_3D(x, y, x_bis, y_bis, i, j):
	if (x == 0 and x_bis == N - 1):
		return ax[i, j].plot([x - 0.49, x - 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (x == N - 1 and x_bis == 0):
		return ax[i, j].plot([x + 0.49, x + 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y == 0 and y_bis == N - 1):
		return ax[i, j].plot([x - 0.49, x + 0.49],[y - 0.49, y - 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y == N - 1 and y_bis == 0):
		return ax[i, j].plot([x - 0.49, x + 0.49],[y + 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (x < x_bis):
		return ax[i, j].plot([x + 0.49, x + 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (x > x_bis):
		return ax[i, j].plot([x - 0.49, x - 0.49],[y - 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y < y_bis):
		return ax[i, j].plot([x - 0.49, x + 0.49],[y + 0.49, y + 0.49], color = "black", linewidth = 4, clip_on = False)
	elif (y > y_bis):
		return ax[i, j].plot([x - 0.49, x + 0.49],[y - 0.49, y - 0.49], color = "black", linewidth = 4, clip_on = False)

def get_i_j(z):
	i = 0
	j = 0
	if(z == 1):
		j = 1
	elif(z == 2):
		i = 1
	elif(z == 3):
		i = 1
		j = 1
	elif(z > 3):
		print("get_i_j pas défini au dela de Z = 4")
	return i, j

def get_i_j_cholesky(iterationk, figure_par_ligne):
	i = int(iterationk) // int(figure_par_ligne)
	j = int(iterationk) % int(figure_par_ligne)
	return i, j

N = int(sys.argv[1])
ORDO = sys.argv[2]
NGPU = int(sys.argv[3])
APPLI = sys.argv[4]
PATH = sys.argv[8]

# Opening input files and initializing tabulars
file_coord = open(PATH + "/Data_coordinates_order_last_SCHEDULER.txt", "r")
file_data = open(PATH + "/Data_to_load_SCHEDULER.txt", "r")
file_coord_prefetch = open(PATH + "/Data_to_load_prefetch_SCHEDULER.txt", "r")
nb_tache_par_gpu = [0 for i in range(NGPU)]
order = [0 for i in range(NGPU)]
data_to_load = [[0] * N for i in range(N)]

# Options you can switch to true or false depending on what you want to plot:

lignes_dans_les_cases = True # If you don't want the lines in the tiles un-comment false
# ~ lignes_dans_les_cases = False

# ~ lignes_sur_les_cote = True # If you don't want the lines of the side un-comment false
lignes_sur_les_cote = False

# ~ numerotation_des_cases = True # If you don't want a numerotation in the tiles un-comment false
numerotation_des_cases = False

# ~ numerotation_des_cases_partielles = True # If you don't a partial numerotation in the tiles un-comment false
numerotation_des_cases_partielles = False

numerotation_axes_complete = True # If you don't want a numerotation of the axis un-comment false
# ~ numerotation_axes_complete = False

# ~ z_dans_les_cases = True # If you don't want to show the line in the tiles for the z dimension in the case of GEMM un-comment false
z_dans_les_cases = False


if (ORDO == "HFP"):
	sous_paquets_and_task_stealing = False # pour afficher ou non les sous paquets et le stealing avec HFP
else:
	sous_paquets_and_task_stealing = False

plt.tick_params(labelsize=50) # Pour la taille des chiffres sur les axes x et y des matrices

if (APPLI == "Matrice_ligne" or APPLI == "Matrice3D" or APPLI == "MatriceZ4" or APPLI == "MatriceZN"):

	NDIMENSIONS = int(sys.argv[5])

	epaisseur_lignes_sur_le_cote = [[1] * N for i in range(NGPU)]
	epaisseur_colonnes_sur_le_cote = [[1] * N for i in range(NGPU)]
	epaisseur_lignes_sur_le_cote_prefetch = [[1] * N for i in range(NGPU)]
	epaisseur_colonnes_sur_le_cote_prefetch = [[1] * N for i in range(NGPU)]

	if (NDIMENSIONS == 1):

		size_numero_dans_les_cases = 19

		ORDRE_GLOBAL = 1

		# ~ ax = plt.gca();
		fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

		# Grid
		if (numerotation_axes_complete == True):
			ax.set_xticks(np.arange(0, N, 1)) # numérotations des axes X et Y
			ax.set_yticks(np.arange(0, N, 1))
		else:
			ax.set_xticks(np.arange(0, N, 5)) # numérotations des axes X et Y
			ax.set_yticks(np.arange(0, N, 5))
		ax.set_xticks(np.arange(0.5, N, 1), minor=True)
		ax.set_yticks(np.arange(0.5, N, 1), minor=True)
		ax.grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)

		# Filling a matrix with 0
		m = np.zeros((N, N, 3))

		for line in file_coord:
			fields = line.split()
			nb_tache_par_gpu[int(fields[2])] = nb_tache_par_gpu[int(fields[2])] + 1

		file_coord.seek(0)

		if (ORDRE_GLOBAL == 1):
			for i in range (0, NGPU):
				nb_tache_par_gpu[i] = N*N

		# Coloring tiles in function of their numbering
		for line in file_coord:
			fields = line.split()
			# X Y GPU
			if (ORDRE_GLOBAL == 0):
				index_ordre = int(fields[2])
			else:
				index_ordre = 0
			m[int(fields[1]), int(fields[0]), :] = gradiant_color(int(fields[2]), order[index_ordre], nb_tache_par_gpu[int(fields[2])])

			if (numerotation_des_cases == True):
				ax.text(int(fields[0]), int(fields[1]), order[index_ordre], va="center", weight="bold", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif (numerotation_des_cases_partielles == True and order[index_ordre]%10 == 0):
				ax.text(int(fields[0]), int(fields[1]), order[index_ordre], weight="bold", va="center", ha="center", color = "white", size = size_numero_dans_les_cases, zorder=10)
			order[index_ordre] = order[index_ordre] + 1

		if (lignes_dans_les_cases == True or lignes_sur_les_cote == True):
			for line in file_data:
				fields = line.split()

				if (int(fields[2]) != 0):
					column_to_load(int(fields[0]), int(fields[1]))

					if (lignes_sur_les_cote == True):
						lignes_sur_le_cote(int(fields[1]), "y", epaisseur_colonnes_sur_le_cote[int(fields[4])][int(fields[1])], int(fields[4]), "solid")
						epaisseur_colonnes_sur_le_cote[int(fields[4])][int(fields[1])] += 4

					data_to_load[int(fields[0])][int(fields[1])] = 1

				if (int(fields[3]) != 0):
					line_to_load(int(fields[0]), int(fields[1]))

					if (lignes_sur_les_cote == True):
						lignes_sur_le_cote(int(fields[0]), "x", epaisseur_lignes_sur_le_cote[int(fields[4])][int(fields[0])], int(fields[4]), "solid")
						epaisseur_lignes_sur_le_cote[int(fields[4])][int(fields[0])] += 4

					if (data_to_load[int(fields[0])][int(fields[1])] != 0):
						data_to_load[int(fields[0])][int(fields[1])] = 3
					else:
						data_to_load[int(fields[0])][int(fields[1])] = 2

			for line in file_coord_prefetch:
				fields = line.split()
				if (int(fields[2]) != 0 and data_to_load[int(fields[0])][int(fields[1])] != 1 and data_to_load[int(fields[0])][int(fields[1])] != 3):
					column_to_load_prefetch(int(fields[0]), int(fields[1]))

					if (lignes_sur_les_cote == True):
						if (epaisseur_colonnes_sur_le_cote[int(fields[4])][int(fields[1])] == 1):
							lignes_sur_le_cote(int(fields[1]), "y", epaisseur_colonnes_sur_le_cote_prefetch[int(fields[4])][int(fields[1])], int(fields[4]), "dashed")
							epaisseur_colonnes_sur_le_cote_prefetch[int(fields[4])][int(fields[1])] += 4
						else:
							lignes_sur_le_cote(int(fields[1]), "y", epaisseur_colonnes_sur_le_cote[int(fields[4])][int(fields[1])], int(fields[4]), "solid")
							epaisseur_colonnes_sur_le_cote[int(fields[4])][int(fields[1])] += 4

				if (int(fields[3]) != 0 and data_to_load[int(fields[0])][int(fields[1])] != 2 and data_to_load[int(fields[0])][int(fields[1])] != 3):
					line_to_load_prefetch(int(fields[0]), int(fields[1]))

					if (lignes_sur_les_cote == True):
						if (epaisseur_lignes_sur_le_cote[int(fields[4])][int(fields[0])] == 1):
							lignes_sur_le_cote(int(fields[0]), "x", epaisseur_lignes_sur_le_cote_prefetch[int(fields[4])][int(fields[0])], int(fields[4]), "dashed")
							epaisseur_lignes_sur_le_cote_prefetch[int(fields[4])][int(fields[0])] += 4
						else:
							lignes_sur_le_cote(int(fields[0]), "x", epaisseur_lignes_sur_le_cote[int(fields[4])][int(fields[0])], int(fields[4]), "solid")
							epaisseur_lignes_sur_le_cote[int(fields[4])][int(fields[0])] += 4

		if (sous_paquets_and_task_stealing == True):
			# Load balance steal
			file_load_balance = open(PATH + "/Data_stolen_load_balance.txt", "r")

			for line in file_load_balance:
				fields = line.split()
				tache_load_balanced(int(fields[0]), int(fields[1]), int(fields[2]))

			file_load_balance.close()

			file_last_package = open(PATH + "/last_package_split.txt", "r")

			hierarchie_paquets = [[0] * N for i in range(N)]
			for line in file_last_package:
				fields = line.split()

				hierarchie_paquets[int(fields[0])][int(fields[1])] = int(fields[3])

			for i in range(N):
				for j in range(N):
					if (i != 0):
						if (hierarchie_paquets[i][j] != hierarchie_paquets[i - 1][j]):
							separation_sous_paquets(i, j, i - 1, j)
					else:
						if (hierarchie_paquets[i][j] != hierarchie_paquets[N - 1][j]):
							separation_sous_paquets(i, j, N - 1, j)
					if (i != N - 1):
						if (hierarchie_paquets[i][j] != hierarchie_paquets[i + 1][j]):
							separation_sous_paquets(i, j, i + 1, j)
					else:
						if (hierarchie_paquets[i][j] != hierarchie_paquets[0][j]):
							separation_sous_paquets(i, j, 0, j)
					if (j != 0):
						if (hierarchie_paquets[i][j] != hierarchie_paquets[i][j - 1]):
							separation_sous_paquets(i, j, i, j - 1)
					else:
						if (hierarchie_paquets[i][j] != hierarchie_paquets[i][N - 1]):
							separation_sous_paquets(i, j, i, N - 1)
					if (j != N - 1):
						if (hierarchie_paquets[i][j] != hierarchie_paquets[i][j + 1]):
							separation_sous_paquets(i, j, i, j + 1)
					else:
						if (hierarchie_paquets[i][j] != hierarchie_paquets[i][0]):
							separation_sous_paquets(i, j, i, 0)

			file_last_package.close()

		plt.imshow(m)
	# End of 2D matrix

	# Start of 3D matrix, in 3D in the files you have x y z so the GPU is always one fields later
	else:
		NROW = 2
		NCOL = 2

		fig, ax = plt.subplots(nrows = NROW, ncols = NCOL)

		size_numero_dans_les_cases = 8

		for i in range(NROW):
			for j in range(NCOL):
				if (numerotation_axes_complete == True):
					ax[i, j].set_xticks(np.arange(0, N, 1)) # numérotations des axes X et Y
					ax[i, j].set_yticks(np.arange(0, N, 1))
				else:
					ax[i, j].set_xticks(np.arange(0, N, 5)) # numérotations des axes X et Y
					ax[i, j].set_yticks(np.arange(0, N, 5))
				ax[i, j].set_xticks(np.arange(0.5, N, 1), minor=True)
				ax[i, j].set_yticks(np.arange(0.5, N, 1), minor=True)
				ax[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax[i, j].tick_params(labelsize=13) # Plus petit pour 3D car dans FGCS c'est agrandiss
		if (NDIMENSIONS == 4):
			i_x_on_side = 2
			j_x_on_side = 0
			i_y_on_side = 2
			j_y_on_side = 1
			i_z_on_side = 2
			j_z_on_side = 2

		# Filling a matrix with 0. m is for colors.
		m = {}
		already_fetched_x = {}
		already_fetched_y = {}
		already_fetched_z = {}
		hierarchie_paquets = {}
		epaisseur_x = {}
		epaisseur_y = {}
		epaisseur_z = {}

		for i in range(NROW):
			for j in range(NCOL):
				already_fetched_x[i, j] = np.zeros((N, N, 1))
				already_fetched_y[i, j] = np.zeros((N, N, 1))
				already_fetched_z[i, j] = np.zeros((N, N, 1))
				hierarchie_paquets[i, j] = np.zeros((N, N, 1))

		for i in range(NROW):
			for j in range(NCOL):
				m[i, j] = np.ones((N, N, 3))

		for line in file_coord:
			fields = line.split()
			nb_tache_par_gpu[int(fields[3])] = nb_tache_par_gpu[int(fields[3])] + 1

		file_coord.seek(0)
		for line in file_coord:
			fields = line.split()
			i, j = get_i_j(int(fields[2]))
			print(i, j, int(fields[1]), int(fields[0]))
			m[i, j][int(fields[1]), int(fields[0]), :] = gradiant_color(int(fields[3]), order[int(fields[3])], nb_tache_par_gpu[int(fields[3])])

			index_ordre = int(fields[3])

			if (numerotation_des_cases == True):
				ax[i,j].text(int(fields[0]), int(fields[1]), order[index_ordre], va="center", weight="bold", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif (numerotation_des_cases_partielles == True and order[index_ordre]%10 == 0):
				ax[i,j].text(int(fields[0]), int(fields[1]), order[index_ordre], va="center", weight="bold", ha="center", color = "white", size = size_numero_dans_les_cases, zorder=10)

			order[int(fields[3])] = order[int(fields[3])] + 1

			for line in file_data:
				fields = line.split()
				i, j = get_i_j(int(fields[2]))

				if (int(fields[3]) != 0):
					column_to_load_3D(int(fields[0]), int(fields[1]), i, j)

					if (lignes_sur_les_cote == True):
						data_sur_le_cote_3D(int(fields[0]), int(fields[1]), "y", epaisseur_y[i_y_on_side, j_y_on_side][int(fields[2]) + (int(fields[6])*4), int(fields[1])], int(fields[6]), "solid", i_x_on_side, j_x_on_side, int(fields[2]))
						epaisseur_y[i_y_on_side, j_y_on_side][int(fields[2]) + (int(fields[6])*4), int(fields[1])] += 4

					already_fetched_x[i, j][int(fields[0]), int(fields[1])] = 1

				if (int(fields[4]) != 0):
					line_to_load_3D(int(fields[0]), int(fields[1]), i, j)

					if (lignes_sur_les_cote == True):
						data_sur_le_cote_3D(int(fields[0]), int(fields[1]), "x", epaisseur_x[i_x_on_side, j_x_on_side][int(fields[0]), int(fields[2]) + (int(fields[6])*4)], int(fields[6]), "solid", i_y_on_side, j_y_on_side, int(fields[2]))
						epaisseur_x[i_x_on_side, j_x_on_side][int(fields[0]), int(fields[2]) + (int(fields[6])*4)] += 4

					already_fetched_y[i, j][int(fields[0]), int(fields[1])] = 1

				# The "diagonal" (Z)
				if (z_dans_les_cases == True):
					if (int(fields[5]) != 0):
						Z_to_load_3D(int(fields[0]), int(fields[1]), i, j)

						if (NGPU == 1 and NDIMENSIONS == 4):
							data_sur_le_cote_3D_heat_map(int(fields[1]), int(fields[0]), "z", epaisseur_z[i_z_on_side, j_z_on_side][int(fields[0]), int(fields[1])], int(fields[6]), i_z_on_side, j_z_on_side, int(fields[2]))
							epaisseur_z[i_z_on_side, j_z_on_side][int(fields[0]), int(fields[1])] += 1
						else:
							print("La heat map 3D ne gère pas plus de 1 GPU ou ZN :/")
							break

						already_fetched_z[i, j][int(fields[0]), int(fields[1])] = 1

			for line in file_coord_prefetch:
				fields = line.split()
				i, j = get_i_j(int(fields[2]))
				if (int(fields[3]) != 0 and already_fetched_x[i, j][int(fields[0]), int(fields[1])] == 0):
					column_to_load_3D_prefetch(int(fields[0]), int(fields[1]), i, j)

					if (lignes_sur_les_cote == True):
						data_sur_le_cote_3D(int(fields[0]), int(fields[1]), "y", 1, int(fields[6]), "dashed", i_x_on_side, j_x_on_side, int(fields[2]))

				if (int(fields[4]) != 0 and already_fetched_y[i, j][int(fields[0]), int(fields[1])] == 0):
					line_to_load_3D_prefetch(int(fields[0]), int(fields[1]), i, j)

					if (lignes_sur_les_cote == True):
						data_sur_le_cote_3D(int(fields[0]), int(fields[1]), "x", 1, int(fields[6]), "dashed", i_y_on_side, j_y_on_side, int(fields[2]))

				if (z_dans_les_cases == True):
					if (int(fields[5]) != 0 and already_fetched_z[i, j][int(fields[0]), int(fields[1])] == 0):
						Z_to_load_3D_prefetch(int(fields[0]), int(fields[1]), i, j)

		if (ORDO == "HFP"):
			file_load_balance = open(PATH + "/Data_stolen_load_balance.txt", "r")

			for line in file_load_balance:
				fields = line.split()
				i, j = get_i_j(int(fields[2]))
				tache_load_balanced_3D(int(fields[0]), int(fields[1]), int(fields[3]), i, j)

			file_load_balance.close()

			file_last_package = open(PATH + "/last_package_split.txt", "r")

			if (sous_paquets_and_task_stealing == True):
				for line in file_last_package:
					fields = line.split()
					i, j = get_i_j(int(fields[2]))
					hierarchie_paquets[i, j][int(fields[0]), int(fields[1])] = int(fields[4])
			if(NDIMENSIONS == 4):
				for i_bis in range(2):
					for j_bis in range(2):
						for i in range(N):
							for j in range(N):
								if (i != 0):
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][i - 1, j]):
										separation_sous_paquets_3D(i, j, i - 1, j, i_bis, j_bis)
								else:
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][N - 1, j]):
										separation_sous_paquets_3D(i, j, N - 1, j, i_bis, j_bis)
								if (i != N - 1):
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][i + 1, j]):
										separation_sous_paquets_3D(i, j, i + 1, j, i_bis, j_bis)
								else:
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][0, j]):
										separation_sous_paquets_3D(i, j, 0, j, i_bis, j_bis)
								if (j != 0):
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][i, j - 1]):
										separation_sous_paquets_3D(i, j, i, j - 1, i_bis, j_bis)
								else:
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][i, N - 1]):
										separation_sous_paquets_3D(i, j, i, N - 1, i_bis, j_bis)
								if (j != N - 1):
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][i, j + 1]):
										separation_sous_paquets_3D(i, j, i, j + 1, i_bis, j_bis)
								else:
									if (hierarchie_paquets[i_bis, j_bis][i, j] != hierarchie_paquets[i_bis, j_bis][i, 0]):
										separation_sous_paquets_3D(i, j, i, 0, i_bis, j_bis)
			else:
				print("hierarchie 3D not implemented yet for Z != 4")
			file_last_package.close()

		for i in range(NROW):
			for j in range(NCOL):
				string = "K=" + str(j*NCOL+i)
				if NGPU == 1:
					ax[j, i].text(-0.29, -0.09, string, fontsize = 13, color="red", transform=ax[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))


		# Printing
		for i in range(NROW):
			for j in range(NCOL):
				ax[i, j].imshow(m[i, j])
	# End of 3D matrix

elif (APPLI == "Cholesky"):

	# ~ numerotation_des_cases = True
	numerotation_des_cases = False

	# ~ numerotation_des_cases_partielles = True
	numerotation_des_cases_partielles = False

	# ~ text_sous_les_figures = True
	text_sous_les_figures = False

	# ~ lignes_dans_les_cases_et_sur_le_cote = True
	lignes_dans_les_cases_et_sur_le_cote = False

	# ~ nb_load_dans_les_cases = True
	nb_load_dans_les_cases = False

	memory_size_in_tiles = True
	# ~ memory_size_in_tiles = False

	if (memory_size_in_tiles == True):
		MEMOIRE = int(sys.argv[6])
		TILE_SIZE = int(sys.argv[7])

	size_numero_dans_les_cases = 1.3

	NCOL = math.ceil(math.sqrt(N))
	NROW = math.ceil(math.sqrt(N))

	if NGPU > 8 or NGPU == 3 or NGPU == 5 or NGPU == 6 or NGPU == 7:
		print(NGPU, "GPUs not dealt with. Please use 1, 2, 4 or 8 GPUs")
		exit

	if NGPU >= 1:
		fig1, ax1 = plt.subplots(nrows = NROW, ncols = NCOL)
	if NGPU >= 2:
		fig2, ax2 = plt.subplots(nrows = NROW, ncols = NCOL)
	if NGPU >= 4:
		fig3, ax3 = plt.subplots(nrows = NROW, ncols = NCOL)
		fig4, ax4 = plt.subplots(nrows = NROW, ncols = NCOL)
	if NGPU >= 8:
		fig5, ax5 = plt.subplots(nrows = NROW, ncols = NCOL)
		fig6, ax6 = plt.subplots(nrows = NROW, ncols = NCOL)
		fig7, ax7 = plt.subplots(nrows = NROW, ncols = NCOL)
		fig8, ax8 = plt.subplots(nrows = NROW, ncols = NCOL)

	for i in range(NROW):
		for j in range(NCOL):
			if NGPU >= 1:
				ax1[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax1[i, j].set_xticks([])
				ax1[i, j].set_yticks([])
			if NGPU >= 2:
				ax2[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax2[i, j].set_xticks([])
				ax2[i, j].set_yticks([])
			if NGPU >= 4:
				ax3[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax3[i, j].set_xticks([])
				ax3[i, j].set_yticks([])
				ax4[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax4[i, j].set_xticks([])
				ax4[i, j].set_yticks([])
			if NGPU >= 8:
				ax5[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax5[i, j].set_xticks([])
				ax5[i, j].set_yticks([])
				ax6[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax6[i, j].set_xticks([])
				ax6[i, j].set_yticks([])
				ax7[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax7[i, j].set_xticks([])
				ax7[i, j].set_yticks([])
				ax8[i, j].grid(which = 'minor', color = 'black', linestyle = '-', linewidth = 1)
				ax8[i, j].set_xticks([])
				ax8[i, j].set_yticks([])

	row_to_suppr = 1
	for i in range(NCOL*NROW - N):
		if (i%(NCOL) == 0 and i != 0):
			row_to_suppr += 1
		if NGPU >= 1:
			fig1.delaxes(ax1[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])
		if NGPU >= 2:
			fig2.delaxes(ax2[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])
		if NGPU >= 4:
			fig3.delaxes(ax3[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])
			fig4.delaxes(ax4[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])
		if NGPU >= 8:
			fig5.delaxes(ax5[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])
			fig6.delaxes(ax6[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])
			fig7.delaxes(ax7[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])
			fig8.delaxes(ax8[NROW - row_to_suppr, NCOL - 1 - i%(NCOL)])

	if NGPU >= 1:
		m1 = {}
	if NGPU >= 2:
		m2 = {}
	if NGPU >= 4:
		m3 = {}
		m4 = {}
	if NGPU >= 8:
		m5 = {}
		m6 = {}
		m7 = {}
		m8 = {}
	already_fetched_x = {}
	already_fetched_y = {}
	already_fetched_z = {}

	for i in range(NROW):
		for j in range(NCOL):
			if NGPU >= 1:
				m1[i, j] = np.ones((N, N, 3))
			if NGPU >= 2:
				m2[i, j] = np.ones((N, N, 3))
			if NGPU >= 4:
				m3[i, j] = np.ones((N, N, 3))
				m4[i, j] = np.ones((N, N, 3))
			if NGPU >= 8:
				m5[i, j] = np.ones((N, N, 3))
				m6[i, j] = np.ones((N, N, 3))
				m7[i, j] = np.ones((N, N, 3))
				m8[i, j] = np.ones((N, N, 3))

	for i in range(NROW):
		for j in range(NCOL):
			already_fetched_x[i, j] = np.zeros((N, N, 1))
			already_fetched_y[i, j] = np.zeros((N, N, 1))
			already_fetched_z[i, j] = np.zeros((N, N, 1))

	if (memory_size_in_tiles == True):
		taille_1_tuile = TILE_SIZE*TILE_SIZE*4 # Car c'est du simple. Pour LU en double ce sera *8
		nb_tuile__qui_rentre_en_memoire = int((MEMOIRE*1000000)/taille_1_tuile)
		x_to_fill = int(math.sqrt(nb_tuile__qui_rentre_en_memoire))
		y_to_fill = int(math.sqrt(nb_tuile__qui_rentre_en_memoire))
		remaining_tile_to_fill = nb_tuile__qui_rentre_en_memoire - (x_to_fill*y_to_fill)

		if x_to_fill < N:
			x = N-1
			y = 0

			for i in range(0, x_to_fill):
				for j in range(0, y_to_fill):
					if NGPU >= 1:
						m1[0, 0][y+i, x-j, :] = (0, 0, 0)
					if NGPU >= 2:
						m2[0, 0][y+i, x-j, :] = (0, 0, 0)
					if NGPU >= 4:
						m3[0, 0][y+i, x-j, :] = (0, 0, 0)
						m4[0, 0][y+i, x-j, :] = (0, 0, 0)
					if NGPU >= 8:
						m5[0, 0][y+i, x-j, :] = (0, 0, 0)
						m6[0, 0][y+i, x-j, :] = (0, 0, 0)
						m7[0, 0][y+i, x-j, :] = (0, 0, 0)
						m8[0, 0][y+i, x-j, :] = (0, 0, 0)
			for i in range(0, remaining_tile_to_fill):
				if NGPU >= 1:
					m1[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
				if NGPU >= 2:
					m2[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
				if NGPU >= 4:
					m3[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
					m4[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
				if NGPU >= 8:
					m5[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
					m6[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
					m7[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
					m8[0, 0][y, x-x_to_fill, :] = (0, 0, 0)
				if i >= y_to_fill:
					x_to_fill-=1
				else:
					y+=1

	next(file_coord)
	for line in file_coord:
		fields = line.split()
		nb_tache_par_gpu[int(fields[3])] = nb_tache_par_gpu[int(fields[3])] + 1

	file_coord.seek(0)
	next(file_coord)
	for line in file_coord:
		fields = line.split()
		i, j = get_i_j_cholesky(fields[4], NCOL)

		if int(fields[3]) == 0:
			m1[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))
		elif int(fields[3]) == 1:
			m2[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))
		elif int(fields[3]) == 2:
			m3[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))
		elif int(fields[3]) == 3:
			m4[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))
		elif int(fields[3]) == 4:
			m5[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))
		elif int(fields[3]) == 5:
			m6[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))
		elif int(fields[3]) == 6:
			m7[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))
		elif int(fields[3]) == 7:
			m8[i, j][int(fields[1]), int(fields[2]), :] = gradiant_multiple_color(order[int(fields[3])], nb_tache_par_gpu[int(fields[3])], NGPU, int(fields[3]))

		if (numerotation_des_cases == True):
			ax[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
		elif (numerotation_des_cases_partielles == True and order[int(fields[3])]%20 == 0):
			if int(fields[3]) == 0:
				ax1[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif int(fields[3]) == 1:
				ax2[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif int(fields[3]) == 2:
				ax3[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif int(fields[3]) == 3:
				ax4[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif int(fields[3]) == 4:
				ax5[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif int(fields[3]) == 5:
				ax6[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif int(fields[3]) == 6:
				ax7[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif int(fields[3]) == 7:
				ax8[i, j].text(int(fields[2]), int(fields[1]), order[int(fields[3])], va="center", ha="center", color = "white", size = size_numero_dans_les_cases)


		order[int(fields[3])] = order[int(fields[3])] + 1

	if (lignes_dans_les_cases_et_sur_le_cote == True):
		next(file_data)
		for line in file_data:
			fields = line.split()

			i, j = get_i_j_cholesky(fields[7], NCOL)

			if (int(fields[3]) == 1):
				ax1[i, j].plot([int(fields[2]), int(fields[2])], [int(fields[1]) - 0.44, int(fields[1]) + 0.44], '#FFBB96', lw = 1.2, zorder = 5)
				already_fetched_x[i, j][int(fields[2]), int(fields[1])] = 1
			if (int(fields[4]) == 1):
				ax1[i, j].plot([int(fields[2]) - 0.44, int(fields[2]) + 0.44], [int(fields[1]), int(fields[1])], '#FFBB96', lw = 1.2, zorder = 5)
				already_fetched_y[i, j][int(fields[2]), int(fields[1])] = 1
			if (int(fields[5]) == 1):
				ax1[i, j].plot([int(fields[2]) - 0.44, int(fields[2]) + 0.44], [int(fields[1]) - 0.44, int(fields[1]) + 0.44], '#FFBB96', lw = 1.2, zorder = 5)
				already_fetched_z[i, j][int(fields[2]), int(fields[1])] = 1

		next(file_coord_prefetch)
		for line in file_coord_prefetch:
			fields = line.split()

			i, j = get_i_j_cholesky(fields[7], NCOL)

			if (int(fields[3]) == 1 and already_fetched_x[i, j][int(fields[2]), int(fields[1])] == 0):
				ax1[i, j].plot([int(fields[2]), int(fields[2])], [int(fields[1]) - 0.44, int(fields[1]) + 0.44], '#FFBB96', lw = 1.2, zorder = 5, linestyle = "dotted")
			if (int(fields[4]) == 1 and already_fetched_y[i, j][int(fields[2]), int(fields[1])] == 0):
				ax1[i, j].plot([int(fields[2]) - 0.44, int(fields[2]) + 0.44], [int(fields[1]), int(fields[1])], '#FFBB96', lw = 1.2, zorder = 5, linestyle = "dotted")
			if (int(fields[5]) == 1 and already_fetched_z[i, j][int(fields[2]), int(fields[1])] == 0):
				ax1[i, j].plot([int(fields[2]) - 0.44, int(fields[2]) + 0.44], [int(fields[1]) - 0.44, int(fields[1]) + 0.44], '#FFBB96', lw = 1.2, zorder = 5, linestyle = "dotted")


	if (nb_load_dans_les_cases == True):
		next(file_data)
		for line in file_data:
			fields = line.split()

			i, j = get_i_j_cholesky(fields[7], NCOL)

			nb_of_fetch = int(fields[3]) + int(fields[4]) + int(fields[5])
			if (nb_of_fetch == 1):
				ax1[i, j].text(int(fields[2]), int(fields[1]), 1, va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif (nb_of_fetch == 2):
				ax1[i, j].text(int(fields[2]), int(fields[1]), 2, va="center", ha="center", color = "white", size = size_numero_dans_les_cases)
			elif (nb_of_fetch == 3):
				ax1[i, j].text(int(fields[2]), int(fields[1]), 3, va="center", ha="center", color = "white", size = size_numero_dans_les_cases)

			if (int(fields[3]) == 1):
				already_fetched_x[i, j][int(fields[2]), int(fields[1])] = 1
			if (int(fields[4]) == 1):
				already_fetched_y[i, j][int(fields[2]), int(fields[1])] = 1
			if (int(fields[5]) == 1):
				already_fetched_z[i, j][int(fields[2]), int(fields[1])] = 1


		next(file_coord_prefetch)
		for line in file_coord_prefetch:
			fields = line.split()

			i, j = get_i_j_cholesky(fields[7], NCOL)

			nb_of_prefetch = 0
			if (int(fields[3]) == 1 and already_fetched_x[i, j][int(fields[2]), int(fields[1])] == 0):
				nb_of_prefetch += 1
			if (int(fields[4]) == 1 and already_fetched_y[i, j][int(fields[2]), int(fields[1])] == 0):
				nb_of_prefetch += 1
			if (int(fields[5]) == 1 and already_fetched_z[i, j][int(fields[2]), int(fields[1])] == 0):
				nb_of_prefetch += 1

			# ~ if (nb_of_fetch == 0):
			if (nb_of_prefetch == 1):
				ax1[i, j].text(int(fields[2]), int(fields[1]), 1, va="center", ha="center", color = "black", size = size_numero_dans_les_cases)
			elif (nb_of_fetch == 2):
				ax1[i, j].text(int(fields[2]), int(fields[1]), 2, va="center", ha="center", color = "black", size = size_numero_dans_les_cases)
			elif (nb_of_fetch == 3):
				ax1[i, j].text(int(fields[2]), int(fields[1]), 3, va="center", ha="center", color = "black", size = size_numero_dans_les_cases)

	# Adding text under the figures
	for i in range(NROW):
		for j in range(NCOL):
			string = "K=" + str(j*NCOL+i)
			if NGPU >= 1:
				ax1[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax1[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))
			if NGPU >= 2:
				ax2[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax2[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))
			if NGPU >= 4:
				ax3[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax3[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))
				ax4[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax4[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))
			if NGPU == 8:
				ax5[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax5[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))
				ax6[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax6[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))
				ax7[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax7[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))
				ax8[j, i].text(-0.39, 0.1, string, fontsize = 7, color="red", transform=ax8[j, i].transAxes, bbox=dict(facecolor='none', edgecolor='red'))

	# Printing
	for i in range(NROW):
		for j in range(NCOL):
			if NGPU >= 1:
				ax1[i, j].imshow(m1[i, j])
			if NGPU >= 2:
				ax2[i, j].imshow(m2[i, j])
			if NGPU >= 4:
				ax3[i, j].imshow(m3[i, j])
				ax4[i, j].imshow(m4[i, j])
			if NGPU >= 8:
				ax5[i, j].imshow(m5[i, j])
				ax6[i, j].imshow(m6[i, j])
				ax7[i, j].imshow(m7[i, j])
				ax8[i, j].imshow(m8[i, j])

else:
	print("Application not supported; Please Use gemm or cholesky")
	sys.exit(1)

# Closing open files
file_coord.close()
file_data.close()
file_coord_prefetch.close()

if (APPLI == "Matrice3D" or APPLI == "MatriceZ4"):
	image_format = 'svg'
	image_name1 = ORDO + '_M3D_N' + str(N) + "." + "image_format"
	fig.savefig(image_name1, format=image_format, dpi=1200)

if (APPLI == "Matrice_ligne"):
	image_format = 'svg'
	image_name1 = ORDO + '_M2D_N' + str(N) + "." + image_format
	fig.savefig(image_name1, format=image_format, dpi=1200)


if (APPLI == "Cholesky"):
	image_format = 'svg'
	if NGPU >= 1:
		image_name1 = ORDO + '_CHO_N' + str(N) + '_GPU_1.' + image_format
		fig1.savefig(image_name1, format=image_format, dpi=1200)
	if NGPU >= 2:
		image_name2 = ORDO + '_CHO_test_GPU_2.' + image_format
		fig2.savefig(image_name2, format=image_format, dpi=1200)
	if NGPU >= 4:
		image_name3 = ORDO + '_CHO_test_GPU_3.' + image_format
		fig3.savefig(image_name3, format=image_format, dpi=1200)
		image_name4 = ORDO + '_CHO_test_GPU_4.' + image_format
		fig4.savefig(image_name4, format=image_format, dpi=1200)
	if NGPU >= 8:
		image_name5 = ORDO + '_CHO_test_GPU_5.' + image_format
		fig5.savefig(image_name5, format=image_format, dpi=1200)
		image_name6 = ORDO + '_CHO_test_GPU_6.' + image_format
		fig6.savefig(image_name6, format=image_format, dpi=1200)
		image_name7 = ORDO + '_CHO_test_GPU_7.' + image_format
		fig7.savefig(image_name7, format=image_format, dpi=1200)
		image_name8 = ORDO + '_CHO_test_GPU_8.' + image_format
		fig8.savefig(image_name8, format=image_format, dpi=1200)
else:
	plt.show()
