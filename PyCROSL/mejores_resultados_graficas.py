import csv
import pandas as pd
from TestFunctions import TiempoAeropuerto
import matplotlib.pyplot as plt


file_path = 'D:\\master\\segundo cuatri\\TFM\\repositorio_cro\\TFM-CROSL-MV\\PyCROSL\\results\\postfix_taxifunction'

best_pop = pd.read_csv(file_path + "\\last_population_fits_1718905699.csv",header=None)
best_pop = best_pop.iloc[:, 1:].values.tolist()
best_pop = best_pop[1:]

second_best_pop = pd.read_csv(file_path + "\\last_population_fits_1719259136.csv",header=None)
second_best_pop = second_best_pop.iloc[:, 1:].values.tolist()
second_best_pop = second_best_pop[1:]

third_best_pop = pd.read_csv(file_path + "\\last_population_fits_1719078984.csv",header=None)
third_best_pop = third_best_pop.iloc[:, 1:].values.tolist()
third_best_pop = third_best_pop[1:]

fourth_best_pop = pd.read_csv(file_path + "\\last_population_fits_1719169217.csv",header=None)
fourth_best_pop = fourth_best_pop.iloc[:, 1:].values.tolist()
fourth_best_pop = fourth_best_pop[1:]

colors = ["b", "g", "r", "y"]

fits = best_pop
fits_inversos = [(-f1, -f2) for f1,f2 in fits]
f1 = [f1 for f1,f2 in fits_inversos]
f2 = [f2 for f1,f2 in fits_inversos]
plt.scatter(f1,f2, marker='o', label="Solo Multipoint", color=colors[0])

fits = second_best_pop
fits_inversos = [(-f1, -f2) for f1,f2 in fits]
f1 = [f1 for f1,f2 in fits_inversos]
f2 = [f2 for f1,f2 in fits_inversos]
plt.scatter(f1,f2, marker='o', label="Multipoint y Multicross", color=colors[1])

fits = third_best_pop
fits_inversos = [(-f1, -f2) for f1,f2 in fits]
f1 = [f1 for f1,f2 in fits_inversos]
f2 = [f2 for f1,f2 in fits_inversos]
plt.scatter(f1,f2, marker='o', label="Multipoint y 1punto", color=colors[2])

fits = fourth_best_pop
fits_inversos = [(-f1, -f2) for f1,f2 in fits]
f1 = [f1 for f1,f2 in fits_inversos]
f2 = [f2 for f1,f2 in fits_inversos]
plt.scatter(f1,f2, marker='o', label="Multipoint, Multicross, permutación y 1punto", color=colors[3])

plt.legend(loc='upper left')
plt.xlabel("Pasajeros")
plt.ylabel("Taxi")
plt.title("Comparativa de mejores resultados")
plt.show()

