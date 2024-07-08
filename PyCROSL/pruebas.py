import pandas as pd
import numpy as np
from TestFunctions import TiempoAeropuerto
import os


os.chdir("D:\\master\\segundo cuatri\\TFM\\repositorio_cro\\TFM-CROSL-MV\\PyCROSL")
data = pd.read_csv("./data/opt_data.csv")
Tin = pd.read_csv("./data/opt_Tin.csv")
Tout = pd.read_csv("./data/opt_Tout.csv")
stands = pd.read_csv("./data/opt_stands.csv")
Tstp = pd.read_csv("./data/opt_Tstp.csv")
emissions = pd.read_csv("./data/plane_emissions.csv")
#stands = pd.read_csv("./data/TodosStands/opt_stands.csv")
dict_tiempos_salidas = pd.read_excel("./data/stand_tiempo_salidas.xlsx")
dict_tiempos_llegadas = pd.read_excel("./data/stand_tiempo_llegadas.xlsx")
option = "time"
# number of optimization variables
Nvar = data["cod"].sum()
Neval = 1000
Niter = 1000
Npops_to_show = 5
# bounds of the encoding
bounds = [0, stands.__len__() - 1]

size = Nvar

#Parte de optimización con el CRO
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = TiempoAeropuerto(Nvar, stands, data, bounds, Tin, Tout, Tstp, emissions, option, dict_tiempos_llegadas, dict_tiempos_salidas)

original = pd.read_csv('D:\\master\\segundo cuatri\\TFM\\repositorio_cro\\TFM-CROSL-MV\\PyCROSL\\data\\20220715_results.csv')

solution = pd.read_csv('D:\\master\\segundo cuatri\\TFM\\repositorio_cro\\TFM-CROSL-MV\\PyCROSL\\data\\solution_comp.csv')

stands = pd.read_csv('D:\\master\\segundo cuatri\\TFM\\repositorio_cro\\TFM-CROSL-MV\\PyCROSL\\data\\opt_stands.csv')

mejores_fits = pd.read_csv('D:\\master\\segundo cuatri\\TFM\\repositorio_cro\\TFM-CROSL-MV\\PyCROSL\\results\\postfix_taxifunction\\last_population_fits_1718905699.csv')
mejores_fits = mejores_fits.iloc[:, 1:]
sol = pd.read_csv("./data/solution_comp.csv")
for a in range(0, sol.__len__()):
    sol.iloc[a,0] = np.where(stands == sol.iloc[a,0])[0][-1]
fitness_real = f.fitness(sol)
fitness_real = (fitness_real[0]*-1, fitness_real[1]*-1)

mejor_fitness_taxi = [f1*-1 for f1,f2 in mejores_fits.values]
mejor_fitness_pasajeros = [f2*-1 for f1,f2 in mejores_fits.values]


dif_taxi = fitness_real[0] - min(mejor_fitness_taxi)
dif_pasajeors = min(mejor_fitness_pasajeros) - fitness_real[1]

print(f"mejora en porcentaje del taxi -> {(dif_taxi/fitness_real[0])*100}")

print(f"mejora en segundos del taxi -> {dif_taxi}")
print(f"sigue siendo peor en minutos de pasajeros -> {dif_pasajeors}")



