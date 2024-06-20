from CoralPopulation import CoralPopulation
import pandas as pd
from CRO_SL import CRO_SL
from SubstrateInt import *
from TestFunctions import TiempoAeropuerto
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import hsv_to_rgb
import os
import warnings
import time
import numpy as np

warnings.filterwarnings("ignore")


def plot_all_populations(pops_fits):
    colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
          for i in range(len(pops_fits))]
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    for i,fits in enumerate(pops_fits):
        fits_inversos = [(-f1, -f2) for f1,f2 in fits]
        f1 = [f1 for f1,f2 in fits_inversos]
        f2 = [f2 for f1,f2 in fits_inversos]
        plt.scatter(f1,f2, marker='o', label="Iter "+str((Niter//Npops_to_show)*i))
    plt.legend(loc='upper left')
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Fitness of all populations")
    plt.show()


def plot_lasts_pareto_optimals(paretos, name="pareto_optimal_solutions.png"):
    paretos_to_plot = len(paretos) if len(paretos) < 5 else len(paretos) // 5

    colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
          for i in range(paretos_to_plot)]
    plt.rc('axes', prop_cycle=(cycler('color', colors)))

     
    for iter, pareto in enumerate(paretos):
        if iter % paretos_to_plot == 0 or iter == len(paretos) - 1:
            f1 = [f1 for f1,f2 in pareto]
            f2 = [f2 for f1,f2 in pareto]
            plt.scatter(f1,f2, marker='o', label="Iter "+str(iter))
    plt.legend(loc='upper left')
    plt.xlabel("Taxi Time")
    plt.ylabel("Passengers Time")
    plt.title("Pareto Optimal Solutions between iterations")
    plt.show()

def save_results(pareto_optimal, pareto_optimal_fits, last_population, last_population_fits):
    pd.DataFrame(pareto_optimal).to_csv(os.getcwd() + "\\results\\postfix_taxifunction\\pareto_optimal_" + str(int(start_time)) + ".csv")
    pd.DataFrame(pareto_optimal_fits).to_csv(os.getcwd() + "\\results\\postfix_taxifunction\\pareto_optimal_fits_" + str(int(start_time)) + ".csv")
    pd.DataFrame(last_population).to_csv(os.getcwd() + "\\results\\postfix_taxifunction\\last_population_" + str(int(start_time)) + ".csv")
    pd.DataFrame(last_population_fits).to_csv(os.getcwd() + "\\results\\postfix_taxifunction\\last_population_fits_" + str(int(start_time)) + ".csv")

def plot_minimos(minimos_its):
    min_taxi = [minimo[0] for minimo in minimos_its]
    min_passengers = [minimo[1] for minimo in minimos_its]
    colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
          for i in range(len(minimos_its)//4)]
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.scatter(min_taxi, min_passengers, marker='o',label="Minimos")
    plt.xlabel("Taxi Time")
    plt.ylabel("Passengers Time")
    plt.title("Minimos de las poblaciones")
    plt.legend(loc='upper left')
    plt.show()

def plot_maximos(maximos_its):
    max_taxi = [maximo[0] for maximo in maximos_its]
    max_passengers = [maximo[1] for maximo in maximos_its]
    colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
          for i in range(len(maximos_its)//4)]
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.scatter(max_taxi, max_passengers, marker='o',label="Maximos")
    plt.xlabel("Taxi Time")
    plt.ylabel("Passengers Time")
    plt.title("Maximos de las poblaciones")
    plt.legend(loc='upper left')
    plt.show()

params = {"F": 0.7, "Pr": 0.8, "Cr": 0.75, "N": 400}
substrates = [
    SubstrateInt("Multipoint", params),
    SubstrateInt("2point", params),
    SubstrateInt("Xor", params)
]

params = {
    "popSize": 100,
    "rho": 0.6,
    "Fb": 0.98,
    "Fd": 0.2,
    "Pd": 0.9,
    "k": 3,
    "K": 20,
    "group_subs": False,

    "stop_cond": "Neval",
    "time_limit": 4000.0,
    "Ngen": 3500,
    "Neval": 100,
    "fit_target": 1000,

    "verbose": False,
    "v_timer": 1,

    "dynamic": True,
    "dyn_method": "success",
    "dyn_metric": "avg",
    "dyn_steps": 75,
    "prob_amp": 0.1
}
# data required for optimization is read
# la linea de abajo es unicamente necesaria por la configuracion de mi VSCode
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

c = CRO_SL(f, substrates, params)
population = CoralPopulation(f, substrates, params)
start_time = time.time()
print("Starting optimization")
population.generate_random()
population.generate_substrates(0)
print("Evolving with substrates")
larvae = population.evolve_with_substrates()
population.larvae_setting(larvae)#aqui se añaden nuevas soluciones
population.depredation()#aqui se eliminan las peores soluciones por lo que puede no mantenerse el tamaño de la población
mejor, mejorfit = population.best_solution()

pops = []
pops_fits = []
pops_fits.append([individuo.get_fitness() for individuo in population.population])
pops.append(population.population)
iter=0
minimos_its = []
maximos_its = []
paretos_optimos_fits = []
paretos_optimos_pop = []
while iter < Niter and int(time.time()-start_time) < 86400: #f.counter < Neval: 86400 es el numero de segundos en un dia
    iter += 1
    print(f"Empiezan las iteraciones del algoritmo: Niter -> {iter}")
    print(f"len de la population: {len(population.population)}")
    population.generate_substrates(0)
    larvae = population.evolve_with_substrates()
    population.larvae_setting(larvae)
    population.depredation()
    mejor, mejorfit = population.best_solution()
    paretos_optimos_pop.append(mejor)
    paretos_optimos_fits.append(list(mejorfit))
    fitness_0 = [individuo.get_fitness()[0] * -1 for individuo in population.population]
    fitness_1 = [individuo.get_fitness()[1] * -1 for individuo in population.population]
    min_fitness_0 = min(fitness_0)
    min_fitness_1 = min(fitness_1)
    minimos_its.append((min_fitness_0, min_fitness_1))
    max_fitness_0 = max(fitness_0)
    max_fitness_1 = max(fitness_1)
    maximos_its.append((max_fitness_0, max_fitness_1))
    print(f"tiempo -> {time.time()-start_time}")
    if((iter % (Niter//Npops_to_show)) == 0 or iter == Niter):
        pops_fits.append([individuo.get_fitness() for individuo in population.population])
        pops.append([coral.solution for coral in population.population])
mejor, mejorfit = population.best_solution()
print(mejorfit)
save_results(paretos_optimos_pop[-1], paretos_optimos_fits[-1], pops[-1], pops_fits[-1])
plot_minimos(minimos_its)
plot_maximos(maximos_its)
plot_all_populations(pops_fits)
plot_lasts_pareto_optimals(paretos_optimos_fits)

#Parte de análisis de soluciones
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sol = pd.read_csv("./data/solution_comp.csv")
for a in range(0, sol.__len__()):
    sol.iloc[a,0] = np.where(stands == sol.iloc[a,0])[0][-1]
fitness_real = f.fitness(sol) #devuelve el fitness con el que nos queremos comparar
fitness_real = (fitness_real[0]*-1, fitness_real[1]*-1)
solopt = pd.read_csv(os.getcwd() + "\\results\\postfix_taxifunction\\pareto_optimal_" + str(int(start_time)) + ".csv")
solopt = solopt.iloc[:, 1:]
solopt_fitness = pd.read_csv(os.getcwd() + "\\results\\postfix_taxifunction\\pareto_optimal_fits_" + str(int(start_time)) + ".csv")
solopt_fitness = solopt_fitness.iloc[:, 1:]
#solopt.rename(columns={"0": "stand"}, inplace=True)
#solopt = solopt.drop(columns=["Unnamed: 0"])
"""
#Para obtener los tiempos y fuel de cada vuelo de las soluciones:
solution = solopt
take_off_time = 120
total_time = np.zeros(data.shape[0])
# Se obtienen los stands asignados en la solucion
data.loc[data["cod"] == 1, "stand"] = [stands.stands[i] for i in solution.values]
data.loc[:, "ts_stand"] = np.zeros(data.shape[0])
data.loc[:, "ts_rw"] = np.zeros(data.shape[0])
data.loc[:, "duration_stop"] = np.zeros(data.shape[0])
data.loc[:, "ts_climb"] = np.zeros(data.shape[0])
for index in range(0, data.shape[0]):
    # Cuando cod es 0, se busca el stand del mismo avión
    if data.loc[index, "cod"] == 0:
        data.loc[index, "stand"] = data.loc[
            np.where(data.loc[0:(index - 1), "aircraft_id"] == data.loc[index, "aircraft_id"])[0][
                -1], "stand"]
    # Para los aterrizajes, se calcula duración como suma de tiempo medio de parada y tiempo medio de Tin
    if data.loc[index, "flight_type"] == 2:
        if data.loc[index, "runway"] == "32L":
            column_rw = "media32L"
        elif data.loc[index, "runway"] == "32R":
            column_rw = "media32R"
        elif data.loc[index, "runway"] == "18L":
            column_rw = "media18L"
        else:
            column_rw = "media18R"
        duration_to_stand = np.nan
        if data.loc[index, "stand"] in set(Tin.stand):
            duration_to_stand = \
            Tin.loc[Tin["stand"] == data.loc[index, "stand"], column_rw].values[
                0]
        duration_stop = 0
        if data.loc[index, "stand"] in set(Tstp.stand):
            duration_stop = Tstp.loc[Tstp["stand"] == data.loc[index, "stand"], "media"].values[
                0]
        # TODO: Sustituir este 500 por la media de Tin
        duration_tin = (duration_to_stand if not pd.isna(duration_to_stand) else 500) + duration_stop
        data.loc[index, "ts_stand"] = data.loc[index, "ts_app_2"] + duration_tin
        total_time[index] = duration_tin
    # Para despegues, se calcula timestamp de llegada al runway como timestamp de taxi out mas tiempo medio de Tout
    elif data.loc[index, "flight_type"] == 1:
        if data.loc[index, "runway"] == "36L":
            column_rw = "media36L"
        elif data.loc[index, "runway"] == "36R":
            column_rw = "media36R"
        elif data.loc[index, "runway"] == "14L":
            column_rw = "media14L"
        else:
            column_rw = "media14R"
        duration_to_stand = np.nan
        if data.loc[index, "stand"] in set(Tout.stand):
            duration_to_stand = \
                Tout.loc[Tout["stand"] == data.loc[index, "stand"], column_rw].values[0]
        # TODO: Sustituir este 500 por la media de Tout
        data.loc[index, "ts_rw"] = data.loc[index, "ts_taxi_out_1"] + (
            duration_to_stand if not pd.isna(duration_to_stand) else 500)
# Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 36L
data36L = data.loc[data["runway"] == "36L"].sort_values(by="ts_rw")
previous_index = -1
for index, row in data36L.iterrows():
    # Si es el primer vuelo en ese runway, no tiene que parar
    if previous_index == -1:
        data.loc[index, "duration_stop"] = 0
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + take_off_time
    # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
    else:
        data.loc[index, "duration_stop"] = max(0,
                                                    data.loc[previous_index, "ts_climb"] - data.loc[
                                                        index, "ts_rw"])
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + data.loc[
            index, "duration_stop"] + take_off_time
    previous_index = index
    # La duración del despegue es desde el taxi out hasta el climb
    total_time[index] = data.loc[index, "ts_climb"] - data.loc[index, "ts_taxi_out_1"]
# Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 36R
data36R = data.loc[data["runway"] == "36R"].sort_values(by="ts_rw")
previous_index = -1
for index, row in data36R.iterrows():
    # Si es el primer vuelo en ese runway, no tiene que parar
    if previous_index == -1:
        data.loc[index, "duration_stop"] = 0
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + take_off_time
    # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
    else:
        data.loc[index, "duration_stop"] = max(0,
                                                    data.loc[previous_index, "ts_climb"] - data.loc[
                                                        index, "ts_rw"])
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + data.loc[
            index, "duration_stop"] + take_off_time
    previous_index = index
    # La duración del despegue es desde el taxi out hasta el climb
    total_time[index] = data.loc[index, "ts_climb"] - data.loc[index, "ts_taxi_out_1"]
# Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 14L
data14L = data.loc[data["runway"] == "14L"].sort_values(by="ts_rw")
previous_index = -1
for index, row in data14L.iterrows():
    # Si es el primer vuelo en ese runway, no tiene que parar
    if previous_index == -1:
        data.loc[index, "duration_stop"] = 0
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + take_off_time
    # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
    else:
        data.loc[index, "duration_stop"] = max(0, data.loc[previous_index, "ts_climb"] -
                                                    data.loc[index, "ts_rw"])
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + data.loc[
            index, "duration_stop"] + take_off_time
    previous_index = index
    # La duración del despegue es desde el taxi out hasta el climb
    total_time[index] = data.loc[index, "ts_climb"] - data.loc[index, "ts_taxi_out_1"]
# Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 14R
data14R = data.loc[data["runway"] == "14R"].sort_values(by="ts_rw")
previous_index = -1
for index, row in data14R.iterrows():
    # Si es el primer vuelo en ese runway, no tiene que parar
    if previous_index == -1:
        data.loc[index, "duration_stop"] = 0
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + take_off_time
    # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
    else:
        data.loc[index, "duration_stop"] = max(0, data.loc[previous_index, "ts_climb"] -
                                                    data.loc[index, "ts_rw"])
        data.loc[index, "ts_climb"] = data.loc[index, "ts_rw"] + data.loc[
            index, "duration_stop"] + take_off_time
    previous_index = index
    # La duración del despegue es desde el taxi out hasta el climb
    total_time[index] = data.loc[index, "ts_climb"] - data.loc[index, "ts_taxi_out_1"]
# Parte para calcular el consumo de fuel
total_fuel = np.zeros(data.shape[0])
media_motores = 2.3836
media_consumo = 0.1808
#Parte para calcular el consumo de fuel
for a in range(0, total_time.shape[0]):
    if data.loc[a, "equip"] in set(emissions["equip"]):
        if pd.isna(emissions.loc[emissions["equip"] == data.loc[a, "equip"],"number engines"].values[0]):
            if pd.isna(emissions.loc[emissions["equip"] == data.loc[a, "equip"],"Fuel Flow Idle (kg/sec)"].values[0]):
                total_fuel[a] = media_motores * total_time[a] * media_consumo
            else:
                total_fuel[a] = media_motores * total_time[a] * emissions.loc[emissions["equip"] == data.loc[a, "equip"], "Fuel Flow Idle (kg/sec)"]
        else:
            if pd.isna(emissions.loc[emissions["equip"] == data.loc[a, "equip"],"Fuel Flow Idle (kg/sec)"].values[0]):
                total_fuel[a] = emissions.loc[emissions["equip"] == data.loc[a, "equip"], "number engines"] * \
                                total_time[a] * media_consumo
            else:
                total_fuel[a] = emissions.loc[emissions["equip"] == data.loc[a, "equip"], "number engines"] * total_time[a] * emissions.loc[emissions["equip"] == data.loc[a, "equip"], "Fuel Flow Idle (kg/sec)"]
    else:  # si no tenemos los datos de consumo del avión multiplicamos el tiempo por un valor costante
        total_fuel[a] = media_motores * media_consumo * total_time[a]

pd.DataFrame(total_fuel).to_csv("esc1_sol_fuels_contr.csv") #Sol del controlador
pd.DataFrame(total_time).to_csv("esc1_sol_times_contr.csv") #Sol del controlador
total_time_contr = total_time
total_fuel_contr = total_fuel
pd.DataFrame(total_fuel).to_csv("esc1_sol_fuels_cro1.csv") #Sol del cro
diff = 100*(total_fuel_contr.sum()-total_fuel.sum())/total_fuel_contr.sum()
"""
# Define the data
colors = ['b']*solopt_fitness.shape[0]
colors.append('r')
list_bars = ["Individuo " + str(i + 1) for i in range(solopt_fitness.shape[0])]
list_bars.append("Real")
fitness_taxi = [f1 for f1,f2 in solopt_fitness.values]
fitness_pasajeros = [f2 for f1,f2 in solopt_fitness.values]
fitness_taxi_pasajeros = fitness_real[0]
fitness_real_pasajeros = fitness_real[1]

fitness_pasajeros.append(fitness_real_pasajeros)
fitness_taxi.append(fitness_taxi_pasajeros)

plt.bar(list_bars, fitness_pasajeros, color=colors)
plt.xlabel("Tiempo de Pasajeros") 
plt.ylabel("Tiempo en segundos") 
plt.show()

plt.bar(list_bars, fitness_taxi, color=colors)
plt.xlabel("Tiempo de Taxi") 
plt.ylabel("Tiempo en segundos") 
plt.show()
"""
# Display the plot
hourly_sum = horas.groupby(pd.Grouper(freq='H')).sum()

hourly_sum.plot(kind='bar')
plt.show()

#ahorro por terminales
data.loc[data.loc[:,"cod"] == 1, "stand"] = stands.loc[solopt.values.reshape(-1),"stands"].values
stands_used = pd.DataFrame(data.loc[:, "stand"])
stands_used.rename(columns={"stand": "stands"}, inplace=True)
stands_used = pd.merge(stands_used, stands, on='stands')
stands_used["time_dif"] = total_time-total_time_contr
terminals_sum = stands_used.groupby("terminal").sum()

terminals_sum.plot(kind='bar')
plt.show()

#diferencia de tiempos y fuel por terminales
stands_dia = pd.read_csv("./data/opt_stands.csv")
dif = pd.DataFrame()
data_contr = data.copy()
data_contr.loc[data_contr.loc[:,"cod"] == 1, "stand"] = stands_dia.loc[sol.values.reshape(-1),"stands"].values
data_contr["fuel"] = total_fuel_contr
data_contr["time"] = total_time_contr
data_contr.rename(columns={"stand": "stands"}, inplace=True)
data_contr = pd.merge(data_contr, stands, on='stands')

data_opti = data.copy()
data_opti.loc[data_opti.loc[:,"cod"] == 1, "stand"] = stands.loc[solopt.values.reshape(-1),"stands"].values
data_opti["fuel"] = total_fuel
data_opti["time"] = total_time
data_opti.rename(columns={"stand": "stands"}, inplace=True)
data_opti = pd.merge(data_opti, stands, on='stands')

suma_contr = data_contr.loc[:, ("fuel", "terminal")].groupby("terminal").sum()
suma_opti = data_opti.loc[:, ("fuel", "terminal")].groupby("terminal").sum()
dif["dif_fuel"] = suma_opti["fuel"] - suma_contr["fuel"]
dif.plot(kind='bar')
plt.show()

dif = pd.DataFrame()
suma_contr = data_contr.loc[:, ("time", "terminal")].groupby("terminal").sum()
suma_opti = data_opti.loc[:, ("time", "terminal")].groupby("terminal").sum()
dif["dif_time"] = suma_opti["time"] - suma_contr["time"]

dif.plot(kind='bar')
plt.show()

#ahorro por compañías aéreas
data_companies = pd.DataFrame(data.loc[:,"company"])
data_companies["fuel_dif"] = total_fuel-total_fuel_contr
companies_sum = data_companies.groupby("company").sum()

companies_sum.loc[["Iberia Airlines", "Iberia Express","Air Europa","Ryanair"],:].plot(kind='bar')
plt.show()
"""