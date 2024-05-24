import time

import numpy as np
import pandas as pd
from PyCROSL.AbsObjectiveFunc import AbsObjectiveFunc
from numba import jit
from collections import deque

class MaxOnes(AbsObjectiveFunc):
    def __init__(self, size, opt="max"):
        self.size = size
        super().__init__(self.size, opt)

    def objective(self, solution):
        return solution.sum()
    
    def random_solution(self):
        return (np.random.random(self.size) < 0.5).astype(np.int32)
    
    def repair_solution(self, solution):
        return (solution.copy() >= 0.5).astype(np.int32)

class DiophantineEq(AbsObjectiveFunc):
    def __init__(self, size, coeff, target, opt="min"):
        self.size = size
        self.coeff = coeff
        self.target = target
        super().__init__(self.size, opt)
    
    def objective(self, solution):
        return abs((solution*self.coeff).sum() - self.target)
    
    def random_solution(self):
        return (np.random.randint(-100, 100, size=self.size)).astype(np.int32)
    
    def repair_solution(self, solution):
        return solution.astype(np.int32)

class MaxOnesReal(AbsObjectiveFunc):
    def __init__(self, size, opt="max"):
        self.size = size
        super().__init__(self.size, opt)

    def objective(self, solution):
        return solution.sum()
    
    def random_solution(self):
        return np.random.random(self.size)
    
    def repair_solution(self, solution):
        return np.clip(solution.copy(), 0, 1)

# https://www.scientificbulletin.upb.ro/rev_docs_arhiva/rez0cb_759909.pdf

class Sphere(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objective(self, solution):
        return (solution**2).sum()
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class Rosenbrock(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objective(self, solution):
        return rosenbrock(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class Rastrigin(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objective(self, solution):
        return rastrigin(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def repair_solution(self, solution):
        return np.clip(solution, -5.12, 5.12)

class Test1(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, name="Test function")

    def objective(self, solution):
        return sum([(2*solution[i-1] + solution[i]**2*solution[i+1]-solution[i-1])**2 for i in range(1, solution.size-1)])
    
    def random_solution(self):
        return 4*np.random.random(self.size)-2
    
    def repair_solution(self, solution):
        return np.clip(solution, -2, 2)

class TimeTest(AbsObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, name="Test function with delay")

    def objective(self, solution):
        time.sleep(0.05)
        return sum([(2*solution[i-1] + solution[i]**2*solution[i+1]-solution[i-1])**2 for i in range(1, solution.size-1)])

    def random_solution(self):
        return 4*np.random.random(self.size)-2

    def repair_solution(self, solution):
        return np.clip(solution, -2, 2)






class TiempoAeropuerto(AbsObjectiveFunc):


    """
        Expected input of dict_tiempos_llegadas and dict_tiempos_salidas:
        dict_tiempos_llegadas = { 
            "Number_stand" : 10,
            "Number_stand2": 20,
            "Number_stand3": 30
            ...
        }
        dict_tiempos_salidas = {
            "Number_stand" : 10,
            "Number_stand2": 20,
            "Number_stand3": 30
            ...
        }


    """

    def __init__(self, size, stands, data, bounds, tin, tout, tstp, emissions, option, dict_tiempos_llegadas, dict_tiempos_salidas ,opt="min"):
        self.size = size
        self.stands = stands
        self.data = data
        self.bounds = bounds
        self.emissions = emissions
        self.Tin = tin
        self.Tout = tout
        self.Tstp = tstp
        self.option = option
        self.dict_tiempos_llegadas = dict_tiempos_llegadas
        self.dict_tiempos_salidas = dict_tiempos_salidas
        self.factor = 1
        super().__init__(self.size, opt)


    def fitness_pasajeros(self, solution):
        solution = pd.DataFrame(solution)
        # esto se hace por si acaso pero no hace falta por que siempre se ejecuta despues de objective que ya lo hace
        self.data.loc[self.data["cod"] == 1, "stand"] = [self.stands.stands[i] for i in solution.values]
        global_time = 0
        aux_flights = zip(self.data.loc[self.data["cod"] == 1, "stand"], self.data.loc[self.data["cod"] == 1, "flight_type"])
        for assigned_stand, flight_type in aux_flights: #esto no se si se debe hacer solo sobre los que son cod 1, no se si se debe hacer sobre todos 
            if assigned_stand.isdigit():
                assigned_stand = str(int(assigned_stand))
            if flight_type == 2: # aterrizaje es decir llegadas
                global_time += self.dict_tiempos_llegadas.loc[np.where(self.dict_tiempos_llegadas == assigned_stand)[0][0], "Tiempo_Llegada"]
            elif flight_type == 1: # despegue es decir salidas
                global_time += self.dict_tiempos_salidas.loc[np.where(self.dict_tiempos_salidas == assigned_stand)[0][0], "Tiempo_Salida"]
        return global_time

    def objective(self, solution):
        solution = pd.DataFrame(solution)
        take_off_time = 120
        total_time = np.zeros(self.data.shape[0])
        # Se obtienen los stands asignados en la solucion
        self.data.loc[self.data["cod"] == 1, "stand"] = [self.stands.stands[i] for i in solution.values]
        self.data.loc[:, "ts_stand"] = np.zeros(self.data.shape[0])
        self.data.loc[:, "ts_rw"] = np.zeros(self.data.shape[0])
        self.data.loc[:, "duration_stop"] = np.zeros(self.data.shape[0])
        self.data.loc[:, "ts_climb"] = np.zeros(self.data.shape[0])
        for index in range(0, self.data.shape[0]):
            # Cuando cod es 0, se busca el stand del mismo avión
            if self.data.loc[index, "cod"] == 0:
                self.data.loc[index, "stand"] = self.data.loc[
                    np.where(self.data.loc[0:(index - 1), "aircraft_id"] == self.data.loc[index, "aircraft_id"])[0][
                        -1], "stand"]
            # Para los aterrizajes, se calcula duración como suma de tiempo medio de parada y tiempo medio de Tin
            if self.data.loc[index, "flight_type"] == 2:
                if self.data.loc[index, "runway"] == "32L":
                    column_rw = "media32L"
                elif self.data.loc[index, "runway"] == "32R":
                    column_rw = "media32R"
                elif self.data.loc[index, "runway"] == "18L":
                    column_rw = "media18L"
                else:
                    column_rw = "media18R"
                duration_to_stand = np.nan
                if self.data.loc[index, "stand"] in set(self.Tin.stand):
                    duration_to_stand = \
                        self.Tin.loc[self.Tin["stand"] == self.data.loc[index, "stand"], column_rw].values[
                            0]
                duration_stop = 0
                if self.data.loc[index, "stand"] in set(self.Tstp.stand):
                    duration_stop = self.Tstp.loc[self.Tstp["stand"] == self.data.loc[index, "stand"], "media"].values[
                        0]
                # TODO: Sustituir este 500 por la media de Tin
                duration_tin = (duration_to_stand if not pd.isna(duration_to_stand) else 500) + duration_stop
                self.data.loc[index, "ts_stand"] = self.data.loc[index, "ts_app_2"] + duration_tin
                total_time[index] = duration_tin
            # Para despegues, se calcula timestamp de llegada al runway como timestamp de taxi out mas tiempo medio de Tout
            elif self.data.loc[index, "flight_type"] == 1:
                if self.data.loc[index, "runway"] == "36L":
                    column_rw = "media36L"
                elif self.data.loc[index, "runway"] == "36R":
                    column_rw = "media36R"
                elif self.data.loc[index, "runway"] == "14L":
                    column_rw = "media14L"
                else:
                    column_rw = "media14R"
                duration_to_stand = np.nan
                if self.data.loc[index, "stand"] in set(self.Tout.stand):
                    duration_to_stand = \
                        self.Tout.loc[self.Tout["stand"] == self.data.loc[index, "stand"], column_rw].values[0]
                # TODO: Sustituir este 500 por la media de Tout
                self.data.loc[index, "ts_rw"] = self.data.loc[index, "ts_taxi_out_1"] + (
                    duration_to_stand if not pd.isna(duration_to_stand) else 500)
        # Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 36L
        data36L = self.data.loc[self.data["runway"] == "36L"].sort_values(by="ts_rw")
        previous_index = -1
        for index, row in data36L.iterrows():
            # Si es el primer vuelo en ese runway, no tiene que parar
            if previous_index == -1:
                self.data.loc[index, "duration_stop"] = 0
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
            # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
            else:
                self.data.loc[index, "duration_stop"] = max(0,
                                                       self.data.loc[previous_index, "ts_climb"] - self.data.loc[
                                                           index, "ts_rw"])
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[
                    index, "duration_stop"] + take_off_time
            previous_index = index
            # La duración del despegue es desde el taxi out hasta el climb
            total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
        # Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 36R
        data36R = self.data.loc[self.data["runway"] == "36R"].sort_values(by="ts_rw")
        previous_index = -1
        for index, row in data36R.iterrows():
            # Si es el primer vuelo en ese runway, no tiene que parar
            if previous_index == -1:
                self.data.loc[index, "duration_stop"] = 0
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
            # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
            else:
                self.data.loc[index, "duration_stop"] = max(0,
                                                       self.data.loc[previous_index, "ts_climb"] - self.data.loc[
                                                           index, "ts_rw"])
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[
                    index, "duration_stop"] + take_off_time
            previous_index = index
            # La duración del despegue es desde el taxi out hasta el climb
            total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
        # Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 14L
        data14L = self.data.loc[self.data["runway"] == "14L"].sort_values(by="ts_rw")
        previous_index = -1
        for index, row in data14L.iterrows():
            # Si es el primer vuelo en ese runway, no tiene que parar
            if previous_index == -1:
                self.data.loc[index, "duration_stop"] = 0
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
            # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
            else:
                self.data.loc[index, "duration_stop"] = max(0, self.data.loc[previous_index, "ts_climb"] -
                                                       self.data.loc[index, "ts_rw"])
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[
                    index, "duration_stop"] + take_off_time
            previous_index = index
            # La duración del despegue es desde el taxi out hasta el climb
            total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
        # Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 14R
        data14R = self.data.loc[self.data["runway"] == "14R"].sort_values(by="ts_rw")
        previous_index = -1
        for index, row in data14R.iterrows():
            # Si es el primer vuelo en ese runway, no tiene que parar
            if previous_index == -1:
                self.data.loc[index, "duration_stop"] = 0
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
            # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
            else:
                self.data.loc[index, "duration_stop"] = max(0, self.data.loc[previous_index, "ts_climb"] -
                                                       self.data.loc[index, "ts_rw"])
                self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[
                    index, "duration_stop"] + take_off_time
            previous_index = index
            # La duración del despegue es desde el taxi out hasta el climb
            total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
        #self.data.to_csv("prueba_data.csv")
        total_fuel = np.zeros(self.data.shape[0])
        media_motores = 2.3836
        media_consumo = 0.1808
        #Parte para calcular el consumo de fuel
        for a in range(total_time.shape[0]):
            if self.data.loc[a, "equip"] in set(self.emissions["equip"]):
                if pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"number engines"].values[0]):
                    if pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"Fuel Flow Idle (kg/sec)"].values[0]):
                        total_fuel[a] = media_motores * total_time[a] * media_consumo
                    else:
                        total_fuel[a] = media_motores * total_time[a] * self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "Fuel Flow Idle (kg/sec)"]
                else:
                    if pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"Fuel Flow Idle (kg/sec)"].values[0]):
                        total_fuel[a] = self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "number engines"] * \
                                        total_time[a] * media_consumo
                    else:
                        total_fuel[a] = self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "number engines"] * total_time[a] * self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "Fuel Flow Idle (kg/sec)"]
            else:  # si no tenemos los datos de consumo del avión multiplicamos el tiempo por un valor costante
                total_fuel[a] = media_motores * media_consumo * total_time[a]
        fitnessOriginal = total_fuel.sum() if self.option == "fuel" else total_time.sum()
        if self.dict_tiempos_salidas is not None and self.dict_tiempos_llegadas is not None:
            fitnessPasajeros = self.fitness_pasajeros(solution)
            return (fitnessOriginal, fitnessPasajeros)
        return fitnessOriginal

    def random_solution(self):
        return np.round((self.bounds[1] - self.bounds[0]) * np.random.random(self.size) + self.bounds[0]).astype(int)

    def check_bounds(self, solution):
        listapr = ['TERMINAL AVIACIÓN GENERAL', 'T-123 REMOTO']
        stands_util = deque()
        solution_mod = pd.DataFrame(np.array(np.clip(solution, self.bounds[0], self.bounds[1])).reshape(-1, 1))
        self.data.loc[self.data["cod"] == 1, "stand"] = solution_mod.values
        solution_mod = pd.DataFrame(np.array(solution_mod).reshape(1, -1))
        ocupado = pd.DataFrame(np.zeros(np.shape(self.stands)[0]))
        ind = 0
        for index in range(self.data.shape[0]):
            if self.data.loc[index, "cod"] == 1 and \
                    self.data.loc[index, "flight_type"] == 2 and \
                    (ocupado.iloc[int(solution_mod[ind][0])] == 0)[0]:
                if self.data.loc[index, "equip"] not in self.emissions["equip"].values:
                    if self.stands.loc[self.data.loc[index, "stand"], "terminal"] != "TERMINAL AVIACIÓN GENERAL":
                        ocupado.iloc[int(solution_mod[ind][0])] = 1
                        if stands_util.__contains__(solution_mod[ind][0]):
                            stands_util.remove(solution_mod[ind][0])
                        stands_util.append(solution_mod[ind][0])
                        ind += 1
                elif self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado" \
                      and self.stands.loc[self.data.loc[index, "stand"], "terminal"] != "TERMINAL AVIACIÓN GENERAL":
                    ocupado.iloc[int(solution_mod[ind][0])] = 1
                    if stands_util.__contains__(solution_mod[ind][0]):
                        stands_util.remove(solution_mod[ind][0])
                    stands_util.append(solution_mod[ind][0])
                    ind += 1
                elif self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] == "Privado" \
                      and listapr.__contains__(str(self.stands.loc[self.data.loc[index, "stand"], "terminal"])):
                    ocupado.iloc[int(solution_mod[ind][0])] = 1
                    if stands_util.__contains__(solution_mod[ind][0]):
                        stands_util.remove(solution_mod[ind][0])
                    stands_util.append(solution_mod[ind][0])
                    ind += 1
            elif self.data.loc[index, "cod"] == 1 and self.data.loc[index, "flight_type"] == 1 and \
                    (ocupado.iloc[int(solution_mod[ind][0])] == 0)[0] and \
                    (((self.data.loc[index, "equip"] not in self.emissions["equip"].values or self.emissions.loc[
                        self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado") and self.stands.loc[self.data.loc[index, "stand"], "terminal"] != "TERMINAL AVIACIÓN GENERAL") or (self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[
                         0] == "Privado" and listapr.__contains__(str(self.stands.loc[self.data.loc[index, "stand"], "terminal"])))):
                ocupado.iloc[int(solution_mod[ind][0])] = 1
                if stands_util.__contains__(solution_mod[ind][0]):
                    stands_util.remove(solution_mod[ind][0])
                stands_util.append(solution_mod[ind][0])
                ind += 1
            elif self.data.loc[index, "cod"] == 0 and index > 0:
                ocupado.loc[self.data.loc[
                    np.where(self.data.loc[0:(index - 1), "aircraft_id"] == self.data.loc[index, "aircraft_id"])[0][
                        -1], "stand"]] = 0
                if stands_util.__contains__(solution_mod[ind][0]):
                    stands_util.remove(solution_mod[ind][0])
                stands_util.append(solution_mod[ind][0])
            else:  # encoded failed
                if index > 0:
                    if ocupado.sum()[0] < self.stands.__len__():
                        # si no se han llenado todos los stands
                        distance = abs(pd.DataFrame(range(0, self.stands.__len__()))[ocupado == 0] - solution_mod[ind][0])
                        # si es comercial se le asigna el que más cerca esté que no sea de la terminal T.A.G.
                        if self.data.loc[index, "equip"] not in self.emissions["equip"].values or self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado":
                            distance[self.stands["terminal"] == "TERMINAL AVIACIÓN GENERAL"] = 1e6
                        else:
                            distance[self.stands["terminal"] == "T-123"] = 1e6
                            distance[self.stands["terminal"] == "T-4"] = 1e6
                            distance[self.stands["terminal"] == "T-4 REMOTO"] = 1e6
                            distance[self.stands["terminal"] == "T-4S"] = 1e6
                            distance[self.stands["terminal"] == "T-4S REMOTO"] = 1e6
                        solution_mod[ind][0] = np.where(distance == distance.min())[0][0]
                        ocupado.iloc[int(solution_mod[ind][0])] = 1
                        if stands_util.__contains__(solution_mod[ind][0]):
                            stands_util.remove(solution_mod[ind][0])
                        stands_util.append(solution_mod[ind][0])
                        ind += 1
                    else:  # si llega el caso le asigno el último que se usó
                        pila = deque()
                        corregido = False
                        # si es comercial se le asigna el stand que no sea de la terminal TAG y que se utilizó hace más tiempo
                        if self.data.loc[index, "equip"] not in self.emissions["equip"].values or self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado":
                            for a in range(stands_util.__len__()):
                                checkstand = stands_util.popleft()
                                if self.stands.loc[checkstand, "terminal"] != "TERMINAL AVIACIÓN GENERAL":
                                    solution_mod[ind][0] = checkstand
                                    corregido = True
                                else:
                                    pila.append(checkstand)
                                if corregido:
                                    break
                        else:
                            for a in range(stands_util.__len__()):
                                checkstand = stands_util.popleft()
                                if listapr.__contains__(str(self.stands.loc[checkstand, "terminal"])):
                                    solution_mod[ind][0] = checkstand
                                    corregido = True
                                else:
                                    pila.append(checkstand)
                                if corregido:
                                    break
                        for a in range(pila.__len__()):
                            stands_util.appendleft(pila.pop())
                        stands_util.append(solution_mod[ind][0])
                        ind += 1
        return np.array(solution_mod).reshape(-1).reshape(-1)

    def repair_solution(self, solution):
        return self.check_bounds(solution)

@jit(nopython=True)
def rosenbrock(solution):
    term1 = solution[1:] - solution[:-1]**2
    term2 = 1 - solution[:-1]
    result = 100*term1**2 + term2**2
    return result.sum()

@jit(nopython=True)
def rastrigin(solution, A=10):
    return (A * len(solution) + (solution**2 - A*np.cos(2*np.pi*solution)).sum())
