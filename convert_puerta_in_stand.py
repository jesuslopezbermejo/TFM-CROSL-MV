

import numpy as np
import pandas as pd
import copy

def existe_puerta_con_tiempos(puerta, tiempos_salidas):
    sol = False
    for item in tiempos_salidas.iterrows():
        puerta_fila = item[1]["Puerta"]
        if puerta in puerta_fila:
            sol = True
        elif '-' in puerta_fila:
            letra_puerta = puerta_fila.split(' ')[1][0]
            if puerta_fila[0] == letra_puerta:
                puertas = puerta_fila.split('-')
                digitos = ["".join(filter(str.isdigit, p)) for p in puertas]
                if int(puerta_fila[1:]) >= int(digitos[0]) and int(puerta_fila[1:]) <= int(digitos[1] + 1):
                    sol = True
        elif ',' in puerta_fila:
            puertas = puerta_fila.split(',')
            letra_puerta = puertas[0].split(" ")[1][0]
            if puerta_fila[0] == letra_puerta:
                primera_puerta = puertas[0].split(" ")[1]
                if puerta == primera_puerta:
                    sol = True
                else:
                    for p in puertas[1:]:
                        p = p.strip()
                        if 'y' in p:
                            puertas = p.split('y')
                            if puerta == puertas[0] or puerta == puertas[1]:
                                sol = True
                        else:
                            if puerta == p:
                                sol = True
        if sol == True:
            return sol   
    return sol            
                
                
                
def convert_puerta_in_stand(stand, puertas, tiempos_salidas):
    np_stand = np.array([stand["Latitude"], stand["Longitude"]])
    np_puertas = np.array(puertas[["lat", "lon"]])
    puertas_copy = copy.deepcopy(puertas)
    existe_puerta = False
    posible_puerta = None
    while(not existe_puerta):
        distances = np.abs(np_puertas - np_stand)
        distances = np.sum(distances, axis=1)
        min_distance = np.min(distances)
        index = np.where(distances == min_distance)
        indice = index[0][0]
        res = puertas_copy.iloc[indice]
        posible_puerta = res["puerta"]
        if existe_puerta_con_tiempos(posible_puerta, tiempos_salidas):
            existe_puerta = True
        else:
            np_puertas = np.delete(np_puertas, indice, axis=0)
            puertas_copy = puertas_copy.drop(indice, axis=0)
            puertas_copy.index = range(len(puertas_copy))
    return posible_puerta

    
def format_stands(df_stands):
    copy_stands = df_stands.copy()
    copy_stands["Latitude"] = df_stands["Latitude"].apply(lambda x: x / (10**(len(str(x))- 2)))
    copy_stands["Longitude"] = df_stands["Longitude"].apply(lambda x: x / (10**(len(str(x))- 2))) # 1 pero son 2 por el signo negativo
    return copy_stands

def fill_time(conversion_df):

    
    lista_puertas_tiempos_salidas = []
    for index, row in tiempos_salidas.iterrows():
        puerta = row['Puerta']
        tiempo = row['tiempo']
        if '-' in puerta:
            letra_puerta = puerta.split(' ')[1][0]
            puertas = puerta.split('-')
            digitos = ["".join(filter(str.isdigit, p)) for p in puertas]
            for d in range(int(digitos[0]), int(digitos[1])+1):
                lista_puertas_tiempos_salidas.append(('Puerta ' + letra_puerta + str(d), tiempo))
        elif ',' in puerta:
            puertas = puerta.split(',')
            lista_puertas_tiempos_salidas.append((puertas[0], tiempo))
            for p in puertas[1:]:
                if 'y' in p:
                    puertas = p.split('y')
                    lista_puertas_tiempos_salidas.append(('Puerta ' + puertas[0], tiempo))
                    lista_puertas_tiempos_salidas.append(('Puerta ' + puertas[1], tiempo))
                else:
                    lista_puertas_tiempos_salidas.append(('Puerta ' + p, tiempo))
        elif 'y' in puerta:
            puertas = puerta.split('y')
            lista_puertas_tiempos_salidas.append((puertas[0], tiempo))
            lista_puertas_tiempos_salidas.append(('Puerta ' + puertas[1], tiempo))
        else:
            lista_puertas_tiempos_salidas.append((puerta, tiempo))
    puertas_tiempo_final_salidas = []
    for i in range(len(lista_puertas_tiempos_salidas)):
        
        puerta_aux = lista_puertas_tiempos_salidas[i][0]
        puerta_aux = ' '.join(puerta_aux.split())
        puerta_aux = puerta_aux.replace('Puertas', 'Puerta')
        puerta_aux = puerta_aux.replace('Puerta', '')
        puerta_aux = puerta_aux.strip()
        puertas_tiempo_final_salidas.append((puerta_aux, lista_puertas_tiempos_salidas[i][1]))
    
    
    lista_puertas_tiempos_llegadas = []
    for index, row in tiempos_llegadas.iterrows():
        puerta = row['Origen']
        tiempo = row['Tiempo']
        if '-' in puerta:
            letra_puerta = puerta.split(' ')[1][0]
            puertas = puerta.split('-')
            digitos = ["".join(filter(str.isdigit, p)) for p in puertas]
            for d in range(int(digitos[0]), int(digitos[1])+1):
                lista_puertas_tiempos_llegadas.append(('Puerta ' + letra_puerta + str(d), tiempo))
        elif ',' in puerta:
            puertas = puerta.split(',')
            lista_puertas_tiempos_llegadas.append((puertas[0], tiempo))
            for p in puertas[1:]:
                if 'y' in p:
                    puertas = p.split('y')
                    lista_puertas_tiempos_llegadas.append(('Puerta ' + puertas[0], tiempo))
                    lista_puertas_tiempos_llegadas.append(('Puerta ' + puertas[1], tiempo))
                else:
                    lista_puertas_tiempos_llegadas.append(('Puerta ' + p, tiempo))
        elif 'y' in puerta:
            puertas = puerta.split('y')
            lista_puertas_tiempos_llegadas.append((puertas[0], tiempo))
            lista_puertas_tiempos_llegadas.append(('Puerta ' + puertas[1], tiempo))
        else:
            lista_puertas_tiempos_llegadas.append((puerta, tiempo))
            
            
            
    puertas_tiempo_final_llegadas = []
    for i in range(len(lista_puertas_tiempos_llegadas)):
        
        puerta_aux = lista_puertas_tiempos_llegadas[i][0]
        puerta_aux = ' '.join(puerta_aux.split())
        puerta_aux = puerta_aux.replace('Puertas', 'Puerta')
        puerta_aux = puerta_aux.replace('Puerta', '')
        puerta_aux = puerta_aux.strip()
        puertas_tiempo_final_llegadas.append((puerta_aux, lista_puertas_tiempos_llegadas[i][1]))
      
    puertas_tiempo_final_salidas.sort(key=lambda x: x[0])  
    puertas_tiempo_final_llegadas.sort(key=lambda x: x[0])
    
    lista_puerta_tiempos_unidas = []
    for tiempo_salida, tiempo_llegadas in zip(puertas_tiempo_final_salidas, puertas_tiempo_final_llegadas):
        lista_puerta_tiempos_unidas.append((tiempo_salida[0], tiempo_salida[1], tiempo_llegadas[1]))
    
    dict_salidas = {}
    dict_llegadas = {}
    for index, row in conversion_df.iterrows():
        target_tuple = next((t for t in lista_puerta_tiempos_unidas if t[0] == row["Puerta"]), None)
        if target_tuple:
            dict_salidas[row["Stand"]] = target_tuple[1]
            dict_llegadas[row["Stand"]] = target_tuple[2]
        else:
            stand = row["Stand"]
            puerta = row["Puerta"]
            print(f"El stand {stand} con la puerta asignada {puerta} no tiene tiempo asignado ya que la puerta no se ha encontrado en el excel de tiempos")
    return dict_salidas, dict_llegadas

    
    
def extract_times(df_stands, df_puertas, tiempos_salidas):
    
    df_stands = df_stands[["Stand", "Latitude", "Longitude"]]
    
    df_stands = format_stands(df_stands)
    
    conversion = {}
    for stand in df_stands.iterrows():
        stand_actual = stand[1]["Stand"]
        if stand_actual == "569":
            pass
        conversion[stand_actual] = convert_puerta_in_stand(stand[1], df_puertas, tiempos_salidas)
        
    
    conversion_df = pd.DataFrame(conversion.items(), columns=["Stand", "Puerta"])
    conversion_df.to_excel("conversion_puerta_stand.xlsx", index=False)
    return fill_time(conversion_df)

if __name__ == "__main__":
    
    tiempos_salidas = pd.read_excel("tiempos_ida.xlsx")
    tiempos_llegadas = pd.read_excel("tiempos_vuelta_a_salas_comunes.xlsx")
    df_stands = pd.read_excel("standsMAD.xlsx")
    df_puertas = pd.read_excel("puertasCoords.xlsx")

    dict_salidas, dict_llegadas = extract_times(df_stands, df_puertas, tiempos_salidas)
    dict_salidas_df = pd.DataFrame(dict_salidas.items(), columns=["Stand", "Tiempo_Salida"])
    dict_llegadas_df = pd.DataFrame(dict_llegadas.items(), columns=["Stand", "Tiempo_Llegada"])

    dict_salidas_df.to_excel("stand_tiempo_salidas.xlsx", index=False)
    dict_llegadas_df.to_excel("stand_tiempo_llegadas.xlsx", index=False)
    