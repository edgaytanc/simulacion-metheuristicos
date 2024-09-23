import streamlit as st
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import numpy as np
import time

# Definir funciones para los algoritmos

def generar_datos_problema(num_parcelas, rango_caña, rango_tiempo, num_vehiculos, rango_capacidad_vehiculos, num_trabajadores, horas_trabajo):
    parcelas = []
    for i in range(1, num_parcelas + 1):
        caña = random.randint(*rango_caña)
        tiempo = round(random.uniform(*rango_tiempo), 2)
        parcelas.append({'Parcela': i, 'Caña (kg)': caña, 'Tiempo (horas)': tiempo})

    vehiculos = [random.randint(*rango_capacidad_vehiculos) for _ in range(num_vehiculos)]
    capacidad_total_trabajo = num_trabajadores * horas_trabajo

    return pd.DataFrame(parcelas), vehiculos, capacidad_total_trabajo

# Función para Algoritmo Genético (GA)
def ejecutar_ga(df_parcelas, vehiculos, capacidad_total_trabajo, population_size, num_generaciones, probabilidad_cruce, probabilidad_mutacion):
    # Inicialización de variables
    cantidades_caña = df_parcelas['Caña (kg)'].values
    tiempos_cosecha = df_parcelas['Tiempo (horas)'].values
    capacidad_vehiculo = min(vehiculos)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def crear_individuo():
        return creator.Individual([random.randint(0, 1) for _ in range(len(df_parcelas))])

    def evaluar_individuo(individuo):
        cantidad_caña_total = sum(individuo[i] * cantidades_caña[i] for i in range(len(df_parcelas)))
        tiempo_total = sum(individuo[i] * tiempos_cosecha[i] for i in range(len(df_parcelas)))

        penalizacion = 0
        if tiempo_total > capacidad_total_trabajo:
            penalizacion += (tiempo_total - capacidad_total_trabajo) * 100
        if cantidad_caña_total > capacidad_vehiculo:
            penalizacion += (cantidad_caña_total - capacidad_vehiculo) * 100

        return max(0, cantidad_caña_total - penalizacion),

    toolbox = base.Toolbox()
    toolbox.register("individual", crear_individuo)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    hall_of_fame = tools.HallOfFame(1)
    poblacion = toolbox.population(n=population_size)

    # Ejecutar el Algoritmo Genético
    poblacion, log = algorithms.eaSimple(poblacion, toolbox, cxpb=probabilidad_cruce, mutpb=probabilidad_mutacion, 
                                          ngen=num_generaciones, stats=stats, halloffame=hall_of_fame, verbose=False)

    mejor_individuo = tools.selBest(poblacion, k=1)[0]
    mejor_aptitud = evaluar_individuo(mejor_individuo)[0]

    historico_aptitud = log.select("avg")

    return mejor_aptitud, historico_aptitud


# Función para Recocido Simulado (SA)
def ejecutar_sa(df_parcelas, vehiculos, capacidad_total_trabajo, temperatura_inicial, alpha, iteraciones):
    cantidades_caña = df_parcelas['Caña (kg)'].values
    tiempos_cosecha = df_parcelas['Tiempo (horas)'].values
    capacidad_vehiculo = min(vehiculos)

    def evaluar_individuo_rc(individuo):
        cantidad_caña_total = sum(individuo[i] * cantidades_caña[i] for i in range(len(df_parcelas)))
        tiempo_total = sum(individuo[i] * tiempos_cosecha[i] for i in range(len(df_parcelas)))

        penalizacion = 0
        if tiempo_total > capacidad_total_trabajo:
            penalizacion += (tiempo_total - capacidad_total_trabajo) * 100
        if cantidad_caña_total > capacidad_vehiculo:
            penalizacion += (cantidad_caña_total - capacidad_vehiculo) * 100

        return max(0, cantidad_caña_total - penalizacion)

    def crear_individuo_inicial():
        return [random.randint(0, 1) for _ in range(len(df_parcelas))]

    def mutar_individuo(individuo):
        nuevo_individuo = individuo[:]
        idx = random.randint(0, len(df_parcelas) - 1)
        nuevo_individuo[idx] = 1 - nuevo_individuo[idx]
        return nuevo_individuo

    def enfriamiento(T, alpha):
        return T * alpha

    estado_actual = crear_individuo_inicial()
    aptitud_actual = evaluar_individuo_rc(estado_actual)

    mejor_estado = estado_actual
    mejor_aptitud = aptitud_actual

    historico_aptitud = [aptitud_actual]
    temperatura = temperatura_inicial

    for _ in range(iteraciones):
        nuevo_estado = mutar_individuo(estado_actual)
        aptitud_nueva = evaluar_individuo_rc(nuevo_estado)

        delta_aptitud = aptitud_nueva - aptitud_actual
        if delta_aptitud > 0 or random.uniform(0, 1) < math.exp(delta_aptitud / temperatura):
            estado_actual = nuevo_estado
            aptitud_actual = aptitud_nueva

        if aptitud_actual > mejor_aptitud:
            mejor_estado = estado_actual
            mejor_aptitud = aptitud_actual

        temperatura = enfriamiento(temperatura, alpha)
        historico_aptitud.append(mejor_aptitud)

    return mejor_aptitud, historico_aptitud


# Función para Colonia de Hormigas (ACO)
def ejecutar_aco(df_parcelas, vehiculos, capacidad_total_trabajo, num_hormigas, num_iteraciones, evaporacion):
    cantidades_caña = df_parcelas['Caña (kg)'].values
    tiempos_cosecha = df_parcelas['Tiempo (horas)'].values
    capacidad_vehiculo = min(vehiculos)

    def evaluar_individuo_ch(individuo):
        cantidad_caña_total = sum(individuo[i] * cantidades_caña[i] for i in range(len(df_parcelas)))
        tiempo_total = sum(individuo[i] * tiempos_cosecha[i] for i in range(len(df_parcelas)))

        penalizacion = 0
        if tiempo_total > capacidad_total_trabajo:
            penalizacion += (tiempo_total - capacidad_total_trabajo) * 10
        if cantidad_caña_total > capacidad_vehiculo:
            penalizacion += (cantidad_caña_total - capacidad_vehiculo) * 10

        return max(0, cantidad_caña_total - penalizacion)

    def seleccionar_parcela(feromonas, cantidades_caña, influencia_feromonas, influencia_visibilidad):
        probabilidades = []
        for i in range(len(df_parcelas)):
            prob = (feromonas[i] ** influencia_feromonas) * (cantidades_caña[i] ** influencia_visibilidad)
            probabilidades.append(prob)
        probabilidades = probabilidades / np.sum(probabilidades)
        return np.random.choice(range(len(df_parcelas)), p=probabilidades)

    def construir_solucion(feromonas):
        solucion = [0] * len(df_parcelas)
        for _ in range(len(df_parcelas)):
            parcela_seleccionada = seleccionar_parcela(feromonas, cantidades_caña, 1.0, 2.0)
            solucion[parcela_seleccionada] = 1
        return solucion

    feromonas = np.full(len(df_parcelas), 1.0)
    mejor_aptitud = 0
    historico_aptitud = []

    for _ in range(num_iteraciones):
        soluciones_hormigas = []
        aptitudes_hormigas = []

        for _ in range(num_hormigas):
            solucion = construir_solucion(feromonas)
            aptitud = evaluar_individuo_ch(solucion)
            soluciones_hormigas.append(solucion)
            aptitudes_hormigas.append(aptitud)

        aptitud_max = max(aptitudes_hormigas)
        if aptitud_max > mejor_aptitud:
            mejor_aptitud = aptitud_max

        feromonas *= (1 - evaporacion)
        for i in range(len(df_parcelas)):
            for solucion, aptitud in zip(soluciones_hormigas, aptitudes_hormigas):
                if solucion[i] == 1 and aptitud > 0:
                    feromonas[i] += aptitud / mejor_aptitud

        historico_aptitud.append(mejor_aptitud)

    return mejor_aptitud, historico_aptitud


# Aplicación en Streamlit
st.title("Simulación de Cosecha de Caña con Algoritmos Metaheurísticos")

# Parámetros de entrada
num_parcelas = st.sidebar.slider("Número de parcelas", 5, 20, 10)
num_vehiculos = st.sidebar.slider("Número de vehículos", 1, 5, 3)
num_trabajadores = st.sidebar.slider("Número de trabajadores", 1, 10, 5)
horas_trabajo = st.sidebar.slider("Horas de trabajo semanales", 20, 60, 40)

# Generar datos del problema
df_parcelas, vehiculos, capacidad_total_trabajo = generar_datos_problema(num_parcelas, (100, 1200), (1.0, 10.0), num_vehiculos, (1000, 2500), num_trabajadores, horas_trabajo)

# Mostrar los datos generados
st.subheader("Datos Generados")
st.write("Parcelas:")
st.dataframe(df_parcelas)
st.write("Capacidades de los vehículos:", vehiculos)
st.write("Capacidad total de trabajo (horas):", capacidad_total_trabajo)

# Selección del algoritmo
algoritmo_seleccionado = st.selectbox("Selecciona un algoritmo", ["Algoritmo Genético (GA)", "Recocido Simulado (SA)", "Colonia de Hormigas (ACO)"])

if algoritmo_seleccionado == "Algoritmo Genético (GA)":
    population_size = st.slider("Tamaño de la población", 10, 500, 100)
    num_generaciones = st.slider("Número de generaciones", 10, 100, 50)
    probabilidad_cruce = st.slider("Probabilidad de cruce", 0.5, 1.0, 0.8)
    probabilidad_mutacion = st.slider("Probabilidad de mutación", 0.01, 0.5, 0.3)

    if st.button("Ejecutar Algoritmo Genético"):
        mejor_aptitud, historico_aptitud_ga = ejecutar_ga(df_parcelas, vehiculos, capacidad_total_trabajo, population_size, num_generaciones, probabilidad_cruce, probabilidad_mutacion)
        st.write(f"Mejor aptitud encontrada: {mejor_aptitud}")
        
        # Graficar la evolución del fitness
        st.line_chart(historico_aptitud_ga)

elif algoritmo_seleccionado == "Recocido Simulado (SA)":
    temperatura_inicial = st.slider("Temperatura inicial", 100, 2000, 1000)
    alpha = st.slider("Tasa de enfriamiento (alpha)", 0.85, 0.99, 0.95)
    iteraciones = st.slider("Número de iteraciones", 10, 200, 100)

    if st.button("Ejecutar Recocido Simulado"):
        mejor_aptitud, historico_aptitud_sa = ejecutar_sa(df_parcelas, vehiculos, capacidad_total_trabajo, temperatura_inicial, alpha, iteraciones)
        st.write(f"Mejor aptitud encontrada: {mejor_aptitud}")
        
        # Graficar la evolución del fitness
        st.line_chart(historico_aptitud_sa)

elif algoritmo_seleccionado == "Colonia de Hormigas (ACO)":
    num_hormigas = st.slider("Número de hormigas", 10, 100, 50)
    num_iteraciones = st.slider("Número de iteraciones", 10, 200, 100)
    evaporacion = st.slider("Tasa de evaporación", 0.1, 0.9, 0.1)

    if st.button("Ejecutar Colonia de Hormigas"):
        mejor_aptitud, historico_aptitud_aco = ejecutar_aco(df_parcelas, vehiculos, capacidad_total_trabajo, num_hormigas, num_iteraciones, evaporacion)
        st.write(f"Mejor aptitud encontrada: {mejor_aptitud}")
        
        # Graficar la evolución del fitness
        st.line_chart(historico_aptitud_aco)

