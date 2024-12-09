#Cargamos librerías necesarias
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import pyodbc # librería para SQL server
import random
import streamlit as st
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
import time

## Hacer que los números aleatorios se mantengan  
np.random.seed(0)
import streamlit as st

# Título de la aplicación
st.title("Simulación de gestión de inventarios")

# Controles para definir los datos
with st.form(key="config_form"):
    repuesto_id = st.text_input("Identificador del Repuesto", value="JD_RE504836")
    precio_fob_unitario = st.number_input("Precio FOB Unitario ($)", value=16 , step=1)
    Costo_pedido_k = st.number_input("Cantidad de Pedidos Anuales", value=4166, step=1)
    costo_mantenimiento_anual_h = st.number_input("Costo Anual de Mantenimiento", value=1.6, step=0.1)

    # Botón para confirmar
    submitted = st.form_submit_button("Confirmar Datos")

# Procesar datos solo si se confirman
if submitted:
    st.success("Datos confirmados")
    
    # Mostrar un resumen de los datos confirmados
    st.write("### Resumen de Datos")
    st.write(f"**Repuesto ID:** {repuesto_id}")
    st.write(f"**Precio FOB Unitario:** ${precio_fob_unitario}")
    st.write(f"**Cantidad de Pedidos Anuales:** {Costo_pedido_k}")
    st.write(f"**Costo de Mantenimiento Anual:** {costo_mantenimiento_anual_h}")
    
    # Llamar a la lógica principal de la aplicación con los datos ingresados
    # Aquí podrías llamar a tus funciones principales usando estas variables
    # ejemplo: calcular_costo_total(precio_fob_unitario, cantidad_pedidos_anuales, ...)


    conexion = pyodbc.connect(
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=172.16.0.161;'
    'DATABASE=nombre_bd;'
    'UID=consultor;'
    'PWD=nibol123;'
    'TrustServerCertificate=yes;')
    Repuesto = repuesto_id
    query_LeadTime = """
    select *,DATEDIFF(day,fechafactura,fechaingreso) diferencia
    from 
    (select refOrdenCompra,proveedor,cast(factFecha as date) FechaFactura from eikp) OC
    Inner Join
    (select pedidoH,cast(max(fechaconta) as date) FechaIngreso,sum(importeML)/6.96 Importe_USD from mseg where claseMovimiento=101 group by pedidoH) I on OC.refOrdenCompra=I.pedidoH
    where proveedor=300005 and fechafactura<>'1900-01-01' and Importe_USD>50000 and year(fechaingreso)>=2021 order by FechaFactura """

    query_Demanda=""" 
    -- Query para demanda por ìtem
    select f.FechaFactura,isnull(V.DemandaDiaria,0) Sales from 
    (select 
        distinct fechafactura 
    from
        v_ZSDVentasBase 
    where 
        FechaFactura between '2023-03-01' and '2024-03-31') F
    left join
    (SELECT 
        FechaFactura,isnull(sum(Cantidadfacturada),0) DemandaDiaria
    FROM 
        v_ZSDVentasBase 
    WHERE 
        material =? and FechaFactura between '2023-03-01' and '2024-03-31' 
        and Clasedefactura not like '%N%%' and anulado<>'X'
    group by 
        FechaFactura) V on V.FechaFactura=F.FechaFactura
    """

    query_Stock=""" 
    -- Query para Stock
    select codigo,sum(libre) Stock from mard 
    where codigo=?
    group by codigo 
    """

    df_LeadTime = pd.read_sql_query(query_LeadTime, conexion)
    df=pd.read_sql_query(query_Demanda, conexion, params=(Repuesto,))
    df_stock=pd.read_sql_query(query_Stock, conexion, params=(Repuesto,)) ## , params=(Repuesto,)
    conexion.close()

    ############################## Grafico de Demanda histórica #####################

    st.markdown("#### Información de Demanda")
    # Sección replegable
    with st.expander("Ver/ocultar tabla de datos"):
        st.dataframe(df, use_container_width=True)

    st.text(df.describe())

    # Convertir la columna 'FechaFactura' a tipo fecha
    df['FechaFactura'] = pd.to_datetime(df['FechaFactura'])

    # Extraer el mes y calcular la suma acumulada
    ##df['Mes'] = df['FechaFactura'].dt.strftime('%Y-%m')
    df_agrupada = df.groupby('FechaFactura', as_index=False)['Sales'].sum()

    # Gráfico interactivo con Plotly
    fig = px.line(
        df_agrupada,
        x='FechaFactura',
        y='Sales',
        labels={'FechaFactura': 'FechaFactura', 'Sales': 'Ventas diarias ($)'},
        markers=True,
    )

    # Personalizar el diseño del gráfico
    fig.update_traces(line_color='green', line_width=2)
    fig.update_layout(
        
        xaxis=dict(title='FechaFactura', showgrid=False),
        yaxis=dict(title='Ventas diarias ($)', showgrid=True),
        plot_bgcolor='white',
    )

    # Mostrar gráfico en Streamlit
    st.markdown("#### Gráfico Demanda")
    st.plotly_chart(fig, use_container_width=True)


    #################### Kernell Distribution  #####################
    # Calcular el ancho de banda de la distribución KDE
    # ahora estimamos la pmf de la demanda
    bandwidth = df['Sales'].std() / (df['Sales'].count()**(1/5)) # ancho de banda de 
    lower = 0
    upper = np.ceil(df['Sales'].max() + 1 * bandwidth)
    x = np.arange(lower,upper)
    kde_dist = gaussian_kde(df['Sales'],bw_method='scott')
    pmf = kde_dist.pdf(x)
    pmf=pmf/sum(pmf)

    # Crear gráfico con Plotly
    fig = go.Figure()

    # Trazar la estimación de la PMF
    fig.add_trace(go.Scatter(
        x=x, y=pmf, mode='lines', name='PMF Estimada', 
        line=dict(color='green', width=2), 
        fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.3)'
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        xaxis_title="Demanda",
        yaxis_title="Probabilidad",
        plot_bgcolor='white',  # Fondo blanco para un diseño limpio
        template='plotly_white',  # Tema limpio y profesional
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=40)
    )

    # Mostrar el gráfico en Streamlit
    st.markdown("#### Distribución de Kernell")
    st.plotly_chart(fig, use_container_width=True)


    ##########################################################################################


    # generamos la fucnión para obtener los atributos de la demanda PMF
    def attributes(pmf,x):
        mu = sum(pmf*x)
        std = np.sqrt(sum(x**2*pmf) - sum(pmf*x)**2)
        return mu, std

    d_mu, d_std = attributes(pmf,x)


    time=365*10# 10 años de simulación
    # Definimos una muestra de la demanda en función a la distribución customizada
    d = np.random.choice(x, size=time, p=pmf)

    # Definición de Lead Time
    L_x = np.array([35,65,85])
    L_pmf = np.array([0.6,0.3,0.1])
    L_mu,L_std = attributes(L_pmf,L_x)
    L_median = 43
    L_max = 262
    #L_mu, L_std = 30,13
    #L_median = 30
    #L_max = 120


    # Suponiendo que df_LeadTime es tu DataFrame y 'diferencia' es la columna de interés
    # df_LeadTime = pd.read_csv("tus_datos.csv") # carga tus datos si es necesario

    # Crear el gráfico
    plt.figure(figsize=(8, 6))
    plt.hist(df_LeadTime['diferencia'], bins=20, range=(0, 50), color='skyblue', edgecolor='black')
    plt.xlabel("Diferencia", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)


    describe_Lead_time=df_LeadTime.diferencia.describe()
    # Mostrar el gráfico en Streamlit
    st.markdown("#### Distribución del lead Time")
    st.pyplot(plt)



    ##############################

    def simulation_plot_Inventario(R, Ss, d_mu, L_mu, L_max, L_median, L_x, L_pmf, h, b, kvar, time, d, df_stock):
        S = round(d_mu * (R + L_mu) + round(Ss))  # Stock ideal
        np.random.seed(0)  # Para números aleatorios consistentes
        hand = np.zeros(time, dtype=int)
        transit = np.zeros((time, L_max + 1), dtype=int)
        unit_shorts = np.zeros(time, dtype=int)
        stockout_period = np.full(time, False, dtype=bool)
        stockout_cycle = []
        
        hand[0] = df_stock.Stock[0]
        transit[1, L_median] = d[0]
        p = np.zeros(time)  # Stock físico
        p[0] = df_stock.Stock[0]
        c_k = 0  # Costo de transacción variable
        c_h = h * p[0]  # Costo de mantenimiento
        c_b = 0  # Costo de penalidad por venta perdida
        cant_pedidos = 0

        for t in range(1, time):
            if transit[t - 1, 0] > 0:
                stockout_cycle.append(stockout_period[t - 1])
            unit_shorts[t] = max(0, d[t] - max(0, hand[t - 1] + transit[t - 1, 0]))
            hand[t] = hand[t - 1] - d[t] + transit[t - 1, 0]
            stockout_period[t] = hand[t] < 0
            transit[t, :-1] = transit[t - 1, 1:]

            if t % R == 0:
                actual_L = np.random.choice(L_x, 1, p=L_pmf)
                net = hand[t] + transit[t].sum()
                transit[t, actual_L] = S - net
                c_k += kvar
                cant_pedidos += 1
            if hand[t] > 0:
                p[t] = (hand[t - 1] + transit[t - 1, 0] + hand[t]) / 2
            else:
                p[t] = max(hand[t - 1] + transit[t - 1, 0], 0)**2 / max(d[t], 1) / 2
            c_h += h * p[t]
            c_b += b * max(0, -hand[t])  # Backlog cost times the total backlog

        SL_alpha = 1 - (sum(stockout_cycle) / len(stockout_cycle))
        fill_rate = 1 - unit_shorts.sum() / sum(d)
        cost = (c_h + c_b + c_k) / time

        plt.figure(figsize=(12, 6))  # Aumentar tamaño para claridad
        plt.plot(p, label='Inventario', color='blue', linewidth=2)  # Línea de inventario
        plt.axhline(S, color='green', linestyle='--', linewidth=2, label=f'Stock Ideal (S = {S})')  # Línea de stock ideal
        plt.axhline(Ss, color='red', linestyle='--', linewidth=2, label=f'Stock de Seguridad (Ss = {Ss})')  # Línea de stock de seguridad

        # Añadir etiquetas, título y leyenda
        plt.xlabel('Tiempo (días)', fontsize=12)
        plt.ylabel('Cantidad de Inventario', fontsize=12)
        plt.title('Simulación del Inventario - Modelo Propuesto', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Cuadrícula más sutil
        plt.tight_layout()

        # Crear un diccionario con los datos
        resultados = {
            "Métrica": ["R", "Ss", "Stock Ideal", "Costo Diario Total", "Nivel de Servicio", "Cantidad de Pedidos"],
            "Valor": [
                R, 
                Ss, 
                S, 
                f"{cost:.2f}", 
                f"{SL_alpha:.2%}", 
                cant_pedidos
            ]
        }

        # Convertirlo a un DataFrame
        df_resultados = pd.DataFrame(resultados)
        st.markdown("### Simulación del modelo propuesto")
        # Mostrar como tabla en Streamlit
        st.dataframe(df_resultados.style.hide(axis="index"))
        st.pyplot(plt)

    # Hacemos la función de simulación (R,Ss) Básica
    def simulation_Nibol(R, Ss, d_mu, L_mu, L_max, L_median, L_x, L_pmf, h, b, kvar, time, d, df_stock):
        S = round((d_mu+(d_std/(len(d)**1/2))*2)*5*30) # Stock ideal
        np.random.seed(0)  # Para números aleatorios consistentes
        hand = np.zeros(time, dtype=int)
        transit = np.zeros((time, L_max + 1), dtype=int)
        unit_shorts = np.zeros(time, dtype=int)
        stockout_period = np.full(time, False, dtype=bool)
        stockout_cycle = []
        
        hand[0] = df_stock.Stock[0]
        transit[1, L_median] = d[0]
        p = np.zeros(time)  # Stock físico
        p[0] = df_stock.Stock[0]
        c_k = 0  # Costo de transacción variable
        c_h = h * p[0]  # Costo de mantenimiento
        c_b = 0  # Costo de penalidad por venta perdida
        cant_pedidos = 0

        for t in range(1, time):
            if transit[t - 1, 0] > 0:
                stockout_cycle.append(stockout_period[t - 1])
            unit_shorts[t] = max(0, d[t] - max(0, hand[t - 1] + transit[t - 1, 0]))
            hand[t] = hand[t - 1] - d[t] + transit[t - 1, 0]
            stockout_period[t] = hand[t] < 0
            transit[t, :-1] = transit[t - 1, 1:]

            if t % R == 0:
                actual_L = np.random.choice(L_x, 1, p=L_pmf)
                net = hand[t] + transit[t].sum()
                transit[t, actual_L] = S - net
                c_k += kvar
                cant_pedidos += 1
            if hand[t] > 0:
                p[t] = (hand[t - 1] + transit[t - 1, 0] + hand[t]) / 2
            else:
                p[t] = max(hand[t - 1] + transit[t - 1, 0], 0)**2 / max(d[t], 1) / 2
            c_h += h * p[t]
            c_b += b * max(0, -hand[t])  # Backlog cost times the total backlog

        SL_alpha = 1 - (sum(stockout_cycle) / len(stockout_cycle))
        fill_rate = 1 - unit_shorts.sum() / sum(d)
        cost = (c_h + c_b + c_k) / time

        plt.figure(figsize=(12, 6))  # Aumentar tamaño para claridad
        plt.plot(p, label='Inventario', color='orange', linewidth=2)  # Línea de inventario
        plt.axhline(S, color='green', linestyle='--', linewidth=2, label=f'Stock Ideal (S = {S})')  # Línea de stock ideal
        plt.axhline(Ss, color='red', linestyle='--', linewidth=2, label=f'Stock de Seguridad (Ss = {Ss})')  # Línea de stock de seguridad

        # Añadir etiquetas, título y leyenda
        plt.xlabel('Tiempo (días)', fontsize=12)
        plt.ylabel('Cantidad de Inventario', fontsize=12)
        plt.title('Simulación del Inventario - Modelo AGROBOL', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Cuadrícula más sutil
        plt.tight_layout()

        # Crear un diccionario con los datos
        resultados = {
            "Métrica": ["R", "Ss", "Stock Ideal", "Costo Diario Total", "Nivel de Servicio", "Cantidad de Pedidos"],
            "Valor": [
                R, 
                Ss, 
                S, 
                f"{cost:.2f}", 
                f"{SL_alpha:.2%}", 
                cant_pedidos
            ]
        }

        # Convertirlo a un DataFrame
        df_resultados = pd.DataFrame(resultados)
        st.markdown("### Simulación del modelo propuesto")
        # Mostrar como tabla en Streamlit
        st.dataframe(df_resultados.style.hide(axis="index"))
        st.pyplot(plt)

    # Continuamos creando la función de simulación que toma el invetario de seguidad como input
    #y retorna un costo promedio simulado por periodo
    P_FOB_unit=precio_fob_unitario
    k= Costo_pedido_k
    kvar = (k/P_FOB_unit) #Costo Variable de pedido
    h_anual=costo_mantenimiento_anual_h
    h = h_anual/365 # Costo de mantenimiento diario
    b = (P_FOB_unit*1.3/0.7/0.87)*0.3 # Costo de penalidad por venta perdida
    time = time # Número de pasos en la simulación
    Qopt=(((4*kvar*(d_mu*55*5))/h_anual))**(1/3)
    R=round((Qopt/(d_mu*55*5))*365,0) # Se multiplica por 365 para llevar R a días
    Q_eoq=((2*(d_mu*55*5)*k)/h_anual)**(1/2)
    R_eoq=round((Q_eoq/(d_mu*55*5))*365,0)

    # Calculamos el nivel de servicio óptimo
    alpha_opt = 1 - h*R/b
    x_std = np.sqrt((L_mu+R)*d_std**2 + L_std**2*d_mu**2)
    Ss = x_std*norm.ppf(alpha_opt)
    Ss = int(round(Ss))
    print(R,Ss,alpha_opt)


    # Crear un diccionario con los datos
    parametros = {
        "Parámetro": [
            "Costo unitario de FOB (P_FOB_unit)",
            "Costo de Pedido (k)",
            "Costo variable de pedido (kvar)",
            "Costo de mantenimiento diario (h)",
            "Costo de penalidad por venta perdida (b)",
            "Cantidad óptima de pedido (Qopt)",
            "Periodo fijo de pedido (R)",
            "Stock de seguridad (Ss)",
            "Nivel de servicio óptimo (alpha)"
        ],
        "Valor": [
            f"{P_FOB_unit:.2f} USD",
            f"{k:.2f} USD",
            f"{kvar:.2f} USD",
            f"{h:.5f} USD/día",
            f"{b:.2f} USD",
            f"{Qopt:.2f} unidades",
            f"{R} días",
            f"{Ss} unidades",
            f"{alpha_opt:.2f} %"
        ]
    }

    # Convertirlo a un DataFrame
    df_parametros = pd.DataFrame(parametros)

    # Mostrar como DataFrame interactivo en Streamlit
    st.markdown("### Parámetros del modelo")
    st.dataframe(df_parametros)

    simulation_plot_Inventario(R, Ss, d_mu, L_mu, L_max, L_median, L_x, L_pmf, h, b, kvar, time, d, df_stock)
    simulation_Nibol(30,Ss, d_mu, L_mu, L_max, L_median, L_x, L_pmf, h, b, kvar, time, d, df_stock)
