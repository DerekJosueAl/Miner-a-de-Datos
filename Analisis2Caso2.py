import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Análisis Exploratorio y Clustering de Fallas")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset_limpio.csv")
    columnas_irrelevantes = ['UDI', 'Product ID']
    df = df.drop(columns=[col for col in columnas_irrelevantes if col in df.columns])
    return df

df = cargar_datos()

# EDA: vista previa
st.subheader("Vista previa del dataset")
st.dataframe(df.head())

# EDA: histogramas
st.subheader("Distribuciones de variables numéricas")
columnas_numericas = df.select_dtypes(include='number').columns.tolist()
for col in columnas_numericas:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribución de {col}")
    st.pyplot(fig)

# EDA: boxplots por falla
if 'falla' in df.columns:
    st.subheader("Boxplots por clase de falla")
    for col in columnas_numericas:
        if col != 'falla':
            fig, ax = plt.subplots()
            sns.boxplot(x='falla', y=col, data=df, ax=ax)
            ax.set_title(f"{col} según falla")
            st.pyplot(fig)

# EDA: mapa de calor
st.subheader("Mapa de calor de correlaciones")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Clustering
st.subheader("Clustering con KMeans")
# Eliminar columnas no numéricas y la columna 'falla'
columnas_validas = df.select_dtypes(include='number').drop(columns=['falla'], errors='ignore')
X_scaled = StandardScaler().fit_transform(columnas_validas)

k = st.slider("Número de clusters", min_value=2, max_value=10, value=3)
modelo = KMeans(n_clusters=k, random_state=42)
df['cluster'] = modelo.fit_predict(X_scaled)

# Visualización de clusters
st.subheader("Distribución de Clusters")
fig2, ax2 = plt.subplots()
sns.countplot(x='cluster', data=df, palette='Set2', ax=ax2)
st.pyplot(fig2)

# Proporción de fallas por cluster
if 'falla' in df.columns:
    st.subheader("Proporción de fallas por cluster")
    falla_por_cluster = df.groupby('cluster')['falla'].mean().reset_index()
    st.dataframe(falla_por_cluster)

# Promedios por cluster
st.subheader("Promedios por cluster")
cluster_means = df.groupby('cluster')[columnas_validas.columns].mean()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap='viridis', ax=ax3)
st.pyplot(fig3)

# calculos de rpm 
frecuencias = df['rotational speed [rpm]'].value_counts()
repetidos = frecuencias[frecuencias > 1]
rpm_max_repetido = repetidos.index.max()

st.markdown(f"**Valor más alto que se repite:** `{rpm_max_repetido}` rpm")

# Crear columna de riesgo
df['rpm_riesgo'] = df['rotational speed [rpm]'] > 2200

# Filtrar registros con RPM superiores
rpm_superiores = df[df['rotational speed [rpm]'] > rpm_max_repetido]

st.write(f"Cantidad de registros con rpm mayores a {rpm_max_repetido}: **{len(rpm_superiores)}**")

# Mostrar primeras filas relevantes
st.dataframe(rpm_superiores[['rotational speed [rpm]', 'torque [nm]', 'tool wear [min]']].head())

# Filtrar y mostrar sumas de fallos
rpm_superiores = df[df['rotational speed [rpm]'] > 2200]
fallos = rpm_superiores[['rotational speed [rpm]', 'machine failure', 'twf', 'hdf', 'pwf', 'osf', 'rnf']]
st.subheader("⚙️ Suma de tipos de fallos (rpm > 2200)")
st.write(fallos.sum())

# Promedios
prom_torque = rpm_superiores['torque [nm]'].mean()
prom_tool = rpm_superiores['tool wear [min]'].mean()

col1, col2 = st.columns(2)
col1.metric("Promedio de Torque (Nm)", f"{prom_torque:.2f}")
col2.metric("Promedio de Desgaste (min)", f"{prom_tool:.2f}")

# ===== GRÁFICO =====
st.subheader("📈 Relación RPM vs Desgaste (rpm > 2200)")
fig, ax = plt.subplots()
ax.scatter(rpm_superiores['rotational speed [rpm]'], rpm_superiores['tool wear [min]'], c='red', alpha=0.7)
ax.set_title("RPM vs Tool Wear (rpm > 2200)")
ax.set_xlabel("Rotational Speed [rpm]")
ax.set_ylabel("Tool Wear [min]")
ax.grid(True)
st.pyplot(fig)