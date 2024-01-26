import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from time import time
from sklearn.preprocessing import LabelEncoder
import numpy as np

from ant_clustering import AntCluster


def transform(df, column):
  labelencoder = LabelEncoder()
  df[column] = labelencoder.fit_transform(df[column])


@st.cache_data
def return_tnse(scaled_df):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(scaled_df)
    transformed_tsne = pd.DataFrame(tsne_results)
    transformed_tsne.columns = ['x', 'y']
    return transformed_tsne


st.title("AntennaClusteringCSV")
st.image('ants.jpg')
dataset = st.file_uploader("Choose a dataset", type='csv')

def highlight_columns(s, columns_to_highlight, color='yellow'):
    """
    Highlight specified columns in a DataFrame with a custom color.
    """
    if s.name in columns_to_highlight:
        return ['background-color: {}'.format(color)] * len(s)
    else:
        return [''] * len(s)

if dataset is not None:
    np.random.seed(100)
    random.seed(100)
    df = pd.read_csv(dataset)
    data_size = 200 if len(df) > 200 else len(df)
    df = df.sample(data_size) 
    df = df.dropna()
    original_df = df.copy(deep=True)
    st.markdown(f"<h2 style='text-align: center;'>Dataset {dataset.name}</h2>", unsafe_allow_html=True)
    st.subheader(f"Number of columns: {df.shape[1]}")
    st.subheader(f"Number of rows: {df.shape[0]}")
    st.markdown(f"<h3 style='text-align: center;'>First five rows of dataset</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(5))
    st.markdown(f"<h3 style='text-align: center;'>Numerical columns description</h3>", unsafe_allow_html=True)
    st.dataframe(df.describe())
    st.markdown(f"<h3 style='text-align: center;'>Null values Heatmap</h3>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    st.pyplot(plt)
    if [column for column in df.columns if df[column].dtype == 'object' and len(df[column].unique()) < 50] != []:
        category = st.selectbox('Categorical column', [column for column in df.columns if df[column].dtype == 'object' and len(df[column].unique()) < 50])
        a = pd.DataFrame(df[category].value_counts())
        a = a.reset_index()
        a.columns = [category, 'value']
        fig = px.pie(df, values=a['value'], names=a[category],
                    title=f'{category}',
                    height=300, width=200)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig, use_container_width=True)
    start = time()
    for column in df.columns:
        if df[column].dtype == 'object':
            transform(df, column)
    end = time()
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)
    on = st.toggle('Reduce to 2 dimensions for visualization', value=True)
    random_columns = [df.columns[random.randint(0, len(df.columns)-1)] for i in range(2)]
    num_sample = 200 if scaled_df.shape[0] > 200 else scaled_df.shape[0]
    transformed_tsne = return_tnse(scaled_df)
    if on:
        fig = px.scatter(transformed_tsne, x="x", y="y")
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig, use_container_width=True)
        #st.scatter_chart(x=tsne_results[:, 0], y=tsne_results[:, 1])
    else:
        options = st.multiselect(
            'Which 2 columns to visualize',
            [column for column in df.columns],
            random_columns,
            max_selections=2,
        )
        if len(options) == 2:
            fig = px.scatter(df, x=options[0], y=options[1])
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
            st.plotly_chart(fig, use_container_width=True)
    clusters = st.slider('How many clusters?', 2, 20, 3)
    ants = st.slider('How many ants?', 1, 20, 10)
    iterations = st.slider('How many iterations?', 16, 1000, 100)
    threads = st.slider('How many Threads?', 1, 16, 1)
    if st.button(f"RUN :sunglasses:", type="primary"):
        ant_cluster = AntCluster(ants, clusters, transformed_tsne.to_numpy(), max_iter=iterations, threads=threads)
        start = time()
        best_solution, best_cost = ant_cluster.run(transformed_tsne)
        end = time()
        colors = [
            "#FF5733",  # Red Orange
            "#33FF57",  # Neon Green
            "#3357FF",  # Blue
            "#F033FF",  # Magenta
            "#33FFF5",  # Cyan
            "#F5FF33",  # Yellow
            "#FF3380",  # Pink
            "#80FF33",  # Light Green
            "#FF9633",  # Orange
            "#33FF80",  # Mint
            "#7F33FF",  # Purple
            "#FF337F",  # Rose
            "#33A2FF",  # Sky Blue
            "#A233FF",  # Violet
            "#33FFA2",  # Aquamarine
            "#FFA233",  # Amber
            "#8CFF33",  # Lime Green
            "#FF338C",  # Hot Pink
            "#337FFF",  # Azure
            "#FF5733"   # Crimson Red
        ]
        c = []
        for i in range(len(scaled_df)):
            for j in range(len(colors)):
                if best_solution[i] == j:
                    c.append(f"class {best_solution[i]}")
                    break
        transformed_tsne['color'] = c
        color_map = {f"class {i}": colors[i] for i in range(clusters)}
        print(color_map)
        fig = px.scatter(transformed_tsne, x="x", y="y", color="color", color_discrete_map=color_map)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h3 style='text-align: center;'>The Ants have finished!</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>The algorithm took {int(end-start)} seconds!</h3>", unsafe_allow_html=True)
        #st.scatter_chart(transformed_tsne.sample(n=num_sample), x='x', y='y', width=50, color=tuple(colors))
        original_df['class'] = transformed_tsne['color']
        styled_df = original_df.style.apply(lambda x: highlight_columns(x, 'class', color='gray'), axis=0)
        st.dataframe(styled_df)