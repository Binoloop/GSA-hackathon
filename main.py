import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset (ensure you have the reduced_df.csv file)
reduced_df = pd.read_csv('data/3d_embedding.csv')

# Step 1: Streamlit UI setup
st.title("Connecting the Dots: Enhanced Data Discovery and Trust")

# Create a sidebar for the chatbot UI
st.sidebar.header("Chatbot")

# Input field for user queries
user_query = st.sidebar.text_input("Ask me anything about the datasets:")

# Simple response logic for the chatbot
if user_query:
    # Here you can add logic to generate a response based on the user query
    # For demonstration purposes, we'll just echo the query
    st.sidebar.write(f"You asked: {user_query}")
    st.sidebar.write("I'm a simple chatbot. How can I help you?")

# Step 2: Create a dropdown to select clusters
clusters = reduced_df['cluster'].unique()
selected_cluster = st.selectbox("Select a Cluster", clusters)

# Step 3: Filter the dataset based on selected cluster
highlighted_data = reduced_df[reduced_df['cluster'] == selected_cluster]

# Step 4: Create the 3D scatter plot using Plotly
fig = px.scatter_3d(
    reduced_df,
    x='PCA1',
    y='PCA2',
    z='PCA3',
    color='cluster',
    title='3D Embedding of Text Clusters',
    labels={'PCA1': 'PCA 1', 'PCA2': 'PCA 2', 'PCA3': 'PCA 3'},
    hover_data={
        'title': True,          # Show the title on hover
        'name': True  # Show additional info on hover
    },
    color_continuous_scale=px.colors.sequential.Viridis
)

# Highlight the points in the selected cluster
fig.add_trace(px.scatter_3d(
    highlighted_data,
    x='PCA1',
    y='PCA2',
    z='PCA3',
    # Highlight color for the selected cluster
    color_discrete_sequence=['red'],
).data[0])

# Show the plot in the main area of the Streamlit app
st.plotly_chart(fig)

# Step 5: Show dataset information for the highlighted cluster
st.subheader("Highlighted Dataset Information")
st.write(highlighted_data[['title', 'name']])
