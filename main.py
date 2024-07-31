import pandas as pd
import streamlit as st
import plotly.express as px
import requests

# Load the dataset (ensure you have the reduced_df.csv file)
reduced_df = pd.read_csv('data/3d_embedding.csv')

# Step 1: Streamlit UI setup
st.title("Connecting the Dots: Enhanced Data Discovery and Trust")

# Create a sidebar for the chatbot UI
st.sidebar.header("Chatbot")

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to display chat messages


def display_chat_history():
    for message in st.session_state.chat_history:
        st.sidebar.write(message)


# Display chat history
display_chat_history()

# Input field for user queries
user_query = st.sidebar.text_input("Ask me anything about the datasets:")

# Simple response logic for the chatbot
if user_query:
    # Add user's message to chat history
    st.session_state.chat_history.append(f"You: {user_query}")

    response = requests.post("http://localhost:8000/data_gov_chat", json={"user_question": user_query})

    # Here you can add logic to generate a response based on the user query
    # bot_response = "I'm a simple chatbot. How can I help you?"
    if response.status_code == 200:
        bot_response = response.json()['answer']
    else:
        bot_response = "Sorry, I couldn't find an answer to your question."
    st.session_state.chat_history.append(f"Bot: {bot_response}")

    # Clear the input box after submission
    st.sidebar.text_input(
        "Ask me anything about the datasets:", value="", key="input")


# Create a selectbox for choosing a dataset title
st.subheader("Select a Dataset Title to Highlight the Cluster")
dataset_titles = reduced_df['title'].tolist()
selected_title = st.selectbox("Select a Dataset Title", dataset_titles)

# Get the corresponding cluster for the selected title
selected_row = reduced_df[reduced_df['title'] == selected_title]
if not selected_row.empty:
    selected_cluster = selected_row['cluster'].values[0]

    # Highlight the points in the selected cluster
    highlighted_df = reduced_df[reduced_df['cluster'] == selected_cluster]

    # Create a new scatter plot with highlighted cluster
    fig = px.scatter_3d(
        reduced_df,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color='cluster',
        title='3D Embedding of Text Clusters',
        labels={'PCA1': 'PCA 1', 'PCA2': 'PCA 2', 'PCA3': 'PCA 3'},
        hover_data={'title': True, 'name': True}
    )

    # Add the highlighted cluster trace
    fig.add_trace(px.scatter_3d(
        highlighted_df,
        x='PCA1',
        y='PCA2',
        z='PCA3',
        color_discrete_sequence=['red'],
        hover_name='title'
    ).data[0])

    # Show the updated plot with the highlighted cluster
    st.plotly_chart(fig, use_container_width=True)

# Step 4: Show dataset information for the highlighted cluster
if 'selected_cluster' in locals():
    st.subheader("Highlighted Dataset Information")
    st.write(reduced_df[reduced_df['cluster'] ==
             selected_cluster][['title', 'name']])
