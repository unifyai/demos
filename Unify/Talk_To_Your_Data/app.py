import streamlit as st
import pandas as pd
import unify
import os

# Page configuration
st.set_page_config(page_title="Chat with CSV", layout="wide")

# Sidebar: Input API Key, Model, and Endpoint
st.sidebar.title("🔧 Configuration")

# Step 1: Input UnifyAI API Key
api_key = st.sidebar.text_input("Enter UnifyAI API Key:", type="password")

if not api_key:
    st.sidebar.warning("⚠️ Please enter your UnifyAI API key to proceed.")
    st.stop()

# Dynamically set the API key
os.environ["UNIFY_KEY"] = api_key

try:
    client = unify.Unify(api_key=api_key)
    models = unify.list_models()
    endpoints = unify.list_endpoints()
except Exception as e:
    st.sidebar.error(f"❌ Error: {str(e)}")
    st.stop()

# Step 2: Select Model and Filter Valid Endpoints
selected_model = st.sidebar.selectbox("Select Model", models)

# Filter endpoints to show only those relevant to the selected model
valid_endpoints = [ep for ep in endpoints if selected_model in ep]

if not valid_endpoints:
    st.sidebar.warning(f"⚠️ No valid endpoints available for {selected_model}.")
    st.stop()

selected_endpoint = st.sidebar.selectbox("Select Endpoint", valid_endpoints)

# Set the endpoint
try:
    client.set_endpoint(selected_endpoint)
except Exception as e:
    st.sidebar.error(f"❌ Error: {str(e)}")
    st.stop()

# Main Content: Upload CSV and Chat
st.title("🗂️ Chat with Your CSV Data using UnifyAI")
st.write("Upload a CSV file and interact with the data using the selected model and endpoint.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    st.write("📋 Dataset Summary:")
    st.text(summary)

    st.subheader("💬 Chat with Your Data")
    user_query = st.text_input("Ask a question about the data:")

    if user_query:
        try:
            response = client.generate(
                user_message=f"{df}\n{user_query}",
                system_message="You are an expert data analyst. Provide insights about the dataset. Please provide the answer not the python code."
            )
            st.write("🤖 **Response:**")
            st.write(response)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

else:
    st.info("👈 Please upload a CSV file to get started.")
