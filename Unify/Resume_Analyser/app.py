import streamlit as st
import unify
import os

# Page configuration
st.set_page_config(page_title="Resume Enhancement with Chat", layout="wide")

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

# Main Content: Upload Resume and Paste Job Description
st.title("📄 Resume Enhancement with Chat")
st.write("Upload your resume, paste the job description, and interact with the model for analysis and suggestions.")

uploaded_resume = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here", height=200, placeholder="Paste the job description here...")

if uploaded_resume:
    # Convert resume PDF to text
    from PyPDF2 import PdfReader

    def extract_text_from_pdf(pdf_file):
        reader = PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages])

    resume_text = extract_text_from_pdf(uploaded_resume)

    st.subheader("📄 Resume Preview")
    st.text_area("Extracted Resume Text", resume_text, height=300, disabled=True)

if uploaded_resume and job_description.strip():
    st.subheader("🔍 Insights")
    st.write("Analyzing your resume against the job description...")

    try:
        # Generate initial insights
        response = client.generate(
            user_message=f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}\n\n"
                         "Provide detailed insights on how the resume can be improved to better match the job description. "
                         "Focus on skills, experience, and keywords alignment.",
            system_message="You are an expert career coach. Provide actionable and specific feedback."
        )
        st.write("🤖 **Insights:**")
        st.write(response)

        # Chat feature
        st.subheader("💬 Chat with the Model")
        st.write("Ask specific questions about the analysis, suggestions, or anything else.")
        user_query = st.text_input("Enter your question:")

        if user_query:
            chat_response = client.generate(
                user_message=f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}\n\nQuestion: {user_query}",
                system_message="You are an expert career coach. Answer the user's questions based on the given resume and job description."
            )
            st.write("🤖 **Response:**")
            st.write(chat_response)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

elif not uploaded_resume:
    st.info("👈 Please upload your resume to get started.")
elif not job_description.strip():
    st.info("👈 Please paste the job description in the text area to proceed.")
