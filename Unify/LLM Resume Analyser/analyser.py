import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import pi
import json
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from data.unify_endpoints_data import model_provider, dynamic_provider
from langchain_unify.chat_models import ChatUnify
from sentence_transformers import SentenceTransformer, util
from httpx import LocalProtocolError

# Function to extract text from a PDF or DOCX document
def extract_text_from_file(file):
    """Extracts text from a PDF or DOCX document."""
    def extract_text_from_doc(file):
        text = ""
        doc = Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text   

    def extract_text_from_pdf(file):
        text = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()     
        return text
    
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'docx':
        return extract_text_from_doc(file)


# Configure the Streamlit page
st.set_page_config("LLM Resume Analyser", page_icon="😎")  
st.title("LLM Resume Analyser 😎")
st.write("Try different LLMs to see how they behave with your resume and a job offer.")

# Initialize session state variables
if 'unify_api_key' not in st.session_state:
    st.session_state.unify_api_key = ''
if 'endpoint' not in st.session_state:
    st.session_state.endpoint = ''
if 'resume_doc' not in st.session_state:
    st.session_state.resume_doc = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ''
if 'job_offer_text' not in st.session_state:
    st.session_state.job_offer_text = ''
if 'job_title' not in st.session_state:
    st.session_state.job_title = ''
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = '.'
if "num_job_offers_input" not in st.session_state:
    st.session_state.num_job_offers_input = 5


with st.sidebar:
    st.write("Choose a LLM to perform the analysis.")
    st.write("")
    st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password",
                                                   placeholder="Enter Unify API Key", 
                                                   help="All your favorite models with [Unify API](https://console.unify.ai)"
                                                )
    
    with st.expander("Model and Provider selection"):
        model_name = st.selectbox("Select Model", options=model_provider.keys(), index=20,
                                  placeholder="Model", help="All your favorite models with Unify API")
    
        if st.toggle("Enable Dynamic Routing"):
            provider_name = st.radio("Select a Provider", options=dynamic_provider, 
                                     help="dynamic routing powered by Unify API")
        else:
            provider_name = st.radio("Select a Provider", options=model_provider[model_name], 
                                     help="Providers with access to the selected model")
        model_temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)   
    st.session_state.endpoint = f"{model_name}@{provider_name}"
    
    with st.expander("Resume and Job offer inputs"):
        st.session_state.resume_doc = st.file_uploader(label="Upload your Resume*", type=("pdf","docx"), accept_multiple_files=False)

        if st.session_state.resume_doc is not None and st.button("Process resume"): 
            try:
                st.session_state.resume_text = extract_text_from_file(st.session_state.resume_doc)
                st.success("Resume has been successfully processed!")
            except Exception as e:
                st.error("Unable to recognize the document. Please try a compatible format.")

        st.session_state.job_offer_text = st.text_area(label="Job offer description*", key="Job_offer", placeholder="Paste here the job offer description")
        
        st.session_state.job_title = st.text_input("Job title*", key="Job_title", placeholder="enter here your desired job title")
    

try:    
    # Unify model initialization     
    model = ChatUnify(
        model=st.session_state.endpoint,
        unify_api_key=st.session_state.unify_api_key,
        temperature=model_temperature
        )
except LocalProtocolError as e:
    if 'Bearer' in str(e):
        st.warning("""
            Insert your UNIFY API key here or get one from 
            [console.unify.ai](https://console.unify.ai)
            """
        )
    else:
        raise e

def feature_match_function(resume_text, job_offer):
    with st.spinner("Setting up the model..."):
        feature_match_prompt = PromptTemplate(
            input_variables=["resume_text", "job_offer"],
            template = """You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
            Your task is to review the provided resume against the given job offer description. 
            Follow the steps below in order to complete the task:

            Step 1: Identify and list:
                - Matching soft skills
                - Matching hard skills
                - Relevant experiences for the position
                - Matching qualifications (education and certifications)
                - Other relevant keywords

            Step 2: Assign a score from 1 to 100 for each category based on how well the resume matches the job offer. Provide the scores in the specified format without any modifications.

            Step 3: Provide the output in two parts:
            1. **Analysis Summary**: An analysis summary of how the candidate's profile aligns with the role description in the job offer. Highlight the strengths and weaknesses of the applicant in relation to the specified job offer description.
            2. **---**: A JSON format detailing the identified matches and their scores in each category using the template below. Ensure that the JSON format is strictly followed to avoid parsing errors:
                {{
                "Soft skills": <soft_skills_score>,
                "Hard skills": <hard_skills_score>,
                "Experience": <experience_score>,
                "Education and certifications": <education_and_certifications_score>,
                "Keywords": <keywords_score>
                }}
                
            Resume Text: {resume_text}
            Job Offer: {job_offer}

            Ensure that the JSON output is correctly formatted and can be parsed without errors.
            """
        )
    with st.sidebar.container(border=True):
        st.text(f"Running prompt: {feature_match_prompt.template}")
    feature_match_chain = LLMChain(llm=model, prompt=feature_match_prompt, verbose=False)
    with st.spinner('Generating match...'):
        match_answer = feature_match_chain.run(resume_text=st.session_state.resume_text, 
                                            job_offer=st.session_state.job_offer_text)
    return match_answer


def match_report(match_answer):
    def extract_text_analysis(match_answer):
        if "{" not in match_answer or "}" not in match_answer:
            st.warning("Please try again. As some language models sometimes have difficulties following precise parsing instructions. If in 5 attempts the model doesn't rise an answer maybe you should consider highly probable that the model is not able to provide the answer.")
        # Extract JSON part from the match answer and convert it to a dictionary
        json_start = match_answer.index("{")
        json_end = match_answer.rindex("}") + 1
        json_part = match_answer[json_start:json_end]
        text_analysis = match_answer[:json_start].strip()
        try:
            scores_dict = json.loads(json_part)
        except json.JSONDecodeError as e:
            st.warning("Please try again. As some language models sometimes have difficulties following precise parsing instructions. If in 5 attempts the model doesn't rise an answer maybe you should consider highly probable that the model is not able to provide the answer.")
        return text_analysis, scores_dict
    
    def create_radar_chart(scores_dict):
        
        labels = list(scores_dict.keys())
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        # Define scores as the list of values in scores_dict
        scores = list(scores_dict.values())
        scores += scores[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40","60", "80", "100"], color="grey", size=7)
        plt.ylim(0, 100)
        ax.plot(angles, scores, linewidth=2, linestyle='solid')
        ax.fill(angles, scores, 'r', alpha=0.1)
         # Add scores to the plot
        for label, angle, score in zip(labels, angles, scores):
            ax.text(angle, score, str(score))
        
        return fig
  
    text_analysis, scores_dict = extract_text_analysis(match_answer)
    fig = create_radar_chart(scores_dict)
    match_report = text_analysis, fig, scores_dict 
    return match_report


def semantic_visualizations_function(resume_text, job_offer):
   # Load pre-trained Sentence-BERT model
    model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    def skill_list_function (resume_text):
        with st.spinner("Setting up the model..."):
            skill_list_prompt = PromptTemplate(
                input_variables=["resume_text"],
                template="""
                First extract the following information from the provided resume_text:
                    1. Soft skills list
                    2. Hard skills list
                    3. General keywords in the resume list
                    4. keywords in professional experiences list
                    5. keywords in education and certifications list
                    6. other relevant knowledge keywords list
                
                With the information extracted from the resume_text provide the output in JSON format using the template below:
                {{
                "soft_skills": ["soft_skill1", "soft_skill2", "soft_skill3", "..."],
                "hard_skills": ["hard_skill1", "hard_skill2", "hard_skill3", "..."],
                "keywords": ["keyword1", "keyword2", "keyword3", "..."],
                "experience": ["experience1", "experience2", "experience3", "..."],
                "education_and_certifications": ["education1", "certification1", "certification2", "..."],
                "other_knowledge": ["other_knowledge1", "other_knowledge2", "other_knowledge3", "..."]
                }}
                resume_text:
                {resume_text}
                """
            )
        with st.sidebar.container(border=True):
            st.text(f"Running prompt: {skill_list_prompt.template}")
        skill_list_chain = LLMChain(llm=model, prompt=skill_list_prompt, verbose=False)
        with st.spinner("extracting skills from resume..."):
            skill_list = skill_list_chain.run(resume_text=st.session_state.resume_text)
            # Parse JSON string to dictionary
            json_start1 = skill_list.index("{")
            json_end1 = skill_list.rindex("}") + 1
            json_part1 = skill_list[json_start1:json_end1]
            skill_dict = json.loads(json_part1)
            # Check if the output is a dictionary
            if not isinstance(skill_dict, dict):
                st.warning("No skills found. Try again or try another model.")
        return skill_dict

    def requirements_list_function (job_offer):
        requirements_list_prompt = PromptTemplate(
            input_variables=["job_offer"],
            template="""First extract the following information from the provided job_offer and make a list:
                1. "requirements": All the keywords that refers to requirements, skills, experiences or other qualities needed for the job offer.
            
            Provide the output in JSON format using the template below:
            {{
            "requirements": ["keyword1", "keyword2", "keyword3", "..."],
            }}
            job_offer:
            {job_offer}
            """
        )
        with st.sidebar.container(border=True):
            st.text(f"Running prompt: {requirements_list_prompt.template}")
        requirements_list_chain = LLMChain(llm=model, prompt=requirements_list_prompt, verbose=False)
        with st.spinner("extracting requirements from job offer..."):
            requirements_list = requirements_list_chain.run(job_offer=st.session_state.job_offer_text)
            # Parse JSON string to dictionary
            json_start2 = requirements_list.index("{")
            json_end2 = requirements_list.rindex("}") + 1
            json_part2 = requirements_list[json_start2:json_end2]
            requirements_dict = json.loads(json_part2)
            # Check if the output is a dictionary
            if not isinstance(requirements_dict, dict):
                st.warning("Output is not a dictionary. Try again or try another model.")
        return requirements_dict       
    
    skill_dict = skill_list_function(resume_text=st.session_state.resume_text)
    requirements_dict = requirements_list_function(job_offer=st.session_state.job_offer_text)  
    categories = list(skill_dict.keys())
    requirements = requirements_dict["requirements"]
    # Combine all categories into one list
    all_skills = []
    for category in skill_dict:
        all_skills.extend(skill_dict[category])

    st.write(f"Total features collected: {len(all_skills)}")
    # Define similarity function using BERT
    def evaluate_similarity(sentence1, sentence2):
        embeddings1 = model_encoder.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model_encoder.encode(sentence2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return similarity.item()
    # Create similarity matrix
    def create_similarity_matrix(skill_list, requirement_list):
        with st.spinner("computing semantic similarity matrix..."):
            try:
                matrix = np.zeros((len(skill_list), len(requirement_list)))
                for i, skill in enumerate(skill_list):
                    for j, req in enumerate(requirement_list):
                        matrix[i, j] = evaluate_similarity(skill, req)
                return matrix
            except Exception as e:
                st.warning("It didn't work this time, try it again! Take in consideration that small models sometimes struggle when it comes to give a formatted answer.")      
        
    similarity_matrix = create_similarity_matrix(all_skills, requirements)
    
    # Plot heatmap
    with st.spinner('Generating Visualization 1...'):
        st.write("##### Semantic Heatmap")
        st.write ("This heatmap represents the semantic similarity matrix that was previously calculated, comparing each skill and experience from the resume against the job offer requirements.")
        def plot_heatmap(matrix, skill_list, requirement_list):
            df = pd.DataFrame(matrix, index=skill_list, columns=requirement_list)
            df = df.loc[:, df.sum().sort_values(ascending=False).index]  # Sort columns by their sum
            df = df.T   # Transpose the dataframe for better visualization
            plt.figure(figsize=(20, 15))
            sns.heatmap(df, annot=False, cmap='viridis', cbar=True, linewidths=.2)
            # plt.title('Similarity Heatmap for All Skills Against Requirements')
            #plt.xlabel('Requirements')
            #plt.ylabel('Skills')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            st.pyplot(plt)
        plot_heatmap(similarity_matrix, all_skills, requirements)
    
    # Plot barplot
    with st.spinner('Generating Visualization 2...'):
        st.write("##### Requirements distributed score")
        st.write("The following bar plot displays the total similarity scores of the job offer keywords as they are distributed across the skills and experiences listed in the resume. This approach evaluates how well the resume aligns with the job requirements by considering the entire profile in a comprehensive manner.")
        def plot_sum_bar(matrix, requirement_list):
            sum_values = np.sum(matrix, axis=0)
            df_sum = pd.DataFrame(sum_values, index=requirement_list, columns=['Sum'])
            df_sum = df_sum.sort_values(by='Sum', ascending=True)
            # Normalize the sum_values for colormap mapping
            norm = mcolors.Normalize(vmin=df_sum['Sum'].min(), vmax=df_sum['Sum'].max())
            colors = [plt.cm.viridis(norm(value)) for value in df_sum['Sum']]

            plt.figure(figsize=(20, 15))
            plt.barh(df_sum.index, df_sum['Sum'], color=colors)
            #plt.xlabel('Distributed score by sum of similarities')
            #plt.ylabel('Requirements')
            plt.xticks(rotation=0)  # Rotate x-axis labels by 90 degrees
            plt.box(False)  # Remove the frame
            st.pyplot(plt)
        plot_sum_bar(similarity_matrix, requirements)

    
def suggested_changes_function(resume_text, job_offer):
    with st.spinner('Generating suggested changes...'):
        feature_suggested_changes_prompt = PromptTemplate(
            input_variables=["resume_text", "job_offer"],
            template="""You are an AI assistant designed to enhance and optimize resumes to better match specific job offers. 
            Your task is to review the provided resume in light of the given job offer description and provide detailed suggestions for improvement.
        
            Follow these steps in order:
            1. Extract and list keywords from both the job offer and the resume.
            2. Identify and list matching soft skills, hard skills, qualifications, and experiences between the resume and the job offer.
            3. Identify missing keywords in the resume that are present in the job offer.
            4. Identify keywords and skills implied by the resume that could be explicitly added.
            5. Identify missing experiences in the resume that are implied and could be explicitly added.

            Output a suggestions list using the following format and nothing else:
                A. Rephrasing suggestions:
                - <bullet-point list of rephrasing suggestions>
                
                B. Adding keywords and skills:
                - <bullet-point list of additional keywords and skills>
                
                C. Adding missing experiences:
                - <bullet-point list of additional experiences>

            Ensure that the suggestions are highly relevant to the job offer, contextually appropriate, professionally beneficial, and they avoid redundant or unnecessary additions. Do not include any other text or explanations outside the specified format.

            resume_text: {resume_text}
            job_offer: {job_offer}
            """
        )
        with st.sidebar.container(border=True):
            st.text(f"Running prompt: {feature_suggested_changes_prompt.template}")
        feature_suggested_changes_chain = LLMChain(llm=model, prompt=feature_suggested_changes_prompt, verbose=False)
        suggested_changes = feature_suggested_changes_chain.run(resume_text=st.session_state.resume_text,
                                                            job_offer=st.session_state.job_offer_text
                                                            )
        st.session_state.suggested_changes = suggested_changes
        return suggested_changes


def apply_changes_function (resume_text, job_offer, suggested_changes):
    apply_changes_prompt = PromptTemplate(
        input_variables=["resume_text", "job_offer", "suggested_changes"],
        template="""You are an AI assistant designed to enhance and optimize resumes to better match specific job offers.
        Given a resume ({resume_text}) and a report with suggested changes ({suggested_changes}), you will apply the changes to create an updated new resume:
            
            1. Add the suggested changes from {suggested_changes} to the new resume, ensuring they fit naturally and contextually.
            2. Apply the rephrasing suggestions while maintaining the overall coherence and focus of the resume.
            3. Verify that the updated resume keep all the existing keywords, skills, qualifications, and experiences from the {resume_text} that are valuable for the job offer ({job_offer}).
            4. Ensure that the new resume is well-structured, concise, and tailored to the job offer

        Return the updated resume as the final output.

        resume_text: {resume_text}
        job_offer: {job_offer}
        suggested_changes: {suggested_changes}
        """
    )
    with st.sidebar.container(border=True):
        st.text(f"Running prompt: {apply_changes_prompt.template}")
    apply_changes_chain = LLMChain(llm=model, prompt=apply_changes_prompt, verbose=False)
    with st.spinner("applying changes..."):
        resume_updated_text = apply_changes_chain.run(resume_text=st.session_state.resume_text,
                                                      job_offer=st.session_state.job_offer_text, 
                                            suggested_changes= st.session_state.suggested_changes                                       
                                            )
        return resume_updated_text 

           
def job_titles_list_function (resume_text, num_job_offers,job_title):
    job_titles_prompt = PromptTemplate(
        input_variables=["resume_text", "num_job_offers"],
        template=""" You are an AI assistant designed to enhance and optimize resumes to better match specific job offers.
        Given a resume ({resume_text}) and an integer ({num_job_offers}) and the current job_title ({job_title})::
            1.Identify and return a list of the {num_job_offers} alternative job titles to the current one that best match the skills and experience described in the resume. 
            2.Focus on titles that directly align with the candidate's qualifications and consider factors like keywords, technologies mentioned, and past job roles.
        
        Return just the list of job titles as a bullet-point list.  
        
        resume_text: {resume_text}
        num_job_offers: {num_job_offers}
        job_title: {job_title}
        """
    )
    with st.sidebar.container(border=True):
        st.text(f"Running prompt: {job_titles_prompt.template}")
    job_titles_chain = LLMChain(llm=model, prompt=job_titles_prompt, verbose=False)
    with st.spinner("generating job titles..."):
        job_titles = job_titles_chain.run(resume_text=st.session_state.resume_text,
                                            num_job_offers=st.session_state.num_job_offers_input,
                                            job_title=st.session_state.job_title                                        
                                            )
        return job_titles  


def custom_prompt_function(user_prompt, resume_text, job_offer, job_title):
    custom_user_prompt = PromptTemplate(
        input_variables=[ "user_prompt","resume_text", "job_offer", "job_title"],
        template="""You are an AI assistant designed to enhance and optimize resumes to better match specific job offers.
        Given the user prompt as a query to answer and use the resume, job offer, and job title as context to provide a short answer that addresses the user's query.
        user_prompt: {user_prompt}
        resume_text: {resume_text}
        job_offer: {job_offer}
        job_title: {job_title}
        """
    )
    custom_prompt_chain = LLMChain(llm=model, prompt=custom_user_prompt, verbose=False)
    custom_QA = custom_prompt_chain.run(user_prompt=st.session_state.user_prompt,
                                        resume_text=st.session_state.resume_text,
                                        job_offer=st.session_state.job_offer_text,
                                        job_title=st.session_state.job_title,
                                            
                                        )
    return custom_QA
    

def create_radar_chart(data):
    
    categories = ["Soft skills", "Hard skills", "Experience", "Education and certifications", "Keywords"]
    num_vars = len(categories)
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.get_cmap("tab10", len(data))
    
    for idx, entry in enumerate(data):
        scores_dict = entry["score"]
        model = entry["model"]
        temperature = entry["temperature"] 
        scores = [
            scores_dict.get("soft_skills_score") or scores_dict.get("Soft skills", 0),
            scores_dict.get("hard_skills_score") or scores_dict.get("Hard skills", 0),
            scores_dict.get("experience_score") or scores_dict.get("Experience", 0),
            scores_dict.get("Education and certifications", 0),  # Some entries may not have this score
            scores_dict.get("keywords_score") or scores_dict.get("Keywords", 0)
        ]
        scores += scores[:1]

        ax.plot(angles, scores, linewidth=2, linestyle='solid', label=f"{model} ({temperature})", color=colors(idx))
        ax.fill(angles, scores, color=colors(idx), alpha=0.25)
        # Add scores to the chart for the last entry only
        if idx == len(data) - 1:
            for angle, score in zip(angles, scores):
                ax.annotate(str(score), xy=(angle, score), ha='center', va='bottom')
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(5)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=7)
    plt.ylim(0, 100)
    ax.axhline(75, color='r', linestyle='--')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    # plt.title('Model Scores Radar Chart')
    plt.show()


def create_bar_plot(data):
    df = pd.DataFrame(data)
    # Normalize the score JSON data
    scores_df = pd.json_normalize(df['score'])
    scores_df['model'] = df['model']
    scores_df['temperature'] = df['temperature']

    # Prepare the data for plotting
    melted_df = scores_df.melt(id_vars=['model', 'temperature'], var_name='Category', value_name='Score')
    melted_df['Label'] = melted_df['model'] + " (" + melted_df['temperature'].astype(str) + ")"

    # Grouped bar chart where bars are grouped by category and colored by model + temperature
    fig, ax = plt.subplots(figsize=(12, 8)) 
    sns.barplot(data=melted_df, x='Category', y='Score', hue='Label', palette='viridis_r')

    # Add a star symbol over the bars for those who have double asterisks at the end of the model value

    plt.xlabel('')
    plt.ylabel('')
    # plt.title('Scores by Category for Different Models and Temperatures')
    ax.text(-0.5, 75, '75', color='r')
    ax.axhline(75, color='r', linestyle='--')
    plt.legend(loc='upper right', title='Model (Temperature)')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    sns.despine()  
    plt.show()    
    
    
    
# UI main structure
tab1, tab2, tab3, tab4 = st.tabs(["Resume VS Job offer","Improve your Resume", "Job Search", "Try your custom prompt"])

with tab1:
    col1, col2, col3 = st.columns(3)
    feature_match_button = col1.button("RESUME MATCH")
    Scores_button = col2.button("SESSION SCORES")
    semantic_visualizations_button= col3.button("SEMANTIC VISUALIZATIONS")
    container1 = st.container(border=True)
    
with tab2:
    col1, col2 = st.columns(2)
    feature_suggested_changes_button = col1.button("SUGGESTED CHANGES")
    apply_changes_button = col2.button("APPLY CHANGES AND COMPARE")
    container2 = st.container(border=True)
    
with tab3:
    col1, col2= st.columns(2)
    feature_suggested_titles_button = col1.button("TITLE NAMES FOR JOB SEARCH")
    st.session_state.num_job_offers_input = col2.slider('Select number of job offers', 1, 20, 5)
    container3 = st.container(border=True)
    
with tab4:
    st.session_state.user_prompt = st.text_area("Try your prompt",placeholder="Enter your prompt here")
    submit_user_prompt_button = st.button("Submit")
    container4 = st.container(border=True)
    
with container1:
    if feature_match_button:
        if st.session_state.job_offer_text and st.session_state.resume_text:
            match_answer = feature_match_function(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text
                                                   )
            analysis_text, radar_chart, scores_dict = match_report(match_answer)
            st.session_state.scores.append({'score': scores_dict,
                                            'model': str(model_name),  # Convert the model to a string for storage
                                            'temperature': model_temperature
                                            }) 
              
            with st.container():
                st.write("### Resume VS Job offer: analysis")
                st.write(analysis_text)
                st.write("##### Radar Chart")
                st.pyplot(radar_chart)
                
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
            
    elif Scores_button:
        st.write("### Scores by the models used in this session*")
        # Radar chart for all models used in the session
        model_names = [entry['model'] for entry in st.session_state.scores]
        num_queries = len(st.session_state.scores)
        st.write("**For better understanding of this feature is recommended to try the 'Resume Match' feature with different models and temperatures.*")
        st.write(f"In your session, you have conducted {num_queries} queries to these models: {model_names}")
        st.write("")
        st.write("")
        
        st.write("##### Radar Chart of Scores by Model and Temperature")
        st.write("The radar chart below is an intuitive way to compare the scores of different models and temperatures.")
        create_radar_chart(st.session_state.scores)
        st.pyplot(plt)
        
        st.write("")  # Add an empty line as a spacer
        st.write("")  
        st.write("")  
        st.write("##### Grouped Bar Chart of Scores by Category for Different Models and Temperatures")
        st.write("This chart shows grouped bars by category and colored by model and temperature. Even the distance between different model outputs some patterns are visible.")
        st.write("")
        create_bar_plot(st.session_state.scores)
        st.pyplot(plt)
        

        
    elif semantic_visualizations_button:
        if st.session_state.resume_text and st.session_state.job_offer_text:
            st.write("### Visualizations based on the semantic similarity matrix")
            st.write(""" After the model extracts a list of skills and experiences from the resume and similarly processes the requirements in the job offer, a similarity matrix is computed using an embedding process. This matrix enables us to thoroughly explore the candidate's match with the job offer by comparing the entire profile against the job requirements. Visualizations provide a more intuitive way to interpret these insights.
                     As the time complexity is O(n^2) depending on the number of skills and experiences extracted, computing time can vary significantly between models.(processing time: 1 to 2 minutes aprox.)""")
            semantic_visualizations_function(resume_text=st.session_state.resume_text, 
                                    job_offer=st.session_state.job_offer_text)
                     
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
            
with container2:                                                  
    if feature_suggested_changes_button:
        if st.session_state.job_offer_text and st.session_state.resume_text:
            suggested_changes_answer = suggested_changes_function(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text
                                                   )
            st.markdown("### Suggested Changes")
            st.write("The following suggestions will help you to find other job opportunities that match with your profile")
            suggested_changes_answer_text = suggested_changes_answer.strip() 
            st.write(suggested_changes_answer_text)
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.")
      
    if apply_changes_button:
        st.write("**Resume VS Job offer analysis will be applied for both resumes. (processing time: 1min aprox.)*")  
        if st.session_state.suggested_changes:
            if st.session_state.job_offer_text and st.session_state.resume_text:
                original_match_answer = feature_match_function(resume_text=st.session_state.resume_text, 
                                                    job_offer=st.session_state.job_offer_text
                                                    )
                original_analysis_text, original_radar_chart, original_scores_dict = match_report(original_match_answer)
                st.session_state.scores.append({'score': original_scores_dict,
                                                'model': str(model_name),  # Convert the model to a string for storage
                                                'temperature': model_temperature
                                                }) 
                    
                resume_updated = apply_changes_function(resume_text=st.session_state.resume_text,
                                                        job_offer=st.session_state.job_offer_text,
                                                        suggested_changes= st.session_state.suggested_changes)
                st.session_state.suggested_changes = ""  
                resume_updated_text = resume_updated.strip() 
                st.session_state.resume_updated_text = resume_updated_text
                new_match_answer = feature_match_function(resume_text=st.session_state.resume_updated_text, 
                                                        job_offer=st.session_state.job_offer_text
                                                        )
                new_analysis_text, new_radar_chart, new_scores_dict = match_report(new_match_answer)
                st.session_state.scores.append({'score': new_scores_dict,
                                                'model': str(model_name + "**"),  # Convert the model to a string for storage
                                                'temperature': model_temperature
                                                }) 
                            
                with st.container():
                    col1, col2 = st.columns(2) 
                    with col1:
                        st.write("##### Original Resume VS Job offer: analysis")
                        st.write(original_analysis_text)
                        st.write("previous Radar Chart")
                        st.pyplot(original_radar_chart)        
                                
                    with col2:
                        st.write("##### Resume with changes VS Job offer: analysis")
                        st.write(new_analysis_text)
                        st.write("new Radar Chart")
                        st.pyplot(new_radar_chart)    
                with st.expander("**updated Resume text**"):
                    st.write(resume_updated_text)
                    st.download_button(
                            label="Download updated resume text",
                            data=resume_updated_text,
                            file_name="updated_resume.txt",
                            mime="text/plain",)
                        
            else:
                st.warning("Please upload a resume and provide a job offer text and job title to proceed.")                
        else:
            st.warning("Please generate suggested changes before.")
               
with container3: 
    if feature_suggested_titles_button:
        if st.session_state.resume_text:
            # st.session_state.num_job_offers_input = st.slider('Select number of job offers', 1, 20, 5)
            # # Check if the slider value has changed
            # if "num_job_offers_input" not in st.session_state or st.session_state.num_job_offers_input != num_job_offers_input:
            #     # Store the current slider value as the previous one for the next run
            #     st.session_state.num_job_offers_input = num_job_offers_input
                
            suggested_job_titles_answer = job_titles_list_function(resume_text=st.session_state.resume_text, 
                                                                    num_job_offers=st.session_state.num_job_offers_input
                                                                    )
            st.markdown("##### Other matching Job Titles")
            st.write("Suggested job titles based on the candidate's profile can expand the job search, uncovering opportunities that were previously overlooked.")
            suggested_job_titles_text= suggested_job_titles_answer.strip()
            st.write(suggested_job_titles_text)       
                
with container4:           
    if submit_user_prompt_button:
        if st.session_state.job_title and st.session_state.job_offer_text and st.session_state.resume_text:
            answer = custom_prompt_function(user_prompt=st.session_state.user_prompt,
                                            resume_text=st.session_state.resume_text, 
                                            job_offer=st.session_state.job_offer_text, 
                                            job_title=st.session_state.job_title
                                            )
            answer_text = answer.strip()
            with st.container():
                st.markdown("### Custom Prompt Answer")
                st.write(answer_text)       
        else:
            st.warning("Please upload a resume and provide a job offer text and job title to proceed.") 
                   
    
        
        
        

