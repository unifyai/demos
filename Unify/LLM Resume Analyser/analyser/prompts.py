from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


#TEMPLATE

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", 
                       content="chickens"
                       )


#RESUME SKILL MATCHING
def feature_match_function(resume_text, job_offer, job_title):
    feature_match_prompt = PromptTemplate(
        input_variables=["resume_text", "job_offer", "job_title"],
        template="""You are an AI assistant powered by a Language Model, designed to provide guidance for enhancing and optimizing resumes. 
        Your task is to review the provided resume against the given job offer description and job title. 
        Follow the steps below to complete the task:
        1. Make a bullet point list of matching skills and experiences and missing keywords.
        2. Score each section as follows:
            - Soft skills: 3 * (matching soft skill)
            - Hard skills: 2 * (matching hard skill)
            - Experience: 4 * (each relevant experience for the role)
            - Keywords: 20 - 1 * (each missing keyword)
        3. Calculate a total score at the end in this way:
            - Total score = (soft_skills_score + hard_skills_score + experience_score + keywords_score)*100/80  
        4. Generate an analysis summary that includes the following information:
                - Whether the candidate's profile aligns with the role description in the job offer with mention to the total score.
                - Highlight the strengths and weaknesses of the applicant in relation to the specified job offer description.
        5. Provide an output in JSON format (without any title as "JSON Output" or "scores") with the scores previously generated following the template below:
                {{
                    "soft_skills_score": <soft_skills_score>,
                    "hard_skills_score": <hard_skills_score>,
                    "experience_score": <experience_score>,
                    "keywords_score": <keywords_score>,
                }}
        resume text : {resume_text}
        job offer : {job_offer}
        job title : {job_title}"""
    )
    feature_match_chain = LLMChain(llm=model, prompt=feature_match_prompt, verbose=False)
    match_answer = feature_match_chain.run(resume_text=st.session_state.resume_text, 
                                                   job_offer=st.session_state.job_offer_text, 
                                                   job_title=st.session_state.job_title)
    print(match_answer) 
    return match_answer



