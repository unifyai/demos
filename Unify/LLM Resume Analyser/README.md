# LLM-Resume-Analyser-using-Unify 


[Demo](https://github.com/Sanjay8602/LLM-Resume-Analyser-using-Unify/assets/121057369/a645f849-4617-43db-8936-3a279787f03b) 


<video width="640" height="480" autoplay>
  <source src="video.mp4 file saved in repository folder" type="video/mp4">
Your browser does not support the video tag.
</video>


Streamlit App: https://ai-llm-resume-analyser.streamlit.app

LLM RESUME ANALYSER is an application that allows you to improve your CV using many Language Model analysis and suggestions

## Tech Stack
Streamlit: Used for creating the web application interface that is intuitive and interactive.

Unify AI: Provides the backend LLMs that power the interactions within the application. Unify's API is utilized to send prompts to the LLMs and receive their responses in real-time.

Langchain: LangChain is a powerful framework designed for building applications that integrate with large language models (LLMs), enabling complex interactions and workflows by chaining together various components like prompts, LLMs, and data sources


## Introduction
 
You find more model/provider information in the [Unify benchmark interface](https://unify.ai/hub).

## Usage:
1. Visit the application: [LLM Resume Analyser](https://ai-llm-resume-analyser.streamlit.app/)
2. Input your Unify API Key. If you don’t have one yet, log in to the [Unify Console](https://console.unify.ai/) to get yours.
3. Select the model and provider of your choice
4. Upload your document(s) and click the Submit button
5. Enter your job description and job title
6. Gain insights on Resume match with the job offer and on how to improve your Resume

## Repository and Deployment
The repository link: (https://github.com/OscarArroyoVega/LLM_Resume_Analyser_Unify) or
                     (https://github.com/Sanjay8602/LLM-Resume-Analyser-using-Unify).
To run the application locally, follow these steps:
1. Clone the repository to your local machine.
```bash
git clone https://github.com/Sanjay8602/LLM-Resume-Analyser-using-Unify
```
2. Set up your virtual environment and install the dependencies from `requirements.txt`:
```bash
python -m venv .venv    # create virtual environment 
```
```bash
source .venv/bin/activate   # on Windows use .venv\Scripts\activate.bat
```
```bash
pip install -r requirements.txt
```
3. Run app.py from Streamlit module 

```bash
python -m streamlit run analyser.py
```

## Contributors

|       Name       |                  GitHub Profile                 |
|------------------|-------------------------------------------------|
| Sanjay Suthar    | [Sanjay0806](https://github.com/Sanjay8602)     |
| OscarArroyoVega  | [OscarAV](https://github.com/OscarArroyoVega)   |
| Mayssa Rekik     | [Mayssa Rekik](https://github.com/iammayssa)    |
| Jeya Balang      | [Jeyabalang](https://github.com/jeyabalang)     |
