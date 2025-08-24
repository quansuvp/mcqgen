import os
import json
import traceback
import pandas as pd
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import generate_review_chain
from src.mcqgenerator.logger import logging

with open("./Response.json", "r") as file:
    RESPONSE_JSON = json.load(file)

st.set_page_config(page_title="MCQ Generator", page_icon=":guardsman:", layout="wide")
st.title("MCQ Generator :guardsman:")

with st.form("user_input"):
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    subject = st.text_input("Enter the subject (e.g., Math, Science):")
    number = st.number_input("Enter the number of questions to generate:", min_value=1, max_value=100, value=5)
    tone = st.text_input("Select the tone of the questions:", "Easy")
    submitted = st.form_submit_button("Generate MCQs")

    if submitted and uploaded_file is not None and number and subject and tone:
        with st.spinner("Generating MCQs..."):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    input_params = {
                        "text": text,
                        "subject": subject,
                        "tone": tone,
                        "number": number,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                    result = generate_review_chain.invoke(input_params)
                    
            except Exception as e:
                traceback.print_exception(type(e), value=e, tb=e.__traceback__)
                st.error(f"Error: {str(e)}")
            
            else:
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                if isinstance(result, dict):
                    quiz_str = result.get("quiz", None)
                    if quiz_str is not None:
                        table_data = get_table_data(quiz_str)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index += 1
                            st.subheader("Generated MCQs")
                            st.table(df)
                            st.subheader("Expert Review")
                            st.write(result['quiz_result'].content)
                            logging.info("MCQs generated successfully.")
                        else:
                            st.error("Failed to parse quiz data.")
                            logging.error("Failed to parse quiz data.")
                else:
                    st.write(result)