import os 
import json
import pandas as pd
import getpass
import traceback


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import LLMChain, SequentialChain
from langchain_community.callbacks.manager import get_openai_callback


os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)


TEMPLATE = """
Text:{text}
You are a quiz generation AI. Your task is to generate a quiz of {number} choice questions for {subject} students in {tone} tone\
based on the provided text. \
Make sure the questions are not repeated and check all the questions to be conforming the text as well. \
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs\
You must always return valid JSON, with double quotes fenced by a markdown code block. Do not return any additional text.
### RESPONSE_JSON
{response_json}
"""

TEMPLATE2 = """
You are an expert english grammarian and writer. Given a multiple choice quiz for {subject} students.\
Your task is to analyze the quiz and provide feedback on its grammatical correctness, clarity, and overall complexity.Only use at max 50 words for complexity.\
Please ensure that your feedback is constructive and aimed at helping the quiz creator improve their work.\
Update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities.\
Quiz_MCQs:
{quiz}

Check from an expert English grammarian and writer of the above quiz:
"""

quiz_generation_prompt = ChatPromptTemplate.from_template(TEMPLATE)
quiz_review_prompt= ChatPromptTemplate.from_template(TEMPLATE2) 

parser = JsonOutputParser()

generate_review_chain = (
    quiz_generation_prompt 
    | llm 
    | parser  # Parse JSON string to dictionary
    | RunnablePassthrough.assign(
            quiz_result=(
                quiz_review_prompt
                | llm
        )
    )
)

