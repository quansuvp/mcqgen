import os 
import json
import pandas as pd
import getpass
import traceback


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain.chains import LLMChain, SequentialChain
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.utilities import SerpAPIWrapper


os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["SERPAPI_API_KEY"] = getpass.getpass("Enter your SerpApi API key: ")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.7, openai_api_key=openai_api_key)


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


TEMPLATE3 ="""
You are an assistant that verifies multiple-choice questions using web search results.

Question:
{question}

Options:
{options_str}

Current correct option text: {current_text}

Web search results:
{web_snippets}

1. Decide whether the current answer is correct based only on the options and the web results.
2. If it is wrong, choose the *correct* option from the options list.
3. You must return only valid JSON with this exact schema:
{{
  "correct_letter": "A" | "B" | "C" | "D",
  "correct_text": "<the text of the chosen option>"
}}
"""

quiz_generation_prompt = ChatPromptTemplate.from_template(TEMPLATE)
quiz_review_prompt= ChatPromptTemplate.from_template(TEMPLATE2) 
revise_prompt = ChatPromptTemplate.from_template(TEMPLATE3)

parser = JsonOutputParser()
search = SerpAPIWrapper(serpapi_api_key= serpapi_api_key)


verify_single_question_chain = (
    revise_prompt
    | llm
    | parser
)

def verify_quiz_with_serpapi(quiz_dict: dict) -> dict:
    print(quiz_dict)
    subject = quiz_dict.get("subject", "")
    quiz = quiz_dict.get("quiz", {})

    for q_id, q in quiz.items():
        question_text = q["mcq"]
        options = q["options"]            # dict: {"A": "...", "B": "...", ...}
        current_letter = q["answer"] # text, not letter
        current_answer_text = options.get(current_letter, "")
        print("Current letter: ", current_letter)
        print("Current answer text: ", current_answer_text)

        # Build a search query using question and all options
        options_concat = " ".join(options.values())
        query = f"{question_text} {options_concat}"

        # Call SerpAPI
        web_snippets = search.run(query)

        # Prepare options string for the LLM
        options_str = "\n".join(
            [f"{letter}. {text}" for letter, text in options.items()]
        )

        # Run verification chain
        result = verify_single_question_chain.invoke({
            "subject": subject,
            "question": question_text,
            "options_str": options_str,
            "current_text": current_answer_text,
            "web_snippets": web_snippets,
        })

        # Update with verified answer
        new_letter = result.get("correct_letter", current_letter)
        new_text = result.get("correct_text", options.get(new_letter, current_answer_text))
        print("New letter: ", new_letter)
        print("New answer text: ", new_text)
        # Ensure consistency: answer is stored as TEXT in your schema
        q["answer"] = new_text
        quiz[q_id] = q

        if current_letter != new_letter:
            print("Verified Question:", question_text, "change from", current_answer_text, "to", new_text)
        else:
            print("Verified Question:", question_text, "answer remains the same.")
        
    quiz_dict["quiz"] = quiz
    return quiz_dict

verify_quiz_runnable = RunnableLambda(verify_quiz_with_serpapi)


generate_review_chain = (
    quiz_generation_prompt 
    | llm 
    | parser  # Parse JSON string to dictionary
    | verify_quiz_runnable
    | RunnablePassthrough.assign(
            quiz_result=(
                quiz_review_prompt
                | llm
        )
    )
)

quiz = {'subject': 'A', 'quiz': {'1': {'mcq': 'What is the main purpose of an autoencoder?', 'options': {'A': 'To classify data', 'B': 'To encode input into a compressed representation and then decode it back to reconstruct the input', 'C': 'To generate random data', 'D': 'To train supervised models for labels'}, 'answer': 'B'}, '2': {'mcq': 'What is the key idea behind denoising autoencoders?', 'options': {'A': 'They remove noise from the dataset before training', 'B': 'They corrupt the input with noise and the model reconstructs the clean input', 'C': 'They produce random noise in the latent space', 'D': 'They use labeled data for training'}, 'answer': 'B'}}}

verify_quiz_with_serpapi(quiz)