import os
import json
import PyPDF2
import traceback

def read_file(uploaded_file):
    import PyPDF2

    filename = uploaded_file.name
    if filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif filename.endswith('.txt'):
        text = uploaded_file.read().decode('utf-8')
        return text
    else:
        raise Exception("Unsupported file format. Please provide a PDF or TXT file.")
    
def get_table_data(quiz_str):
    try:
        quiz_dict=quiz_str
        quiz_table_data = []

        for key, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " | ".join(
                [
                    f"{option} -> {option_value}" for option,option_value in value["options"].items()
                ]
            )
            answer = value["answer"]
            quiz_table_data.append({"MCQ": mcq, "Options": options, "Answer": answer})
        return quiz_table_data
    except Exception as e:
        traceback.print_exception(type(e), value=e, tb=e.__traceback__)
        return False