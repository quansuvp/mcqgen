from setuptools import find_packages,setup

setup(
    name='mcqgenerator',
    version='0.0.1',
    author='Khang Nguyen',
    author_email='khangqn1212@gmail.com',
    install_requires=['openai','langchain','langchain-community',
                      'langchain-openai','streamlit','python-dotenv','PyPDF2'],
    packages= find_packages()
)