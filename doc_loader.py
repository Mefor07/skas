from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
import os
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document



os.environ["OPENAI_API_KEY"] = ""
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=2000)

loader = TextLoader("./state_of_the_union.txt", encoding = 'UTF-8')

doc = loader.load()[0]


splitter_ques_gen = TokenTextSplitter(
    model_name = 'gpt-3.5-turbo',
    chunk_size = 500, #changed from 10000 to 1000
    chunk_overlap = 100
)

chunks_ques_gen = splitter_ques_gen.split_text(doc.page_content)

document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

llm_ques_gen_pipeline = ChatOpenAI(
    temperature = 0.3,
    model = "gpt-3.5-turbo"
)

prompt_template = """
You are an expert at creating questions based on study materials and reference guides.
Your goal is to prepare a student or teacher for their exams and tests.
You do this by asking questions about the text below:
------------
{text}
------------
Create questions that will prepare the student or teacher for their test
Make sure not to lose any important information.
QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(
    template = prompt_template,
    input_variables = ["text"]
)

refine_template = ("""
You are an expert at creating practice questions based on study materials and reference guides.
Your goal is to help a student or teacher for their exams and tests.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables = ["extisting_answer", "text"],
    template = refine_template
)

ques_gen_chain = load_summarize_chain(
    llm = llm_ques_gen_pipeline,
    chain_type = "refine",
    verbose = True,
    question_prompt = PROMPT_QUESTIONS,
    refine_prompt = REFINE_PROMPT_QUESTIONS
)
my_list = []
my_list.append(document_ques_gen[0])

# ques = ques_gen_chain.run(my_list)

print(my_list)
