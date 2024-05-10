from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import json
import time
import uvicorn
import aiofiles
import csv
from PyPDF2 import PdfReader
import mysql.connector


#chatbot imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.document_transformers import DoctranQATransformer
from dotenv import load_dotenv
import os
import json
import asyncio

from langchain.schema import Document
from langchain.document_transformers import DoctranQATransformer
#end of chatbot imports

#semantic similarity part
import math
import re
from collections import Counter as Count

#using spacy for semantic similarity
import spacy

nlp = spacy.load("C:/Users/Ekafor/AppData/Local/Programs/Python/Python312/Lib/site-packages/en_core_web_lg/en_core_web_lg-3.7.1")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

os.environ["OPENAI_API_KEY"] = ""



#define database connection here
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='qa_dev'
)


def count_pdf_pages(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print("Error:", e)
        return None


def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    #summarize data
    

    question_gen = ''

    for page in data:
        question_gen += page.page_content
    

    splitter_ques_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 1000, #changed from 10000 to 1000
        chunk_overlap = 200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]


    splitter_ans_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 1000,
        chunk_overlap = 100
    )

    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

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

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatOpenAI(
        temperature = 0.1,
        model = "gpt-3.5-turbo"
    )

    ques_list = ques.split("\n")

    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list




def get_csv (file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        #make connection to the database here.
        cursor = connection.cursor()

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])

            #save question and answer to database
            insert_query = "INSERT INTO q_a (question, answer) VALUES (%s, %s)"
            values = (question, answer)
            cursor.execute(insert_query, values)
            
    # Close the cursor and connection
    cursor.close()
    connection.close()        
    return output_file


word = re.compile(r"\w+")


def cosine_similarity(vector_1, vector_2):
    inter = set(vector_1.keys()) & set(vector_1.keys())
    num = sum([vector_1[i] * vector_2[i] for i in inter])

    sen_1 = sum([vector_1[i] ** 2 for i in list(vector_1.keys())])
    sen_2 = sum([vector_1[i] ** 2 for i in list(vector_1.keys())])
    denominator = math.sqrt(sen_1) * math.sqrt(sen_2)

    if not denominator:
        return 0.0
    else:
        return float(num) / denominator


def generate_vectors(sent):
    w = word.findall(sent)
    return Count(w)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})


@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)
    # page_count = count_pdf_pages(pdf_filename)
    # if page_count > 5:
    #     return Response(jsonable_encoder(json.dumps({"msg": 'error'})))
    response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": pdf_filename}))
    res = Response(response_data)
    return res



@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res






# sample_text = """When Kyle's family moved to a new town, Kyle
# had to leave all of his old friends behind.
# "You will make new friends," his mother told him.
# But Kyle wasn't so sure. In school, he was very shy, so even
# though some kids talked to him, Kyle didn't say much back. In
# the afternoons after school, he didn't have anyone to play with.
# So Kyle started running around the neighborhood. Every day he
# ran for an hour. Day after day, week after week, Kyle ran. The
# kids in the neighborhood noticed him. So did Coach Benny.
# They all said he was the fastest boy they had ever seen.
# One Saturday, Coach Benny knocked on Kyle's door. "Why
# don't you come with me to soccer club practice today, Kyle?"
# he said. "Everyone wants you on their team. I will introduce you
# to everyone."
# "I thought I was running away from a problem," Kyle told his
# parents later, "but I actually ran towards the answer!"
# """


# sample_text = """Data mining deals with search for description especially in large databases.
# Basically, this search is meant for certain information elicitation with the
# presence of others. More explicitly, data mining refers to variety of techniques to
# extract information by identi@ing relationships and global patterns that exist in
# large databases. Such information is mostly obscured among the vast amount of
# data. Since the data around the information of interest is considered to be
# irrelevant, they are effectively termed to be noise, which are practically desired
# to be absent. In this noisy environment, there are many diverse methods, which
# explain relationships or identi& patterns in the data set. Among these methods
# reference may be made to cluster analysis, discriminant analysis, regression
# analysis, principal component analysis, hypothesis testing, modeling and so
# forth.
# The majority of the data mining methods fall in the category of statistics.
# What makes the data mining methods different is the source and amount of the
# data. Namely, the source is a database, which supposedly has a big amount of
# relevant and irrelevant data in suitable and/or unsuitable form for the intended
# information to be extracted. Referring to the large size of the data set, the
# conventional statistical approaches may not be conclusive. This is because of the
# complexity of the data where only few is known about the properties so that it
# can neither be treated in a statistical framework nor in a deterministic modeling
# framework, for instance. A large data set ffom a stock market is a good example
# where apparently there is no established physical process behind so that the
# properties and behavior can only be modeled in a non-parametric form. The
# validation of such a model is subject to elaborated investigations.
# The main feature of a data set subject to mining is complexity and the
# characteristic feature of the data mining methods distinguishes themselves by
# “learning” as to the conventional statistical methods. That is, even the wellknown 
# statistical or non-statistical or alternatively parametric or non-parametric
# methods are used, the final model parameters or feature vectors of concern are
# established partly or filly by learning. Note that, in conventional statistical
# modeling for relationship or pattern identification, model or pattern parameters
# are established by statistical computation in contrast with learning in data mining
# exercise. Although statistical techniques are apparently ubiquitous in data
# mining, data mining should not be carried out with statistics unless this is
# justified. Statistical methods assist the user in preparation the data for mining.
# This assistance might be in the form of data reduction and hypothesis forming.
# Such a preparation is especially beneficial for knowledge discovery by soft
# computing following the information extraction by data mining. In this work,
# learning in soft computing is accomplished by machine learning methods
# """

sample_text = """Soft computing possesses a variety of special methodologies that work
synergistically and provides “intelligent” information processing capability,
which leads to knowledge discovery. Here the involvement of the methodologies
with learning is the key feature deemed to be essential for intelligence. In the
traditional computing the prime attention is on precision accuracy and certainty.
By contrast, in soft computing imprecision, uncertainty, approximate reasoning,
and partial truth are essential concepts in computation.
These features make the computation soft, which is similar to the neuronal
computation in human brain with remarkable ability so that it yields similar
outcomes but in much simpler form for decision-making in a pragmatic way.
Among the soft computing methodologies, the outstanding paradigms are neural
networks, fuzzy logic and genetic algorithms. In general, both neural networks
and fuzzy systems are dynamic, parallel processing systems that establish the
input output relationships or identify patterns as prime desiderata which are
searched for knowledge discovery in databases. By contrast, the GA paradigm
uses search methodology in a multidimensional space for some optimization
tasks. This search may be motivated by a functional relationship or pattern
identification, as is the case with neural network and fuzzy logic systems.
"""

# Initialize the DoctranQATransformer
qa_transformer = DoctranQATransformer(openai_api_model='gpt-3.5-turbo')


# Transform the document to generate a question and answer
transformed_document = qa_transformer.transform_documents([Document(page_content=sample_text)])


@app.post("/topic")
async def chat(request: Request, topic: str = Form(...)):
    sample_text = """I thought I was running away from a problem," Kyle told his
    parents later, "but I actually ran towards the answer!
    """


# Retrieve the list of questions and answers
qa_pairs = transformed_document[0].metadata['questions_and_answers']

@app.post("/answer")
async def chat(request: Request, answer: str = Form(...), prev_ans: str = Form(...)):

    
    
    # Main loop
    while True:
        # input("Press Enter to receive a question or 'q' to quit: ")
        
        # Check if there are remaining questions
        if not qa_pairs:
            # print("No more questions available.")
            response_data = jsonable_encoder(json.dumps({"question": "No more questions available.", "answer": "No answer"}))
            res = Response(response_data)
            # Add a small delay
            await asyncio.sleep(4)
            return res
            break
        
        # Get and display the next question
        next_question = qa_pairs.pop(0)
        next_q = next_question['question']
        next_a = next_question['answer']

        #do vector calculation here
        vector_a = generate_vectors(prev_ans)
        vector_b = generate_vectors(answer)

        #check cosine similarity here
        cos_sim = cosine_similarity(vector_a, vector_b)

        #use spacy to check semantic similarity here
        phrase_a = nlp(u""+prev_ans)
        phrase_b = nlp(u""+answer)

        sem_sim = phrase_a.similarity(phrase_b)

       
        response_data = jsonable_encoder(json.dumps({"question": next_q, "answer": next_a, "student_answer": prev_ans, "cosine_similarity": sem_sim}))
        res = Response(response_data)

        
        # Add a small delay
        await asyncio.sleep(4)
        return res
        # print("Question:", next_question['question'])
        #print("Reecieved Param:", answer)