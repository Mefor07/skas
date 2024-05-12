import asyncio
import json
import uvicorn
from fastapi import FastAPI, Form, Request, Response
# from doctran_qa_transformer import DoctranQATransformer
from langchain.document_transformers import DoctranQATransformer
from langchain.schema import Document
# from doctran_qa_transformer.types import Document
# from sklearn.metrics.pairwise import cosine_similarity

#semantic similarity part
import math
import re
from collections import Counter as Count
import spacy
import os

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Initialize the DoctranQATransformer
qa_transformer = DoctranQATransformer(openai_api_model='gpt-3.5-turbo')

# Load the SpaCy model
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("C:/Users/Ekafor/AppData/Local/Programs/Python/Python312/Lib/site-packages/en_core_web_lg/en_core_web_lg-3.7.1")

# Default question set
file_path = "static/docs/softcomputing.txt"
with open(file_path, "r", encoding="utf-8") as file:
  default_question_set = file.read()

# Transform the default question set to generate a question and answer
transformed_default_questions = qa_transformer.transform_documents([Document(page_content=default_question_set)])
default_qa_pairs = transformed_default_questions[0].metadata['questions_and_answers']

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

# Endpoint to restart the FastAPI server and change the question set
@app.post("/restart-server")
async def restart_server(request: Request, topic: str = Form(...)):
    global default_qa_pairs
    file_path = "static/docs/"+topic+".txt"
    with open(file_path, "r", encoding="utf-8") as file:
      topic = file.read()
    transformed_question_set = qa_transformer.transform_documents([Document(page_content=topic)])
    default_qa_pairs = transformed_question_set[0].metadata['questions_and_answers']
    return {"message": "Server restarted successfully with new question set."}

# Endpoint to get the next question
@app.post("/answer")
async def next_question(request: Request, answer: str = Form(...), prev_ans: str = Form(...)):
    global default_qa_pairs
    
    # Check if there are remaining questions in the default set
    if not default_qa_pairs:
        await asyncio.sleep(4)
        return {"question": "No more questions available.", "answer": "No answer"}
    
    # Get the next question
    next_question = default_qa_pairs.pop(0)
    next_q = next_question['question']
    next_a = next_question['answer']

    # Calculate similarity metrics (dummy implementation)
    cos_sim = 0.0
    sem_sim = 0.0

    #do vector calculation here
    vector_a = generate_vectors(prev_ans)
    vector_b = generate_vectors(answer)

    #check cosine similarity here
    cos_sim = cosine_similarity(vector_a, vector_b)

    #use spacy to check semantic similarity here
    phrase_a = nlp(u""+prev_ans)
    phrase_b = nlp(u""+answer)
    sem_sim = phrase_a.similarity(phrase_b)
    
    await asyncio.sleep(4)
    return {
        "question": next_q,
        "answer": next_a,
        "student_answer": prev_ans,
        "cosine_similarity": cos_sim,
        "semantic_similarity": sem_sim
    }

if __name__ == "__main__":
    uvicorn.run("qa_server:app", host="127.0.0.1", port=8000, reload=True)
