from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.document_transformers import DoctranQATransformer
from dotenv import load_dotenv
import os
import json

# load_dotenv()
os.environ["OPENAI_API_KEY"] = ""

# splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
# loader = TextLoader("./state_of_the_union.txt", encoding = 'UTF-8')

# content = loader.load()[0]
# chunks = splitter.split_text(content.page_content)
# documents = [Document(page_content=chunk) for chunk in chunks]


# qa_transformer = DoctranQATransformer(openai_api_key='', openai_api_model='gpt-3.5-turbo')

# qa_documents = await qa_transformer.atransform_documents(documents)


# # Call the function within an event loop or within another async function
# result = await process_qa_documents(qa_transformer, documents)


# # qa = []
# # for document in qa_documents:
# #   qa.extend(document.metadata['questions_and_answers'])



# sample_text = """[Generated with ChatGPT]

# Confidential Document - For Internal Use Only

# Date: July 1, 2023

# Subject: Updates and Discussions on Various Topics

# Dear Team,

# I hope this email finds you well. In this document, I would like to provide you with some important updates and discuss various topics that require our attention. Please treat the information contained herein as highly confidential.

# Security and Privacy Measures
# As part of our ongoing commitment to ensure the security and privacy of our customers' data, we have implemented robust measures across all our systems. We would like to commend John Doe (email: john.doe@example.com) from the IT department for his diligent work in enhancing our network security. Moving forward, we kindly remind everyone to strictly adhere to our data protection policies and guidelines. Additionally, if you come across any potential security risks or incidents, please report them immediately to our dedicated team at security@example.com.

# HR Updates and Employee Benefits
# Recently, we welcomed several new team members who have made significant contributions to their respective departments. I would like to recognize Jane Smith (SSN: 049-45-5928) for her outstanding performance in customer service. Jane has consistently received positive feedback from our clients. Furthermore, please remember that the open enrollment period for our employee benefits program is fast approaching. Should you have any questions or require assistance, please contact our HR representative, Michael Johnson (phone: 418-492-3850, email: michael.johnson@example.com).

# Marketing Initiatives and Campaigns
# Our marketing team has been actively working on developing new strategies to increase brand awareness and drive customer engagement. We would like to thank Sarah Thompson (phone: 415-555-1234) for her exceptional efforts in managing our social media platforms. Sarah has successfully increased our follower base by 20% in the past month alone. Moreover, please mark your calendars for the upcoming product launch event on July 15th. We encourage all team members to attend and support this exciting milestone for our company.

# Research and Development Projects
# In our pursuit of innovation, our research and development department has been working tirelessly on various projects. I would like to acknowledge the exceptional work of David Rodriguez (email: david.rodriguez@example.com) in his role as project lead. David's contributions to the development of our cutting-edge technology have been instrumental. Furthermore, we would like to remind everyone to share their ideas and suggestions for potential new projects during our monthly R&D brainstorming session, scheduled for July 10th.

# Please treat the information in this document with utmost confidentiality and ensure that it is not shared with unauthorized individuals. If you have any questions or concerns regarding the topics discussed, please do not hesitate to reach out to me directly.

# Thank you for your attention, and let's continue to work together to achieve our goals.

# Best regards,

# Jason Fan
# Cofounder & CEO
# Psychic
# jason@psychic.dev
# """


sample_text = """Data mining deals with search for description especially in large databases.
Basically, this search is meant for certain information elicitation with the
presence of others. More explicitly, data mining refers to variety of techniques to
extract information by identi@ing relationships and global patterns that exist in
large databases. Such information is mostly obscured among the vast amount of
data. Since the data around the information of interest is considered to be
irrelevant, they are effectively termed to be noise, which are practically desired
to be absent. In this noisy environment, there are many diverse methods, which
explain relationships or identi& patterns in the data set. Among these methods
reference may be made to cluster analysis, discriminant analysis, regression
analysis, principal component analysis, hypothesis testing, modeling and so
forth.
The majority of the data mining methods fall in the category of statistics.
What makes the data mining methods different is the source and amount of the
data. Namely, the source is a database, which supposedly has a big amount of
relevant and irrelevant data in suitable and/or unsuitable form for the intended
information to be extracted. Referring to the large size of the data set, the
conventional statistical approaches may not be conclusive. This is because of the
complexity of the data where only few is known about the properties so that it
can neither be treated in a statistical framework nor in a deterministic modeling
framework, for instance. A large data set ffom a stock market is a good example
where apparently there is no established physical process behind so that the
properties and behavior can only be modeled in a non-parametric form. The
validation of such a model is subject to elaborated investigations.
The main feature of a data set subject to mining is complexity and the
characteristic feature of the data mining methods distinguishes themselves by
“learning” as to the conventional statistical methods. That is, even the wellknown 
statistical or non-statistical or alternatively parametric or non-parametric
methods are used, the final model parameters or feature vectors of concern are
established partly or filly by learning. Note that, in conventional statistical
modeling for relationship or pattern identification, model or pattern parameters
are established by statistical computation in contrast with learning in data mining
exercise. Although statistical techniques are apparently ubiquitous in data
mining, data mining should not be carried out with statistics unless this is
justified. Statistical methods assist the user in preparation the data for mining.
This assistance might be in the form of data reduction and hypothesis forming.
Such a preparation is especially beneficial for knowledge discovery by soft
computing following the information extraction by data mining. In this work,
learning in soft computing is accomplished by machine learning methods
"""
documents = [Document(page_content=sample_text)]
qa_transformer = DoctranQATransformer(openai_api_model='gpt-3.5-turbo')



# # Main loop
# while True:
#     input("Press Enter to receive a question and answer or 'q' to quit: ")
    
#     # Initialize the DoctranQATransformer for each iteration
#     qa_transformer = DoctranQATransformer(openai_api_model='gpt-3.5-turbo')
    
#     # Generate a new question and answer pair
#     transformed_document = qa_transformer.transform_documents([Document(page_content=sample_text)])
    
#     # Retrieve the list of questions and answers
#     qa_pairs = transformed_document[0].metadata['questions_and_answers']
    
#     # Check if there are any questions generated
#     if qa_pairs:
#         # Get the first question and answer pair
#         question_answer_pair = qa_pairs[0]
        
#         # Display the question and answer
#         print("Question:", question_answer_pair['question'])
#         print("Answer:", question_answer_pair['answer'])
#     else:
#         print("No questions generated.")
#         break
from langchain.schema import Document
from langchain.document_transformers import DoctranQATransformer

# Initialize the DoctranQATransformer
qa_transformer = DoctranQATransformer(openai_api_model='gpt-3.5-turbo')

# Transform the document to generate a question and answer
transformed_document = qa_transformer.transform_documents([Document(page_content=sample_text)])
    
# Retrieve the list of questions and answers
qa_pairs = transformed_document[0].metadata['questions_and_answers']

# Main loop
while True:
    input("Press Enter to receive a question or 'q' to quit: ")
    
    # Check if there are remaining questions
    if not qa_pairs:
        print("No more questions available.")
        break
    
    # Get and display the next question
    next_question = qa_pairs.pop(0)
    print("Question:", next_question['question'])