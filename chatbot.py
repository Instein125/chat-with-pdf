#Importing libraries
import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

import streamlit as st

class ChatBot:
    """Allows to answer the qestion based on given document"""
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API")
        genai.configure(api_key=self.api_key)

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5, convert_system_message_to_human=True)
        self.prompt_template = """
        Answer the given question according to the context provided. Study the context and understand the answer relevant to given query and then answer in simple terms.
        If answer is not in the provided context simply answer "I dont know the answer", donot provide wrong answer
        \n
        context: {context}
        \n
        question: {query}
        \n

        Answer: 
        """
        pass

    def get_pdf_texts(self, pdf_docs):
        """
        Extract the text from pdf_docs
        """
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            except:
                st.sidebar.text("Error while extracting texts")

    def get_text_chunks(self, text):
        """
        Convert the text into chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 2500)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        """
        Make a vectorstore for storing the chunks as embeddings to be used for query
        """
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local('faiss_index')

    def get_qa_chain(self):
        """Return a qa chain with prompt template"""
        prompt = PromptTemplate(template=self.prompt_template, input_variables=['context', 'query'])
        chain = load_qa_chain(llm=self.llm, chain_type='stuff', prompt=prompt)
        return chain

    def get_response(self, query):
        """
        Return answet/response based on the pdfs
        """
        db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization= True)
        relevant_docs = db.similarity_search(query)
        # st.write(relevant_docs)
        chain = self.get_qa_chain()
        response = chain(
            {
                "input_documents": relevant_docs,
                'query': query,
            },
            return_only_outputs=True
        )
        return response



if __name__ == "__main__":
    pass
