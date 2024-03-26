import streamlit as st
import os

from chatbot import ChatBot



def main():
    chatbot = ChatBot()
    st.sidebar.header("Upload Documents")
    uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", type = 'pdf', accept_multiple_files = True)
    spinner = st.empty()
    if st.sidebar.button("Upload & Process"):
        with spinner.container():
            st.markdown("**Processing PDFs...**")
            with st.spinner("Extracting text and Storing in vectorstore..."):
                text = chatbot.get_pdf_texts(pdf_docs=uploaded_pdfs)
                text_chunks = chatbot.get_text_chunks(text=text)
                chatbot.get_vector_store(text_chunks=text_chunks)
                st.sidebar.success("Documents uploaded and processed successfully")
        spinner.empty()


    st.header("Chat with PDFs")
    query_imput = st.text_input("Enter your question")
    if query_imput:
        response = chatbot.get_response(query=query_imput)
        st.write("Answer: ", response['output_text'])

if __name__ == "__main__":
    main()