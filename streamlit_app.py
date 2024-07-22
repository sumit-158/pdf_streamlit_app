from PyPDF2 import PdfReader

# import openai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback


def extract_text_from_pdf(pdf):
    texts = []
    reader = PdfReader(pdf)
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append((page_number + 1, text))
    return texts


with st.sidebar:
    st.title("LLM Pdf Chat")
    pdf = st.file_uploader("Upload your PDF", type="pdf")

# load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


def main():

    if pdf is not None:
        texts = extract_text_from_pdf(pdf)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        # creating chunks of text
        chunks = []
        chunk_page_mapping = []

        for page_number, text in texts:
            page_chunks = text_splitter.split_text(text=text)
            chunks.extend(page_chunks)
            chunk_page_mapping.extend([page_number] * len(page_chunks))
        embeddings = OpenAIEmbeddings(api_key = OPENAI_API_KEY)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        query = st.text_input("Ask questions about your PDF file:")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            # Adding page numbers to the retrieved documents
            retrieved_docs_with_pages = []
            for doc in docs:
                doc_index = chunks.index(doc.page_content)
                page_number = chunk_page_mapping[doc_index]
                retrieved_docs_with_pages.append((doc, page_number))

            llm = OpenAI(api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(
                    input_documents=[doc[0] for doc in retrieved_docs_with_pages],
                    question=query,
                )
                response_with_pages = response + "\n\nReferences:\n"
                for doc, page_number in retrieved_docs_with_pages[:2]:
                    response_with_pages += f"Page {page_number} "
                print(cb)
            st.write(response_with_pages)


if __name__ == "__main__":
    main()
