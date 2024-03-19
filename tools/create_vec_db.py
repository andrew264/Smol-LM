import time

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

from model import HFNomicEmbeddings

if __name__ == '__main__':
    rag_content = './data/rag_content.txt'
    embeddings_fn = HFNomicEmbeddings(device="cuda")

    loader = TextLoader(rag_content)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64, separator="\n")
    docs = text_splitter.split_documents(documents)

    print("Creating vector database")
    t = time.time()
    db = Chroma.from_documents(docs, embeddings_fn, persist_directory="./data/rag_chroma")
    print(f"Database created in {time.time() - t:.2f} seconds")
    query = "What did shadowcon do?"
    docs = db.similarity_search(query)
    print(f"query: {query}\n\nFound:\n")
    print(docs[0].page_content)
