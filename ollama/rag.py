# Load web page
import argparse

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and store
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings # We can also try Ollama embeddings

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():
    parser = argparse.ArgumentParser(description='Filter out URL argument.')
    parser.add_argument('--url', type=str, default='http://example.com', required=True, help='The URL to filter out.')

    args = parser.parse_args()
    url = args.url
    print(f"using URL: {url}")

    loader = WebBaseLoader(url)
    data = loader.load()

    # Split into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=GPT4AllEmbeddings())

    # Retrieve
    # question = "What are the latest headlines on {url}?"
    # docs = vectorstore.similarity_search(question)

    print(f"Loaded {len(data)} documents")
    # print(f"Retrieved {len(docs)} documents")

    # RAG prompt
    from langchain import hub
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


    # LLM
    llm = Ollama(model="llama2",
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    print(f"Loaded LLM model {llm.model}")

    # QA chain
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},

    )

    # Ask a question
    question = f"What are the latest headlines on {url}?"
    result = qa_chain({"query": question})

    # print(result)
    


if __name__ == "__main__":
    main()








