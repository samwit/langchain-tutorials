
from langchain.retrievers import MultiQueryRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

db = Chroma(persist_directory="./db-hormozi",
            embedding_function=embeddings)

# # Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)

# # Create Ollama language model - Gemma 2
local_llm = 'gemma2'

llm = ChatOllama(model=local_llm,
                 keep_alive="3h", 
                 max_tokens=512,  
                 temperature=0)

# Create prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(template)

# Function to print the prompt for a runnable assign
def print_prompt(input_dict):
    formatted_prompt = prompt.format(**input_dict)
    print("Generated Prompt:")
    print(formatted_prompt)
    print("-" * 50)
    return input_dict

# Function to print and pass through the formatted prompt - string output
def print_and_pass_prompt(formatted_prompt):
    print("Generated Prompt:")
    print(formatted_prompt)
    print("-" * 50)
    return formatted_prompt


# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | print_and_pass_prompt
    | llm
)

# Function to ask questions
def ask_question(question):
    print("Answer:", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
        # print("\nFull answer received.\n")




# # pip install langchain-chroma
