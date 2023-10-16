from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
                                 
llm = Ollama(model="llama2", 
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

llm("Tell me 5 facts about Roman history:")