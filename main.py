# import libraries
import os  # for file handling and controlling timing
import time  # for file handling and controlling timing
# gradio: a framework to create user-friendly web interfaces for ML modles.
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, PromptTemplate

"""
VectorStoreIndex: used to create a searchable vector index of documents.
SimpleDirectoryReader: reads and loads documents from a specified directory.
StorageContext and load_index_from_storage: manage storage and loading of indexed data.
"""


# OpenAI: initializes the OpenAI LLM with specific parameters.
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


# directory where the documents (like a handbook) are stored.
path_modulhandbuch = "./modulhandbuch"
# create a sub-directory, persist, in path_modulhandbuch to store the indexed data for persistent access.
path_persist = os.path.join(path_modulhandbuch, "persist")


# This initializes the LLM model (gpt-4o-mini) with temperature=0.1, which controls the answer’s randomness
# temperature=0.0 kwould result in very straightforward answers, while 0.1 introduces slight variablity
Settings.llm = OpenAI(temperature=0.1, model="gpt-4o-mini")

if not os.path.exists(path_persist):  # if persist doesn't exist
    # load documentss from ("./modulhandbuch") using SimpleDirectoryReader
    documents = SimpleDirectoryReader("./modulhandbuch/").load_data()
    # creates a vector index of documents with VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents)
    # Saves (persists) the index in path_persist for future use.
    index.storage_context.persist(persist_dir=path_persist)
else:  # if persist exists
    # Loads the stored index using StorageContext and load_index_from_storage.
    storage_context = StorageContext.from_defaults(persist_dir=path_persist)
    index = load_index_from_storage(storage_context)

"""
Defines the prompt template with placeholders {context_str} and {query_str}:

{context_str} holds the document context that was indexed.
{query_str} holds the question input.
This prompt setup ensures the model only uses the document data to answer questions without relying on external knowledge.
"""

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    # เราสามารถ add ได้ เช่น always answers in German and Shakespere's Style
    "Given only this information and without using ur general knowledge, please answer the question: {query_str}\n"
)
# initializes the prompt as a PromptTemplate
qa_template = PromptTemplate(template)
# query_engine: converts the indexed data (index) into a query engine with streaming=True, allowing answers to be generated as a continuous stream of text. It also uses qa_template to format responses.
query_engine = index.as_query_engine(
    streaming=True, text_qa_template=qa_template)


def response(message, history):
    streaming_response = query_engine.query(message)

    answer = ""
    for text in streaming_response.response_gen:
        # เผื่อไม่ให้คำตอบออกมาพร้อมกันทั้งหมด creating a type-effect
        time.sleep(0.05)
        answer += text
        yield answer  # yield give the result back to function


def main():
    # chatbot = gr.Chatbot(
    #     value=[{"role": "assistant", "content": "How can I help you today?"}], # ทำหน้าที่เป็น Assistant แล้วเริ่มคำถาม
    #     show_label=False,
    #     type="messages"
    #     avatar_images=("./avatar_images/human.png", "./avatar_images/robot.png") (ต้องสร้าง folder ใน ID-Bot)
    # )
    chatbot = gr.ChatInterface(
        fn=response,
        type="messages",
    )

    chatbot.launch(inbrowser=True)


if __name__ == "__main__":
    main()
