import os
import time
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from theme import CustomTheme


path_modulhandbuch = "./dokumente"
path_persist = os.path.join(path_modulhandbuch, "persist")

Settings.llm = OpenAI(temperature=0.1, model="gpt-4o-mini")

if not os.path.exists(path_persist):
    documents = SimpleDirectoryReader("./dokumente/").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=path_persist)
else:
    storage_context = StorageContext.from_defaults(persist_dir=path_persist)
    index = load_index_from_storage(storage_context)

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given only this information and without using ur general knowledge, please answer in german and address with 'du': {query_str}\n"
)
qa_template = PromptTemplate(template)
query_engine = index.as_query_engine(
    streaming=True, text_qa_template=qa_template)


def response(message, history):
    # Logic to determine if the chatbot should ask follow-up questions
    if "wear" in message.lower() or "anziehen" in message.lower():
        # Respond with a sustainability-focused follow-up question
        follow_up = "Was hast du bereits in deinem Kleiderschrank? Vielleicht können wir etwas kombinieren, anstatt etwas Neues zu kaufen. 😊"
        yield follow_up
    else:
        # Standard response from the query engine
        streaming_response = query_engine.query(message)

        answer = ""
        for text in streaming_response.response_gen:
            time.sleep(0.05)
            answer += text
            yield answer


theme = CustomTheme()


# def main():
#     chatbot = gr.Chatbot(
#         value=[{"role": "assistant", "content": "How can I help you today?"}],
#         type="messages",
#         show_label=False,
#         avatar_images=("./avatar_images/human.png",
#                        "./avatar_images/robot.png"),
#         elem_id="CHATBOT"
#     )

#     with gr.Blocks() as demo:
#         with gr.Row():
#             with gr.Column():
#                 textbox1 = gr.Textbox()
#                 textbox2 = gr.Textbox()
#             with gr.Column():
#                 chatinterface = gr.ChatInterface(
#                     fn=response,
#                     chatbot=chatbot,
#                     type="messages",
#                     theme=theme,
#                     css_paths="./styles.css"
#                 )

#     demo.launch(inbrowser=True)


# if __name__ == "__main__":
#     main()

def main():
    chatbot = gr.Chatbot(
        value=[{"role": "assistant",
                "content": "Hey! Was steht heute an? Brauchst du Outfit-Ideen oder Styling-Tipps?"}],
        type="messages",
        show_label=False,
        avatar_images=("./avatar_images/stylemate_chatbot.jpg",
                       "./avatar_images/stylemate_chatbot.jpg"),
        elem_id="CHATBOT"
    )

    chatinterface = gr.ChatInterface(
        fn=response,
        chatbot=chatbot,
        type="messages",
        theme=theme,
        css_paths="./styles.css"
    )

    chatinterface.launch(inbrowser=True)


if __name__ == "__main__":
    main()
