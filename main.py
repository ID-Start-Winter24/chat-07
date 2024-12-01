import os
import time
import base64
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

background_path = os.path.join("background", "closet.png")
with open("background/closet.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
custom_css = f"""
.gradio-container {{
 background: url("data:image/png;base64,{encoded_string}") !important;
 background-size: cover !important;
 background-position: center !important;
 max-width: 100% !important;
 height: auto !important;
}}
"""


def response(message, history):
    import random

    # Keywords indicating dissatisfaction with any outfit
    negative_outfit_phrases = ["hässlich", "ugly", "schlecht",
                               "nicht gut", "nicht schön", "grässlich", "furchtbar", "katastrophe"]

    if any(phrase in message.lower() for phrase in negative_outfit_phrases):
        uplifting_responses = [
            "Es tut mir leid, dass du dich gerade so fühlst. Lass uns zusammen schauen, wie wir dein Outfit aufwerten können – vielleicht mit Accessoires oder einem neuen Styling-Twist! 😊",
            "Mode ist, wie du dich darin fühlst – nicht nur das Kleidungsstück selbst. Ich bin sicher, wir finden etwas, das dich zum Strahlen bringt! 💖",
            "Manchmal machen kleine Details einen großen Unterschied. Vielleicht können wir dein Outfit mit einem Gürtel, einer Jacke oder Schmuck aufpeppen? Soll ich dir helfen? 🌟",
            "Dein Stil ist einzigartig, und das ist etwas Besonderes. Wenn du magst, können wir das Outfit so anpassen, dass es sich mehr wie 'du' anfühlt!",
            "Wir alle haben Tage, an denen wir uns unsicher fühlen. Aber dein Outfit hat Potenzial! Lass uns gemeinsam überlegen, was dir daran gefallen könnte oder wie wir es optimieren können. 💡"
        ]
        yield random.choice(uplifting_responses)
    elif "anziehen" in message.lower() or "wear" in message.lower():
        # Sustainability-focused follow-up
        follow_up = "Was hast du bereits in deinem Kleiderschrank? Vielleicht können wir etwas kombinieren, anstatt etwas Neues zu kaufen. 😊"
        yield follow_up
    else:
        # Standard query engine response
        streaming_response = query_engine.query(message)

        answer = ""
        for text in streaming_response.response_gen:
            time.sleep(0.05)
            answer += text
            yield answer


theme = CustomTheme()


def main():
    chatbot = gr.Chatbot(
        value=[{"role": "assistant",
                "content": "Hey! Was steht heute an? Brauchst du Outfit-Ideen oder Styling-Tipps?"}],
        type="messages",
        show_label=False,
        avatar_images=("./avatar_images/avatar-person.jpeg",
                       "./avatar_images/avatar-chat.png"),
        elem_id="CHATBOT"
    )

    with open("background_new.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    custom_css = f"""
    .gradio-container {{
        background: url("data:image/png;base64,{encoded_string}") !important;
        background-size: cover !important;
        background-position: center !important;
        max-width: 100% !important;
        height: auto !important;
    }}"""

    chatinterface = gr.ChatInterface(
        fn=response,
        chatbot=chatbot,
        type="messages",
        theme=theme,
        css=custom_css,
        css_paths="./styles.css"
    )

    chatinterface.launch(inbrowser=True)


if __name__ == "__main__":
    main()
