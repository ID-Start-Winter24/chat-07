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
    "Given only this information and without using general knowledge, please answer in the appropriate language (German or English) based on the query: {query_str}\n"
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

    # Keywords indicating interest in new outfits
    outfit_request_phrases = ["outfit", "empfehlung", "style", "anziehen"]
    buy_new_phrases = ["neu kaufen", "buy new", "neues kaufen", "new outfit"]

    # Detect language (simple check for English or German based on keywords)
    is_english = any(word in message.lower()
                     for word in ["outfit", "wear", "style", "buy", "new", "ugly"])

    if any(phrase in message.lower() for phrase in negative_outfit_phrases):
        uplifting_responses_de = [
            "Es tut mir leid, dass du dich gerade so fühlst. Lass uns zusammen schauen, wie wir dein Outfit aufwerten können – vielleicht mit Accessoires oder einem neuen Styling-Twist! 😊",
            "Mode ist, wie du dich darin fühlst – nicht nur das Kleidungsstück selbst. Ich bin sicher, wir finden etwas, das dich zum Strahlen bringt! 💖",
            "Manchmal machen kleine Details einen großen Unterschied. Vielleicht können wir dein Outfit mit einem Gürtel, einer Jacke oder Schmuck aufpeppen? Soll ich dir helfen? 🌟",
            "Dein Stil ist einzigartig, und das ist etwas Besonderes. Wenn du magst, können wir das Outfit so anpassen, dass es sich mehr wie 'du' anfühlt!",
            "Wir alle haben Tage, an denen wir uns unsicher fühlen. Aber dein Outfit hat Potenzial! Lass uns gemeinsam überlegen, was dir daran gefallen könnte oder wie wir es optimieren können. 💡"
        ]
        uplifting_responses_en = [
            "I'm sorry you're feeling this way. Let's see how we can improve your outfit – maybe with accessories or a fresh styling twist! 😊",
            "Fashion is about how you feel in it – not just the clothing itself. I'm sure we can find something that makes you shine! 💖",
            "Small details can make a big difference. Maybe we can enhance your outfit with a belt, a jacket, or some jewelry? Shall I help you? 🌟",
            "Your style is unique, and that's special. If you'd like, we can adjust the outfit so it feels more like 'you'!",
            "We all have days when we feel uncertain. But your outfit has potential! Let’s think about what you might like or how we can tweak it. 💡"
        ]
        yield random.choice(uplifting_responses_en if is_english else uplifting_responses_de)

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
                       "./avatar_images/avatar-bot.png"),
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
