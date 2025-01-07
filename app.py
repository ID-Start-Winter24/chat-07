import os
import time
import base64
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from langdetect import detect

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
    width: 100% !important;
    height: 100vh !important; /* Full screen height */
    display: flex;
    flex-direction: column;
}}

#CHATBOT {{
    flex-grow: 1; /* Ensures the chatbot grows to fill available space */
    overflow-y: auto; /* Allows scrolling if content exceeds screen height */
}}


"""


def response(message, history):
    import random

    # Keywords indicating dissatisfaction with any outfit
    negative_outfit_phrases = ["hässlich", "ugly", "schlecht",
                               "nicht gut", "nicht schön", "grässlich", "furchtbar", "katastrophe"]

    # Keywords indicating interest in new outfits
    outfit_request_phrases = [
        "outfit", "empfehlung", "style", "anziehen", "kombination", "kleidung",
        "modetipps", "styling", "look", "neu kombinieren", "outfit ideen", "kleiderideen",
        "outfit ideas", "clothing suggestions", "style ideas", "fashion tips", "mix and match",
        "wardrobe suggestions", "what to wear", "fashion styling"
    ]

    # Keywords indicating interest in buying new clothes
    buy_new_phrases = [
        "neu kaufen", "buy new", "neues kaufen", "new outfit", "neue kleidung kaufen",
        "kaufen", "shoppen", "shopping", "neue sachen kaufen", "mode kaufen", "neues kleidungsstück",
        "neue mode", "buy clothes", "buy fashion", "new clothes", "shopping spree", "buy a new look"
    ]

    # Detect language (using langdetect for better accuracy)
    try:
        language = detect(message)  # This will return either 'en' or 'de'
    except:
        language = 'de'  # Fallback to German if detection fails

    # Detecting if the user is dissatisfied with their outfit
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
        yield random.choice(uplifting_responses_en if language == 'en' else uplifting_responses_de)

    elif any(phrase in message.lower() for phrase in outfit_request_phrases):
        # Check if the user is requesting to buy new items
        if any(phrase in message.lower() for phrase in buy_new_phrases):
            # Proceed with suggesting new outfits
            new_outfit_responses_de = [
                "Verstehe, du möchtest also etwas Neues kaufen! Ich kann dir helfen, das perfekte Outfit zu finden. Was hast du im Kopf?",
                "Wenn du etwas Neues kaufen möchtest, lass uns schauen, was aktuell im Trend ist. Ich helfe dir, das perfekte Teil zu finden!",
                "Klar, neue Outfits sind immer spannend! Lass uns herausfinden, was du suchst und was zu deinem Stil passt."
            ]
            new_outfit_responses_en = [
                "Got it, you'd like to buy something new! I can help you find the perfect outfit. What do you have in mind?",
                "If you're looking to buy something new, let's see what's trending. I'll help you find the perfect piece!",
                "Sure, new outfits are always exciting! Let's figure out what you're looking for and what fits your style."
            ]
            yield random.choice(new_outfit_responses_en if language == 'en' else new_outfit_responses_de)

        else:
            # Ask the user about their current wardrobe only if they are not requesting to buy new items
            wardrobe_responses_de = [
                "Bevor wir etwas Neues kaufen, was hast du bereits in deinem Kleiderschrank? Vielleicht können wir damit ein tolles Outfit zusammenstellen!",
                "Ich schlag vor, wir schauen zuerst in deinem Kleiderschrank nach. Was hast du schon, mit dem wir etwas zusammenstellen können?",
                "Es ist immer gut, zuerst zu schauen, was du schon hast. Hast du vielleicht ein Lieblingsstück, mit dem wir beginnen können?"
            ]
            wardrobe_responses_en = [
                "Before we buy anything new, what do you already have in your wardrobe? Maybe we can put together a great outfit with that!",
                "I suggest we take a look at what you already have in your wardrobe first. What do you have that we can work with?",
                "It's always good to check what you already own first. Do you have a favorite piece we could start with?"
            ]
            yield random.choice(wardrobe_responses_en if language == 'en' else wardrobe_responses_de)

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
