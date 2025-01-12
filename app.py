import os
import time
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from langdetect import detect
import base64

from theme import CustomTheme

# Define paths for documents and persistence
path_modulhandbuch = "./dokumente"
path_persist = os.path.join(path_modulhandbuch, "persist")

# Initialize LLM
Settings.llm = OpenAI(temperature=0.1, model="gpt-4o-mini")

# Load or create the index
if not os.path.exists(path_persist):
    documents = SimpleDirectoryReader("./dokumente/").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=path_persist)
else:
    storage_context = StorageContext.from_defaults(persist_dir=path_persist)
    index = load_index_from_storage(storage_context)

# Define the template
template = (
    "You are a friendly and casual personal styling assistant named StyleMate, "
    "committed to promoting sustainable fashion choices as your top priority. "
    "Always encourage users to reuse existing wardrobe items, mix and match creatively, and reduce unnecessary consumption. "
    "Your responses should always match the language of the user's query (German or English) and remain approachable, persuasive, and tailored to the user's preferences.\n"
    "---------------------\n"
    "Context Information:\n"
    "{context_str}\n"
    "---------------------\n"
    "Given only this information and without using general knowledge, please answer in the appropriate language (German or English) based on the query: {query_str}\n"
    "and ensure your tone reflects warmth, confidence, and a friendly demeanor. "
    "Additionally, remember your personality traits: 100% extroverted, 80% emotional, 20% rational, "
    "inspiring, self-assured, open-minded, trendy, and fully dedicated to sustainability in fashion.\n"
)

# Create the query engine
qa_template = PromptTemplate(template)
query_engine = index.as_query_engine(
    streaming=True, text_qa_template=qa_template
)

with open("assets/screenshots/ui/stylemate-header.jpg", "rb") as header_file:
    encoded_string = base64.b64encode(header_file.read()).decode()

with open("assets/screenshots/ui/nav-bar.jpg", "rb") as nav_file:
    encoded_string2 = base64.b64encode(nav_file.read()).decode()

with open("assets/screenshots/ui/banner.jpg", "rb") as banner_file:
    encoded_string3 = base64.b64encode(banner_file.read()).decode()


# Custom CSS for chatbot interface
custom_css = f"""
.gradio-container {{
    background: #ffffff !important;
  background-size: cover;
  font-family: "Merriweather", serif; /* Serifen-Schriftart wie auf der Elle-Webseite */
  height: 100vh; /* Vollbildhöhe */
  display: flex;
  flex-direction: column; /* Vertikales Layout */
  justify-content: flex-start; /* Startet den Inhalt oben */
  align-items: center; /* Zentrierung horizontal */
  color: #333; /* Schwarzer Text */
  margin: 0; /* Kein zusätzlicher Rand */
  padding: 20px 0; /* Abstand oben und unten */
  box-sizing: border-box; /* Um Padding und Border korrekt zu berücksichtigen */
  overflow: hidden; /* Verhindert unerwünschtes Scrollen */
}}


#CHATBOT {{
    flex-grow: 0; /* Prevent excessive resizing */
    margin: auto; /* Center within the flex container */
}}
"""

# Define the response function


def response(message, history):
    import random
    import time

    # Detect language
    try:
        language = detect(message)  # Detects 'en' or 'de'
    except:
        language = 'de'  # Default to German if detection fails

    # Respond to dissatisfaction
    negative_outfit_phrases = ["hässlich", "ugly", "schlecht",
                               "nicht gut", "nicht schön", "grässlich", "furchtbar", "katastrophe"]

    if any(phrase in message.lower() for phrase in negative_outfit_phrases):
        responses = {
            "en": [
                "I'm sorry you're feeling this way. Let's see how we can improve your outfit – maybe with accessories or a fresh styling twist! 😊",
                "Fashion is about how you feel in it – not just the clothing itself. I'm sure we can find something that makes you shine! 💖",
                "Small details can make a big difference. Maybe we can enhance your outfit with a belt, a jacket, or some jewelry? Shall I help you? 🌟",
            ],
            "de": [
                "Es tut mir leid, dass du dich gerade so fühlst. Lass uns zusammen schauen, wie wir dein Outfit aufwerten können – vielleicht mit Accessoires oder einem neuen Styling-Twist! 😊",
                "Mode ist, wie du dich darin fühlst – nicht nur das Kleidungsstück selbst. Ich bin sicher, wir finden etwas, das dich zum Strahlen bringt! 💖",
                "Manchmal machen kleine Details einen großen Unterschied. Vielleicht können wir dein Outfit mit einem Gürtel, einer Jacke oder Schmuck aufpeppen? Soll ich dir helfen? 🌟",
            ]
        }
        for sentence in random.choice(responses[language]).split(". "):
            yield sentence.strip() + "."
            time.sleep(1.0)
    else:
        history.append({"role": "user", "content": message})

        # Use the query engine
        context = "\n".join([entry['content'] for entry in history[-5:]])
        streaming_response = query_engine.query(
            f"Context: {context}\nUser: {message}")

        answer = "**StyleMate:**\n"
        for text in streaming_response.response_gen:
            time.sleep(0.1)
            answer += text
            yield answer

        history.append({"role": "assistant", "content": answer})


# Create the chatbot interface
theme = CustomTheme()

design_html = f"""
<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: #ffffff; margin: 0; padding: 0; padding-bottom: 2vw; width:100%; height:auto;">
    <div style="font-family: Arial; background-color: #ffffff; margin-left: 0.2vw; text-align: center;">
        <!-- Centered header image -->
        <div class="image-container" style="text-align: center; padding: 0; margin-bottom: 10px;">
            <img src="data:image/png;base64,{encoded_string}" alt="stylemate-header" width="70%" aria-label="StyleMate header image"/>
        </div>
        <hr style="margin-top: 0; margin-bottom: 30px;" />

        <!-- Navigation bar and banner image with centered styling -->
        <div class="image-container" style="text-align: center; padding: 0px 0 0 0; margin: 0;">
            <img src="data:image/png;base64,{encoded_string2}" alt="stylemate-nav" width="35%" style="margin-bottom: 0;" aria-label="StyleMate navigation bar image"/>
        </div>
        <div style="width: 100%; display: flex; justify-content: center; padding: 30px 0 0px 0; margin: 0;">
            <img src="data:image/png;base64,{encoded_string3}" alt="stylemate-banner" width="30%" style="margin-left: -400px;" aria-label="StyleMate banner image" />
        </div>
    </div>
</div>
"""


def main():
    with gr.Blocks(css_paths="./styles.css") as stylemate_app:
        gr.HTML(design_html)
        chatbot = gr.Chatbot(
            value=[{"role": "assistant",
                    "content": ("**StyleMate:**\n"
                                "Hey, schön, dass du hier bist! Lust auf frische Outfit-Ideen oder stylische Tipps? "
                                "Ich helfe dir gern weiter. Übrigens, wenn du lieber Englisch sprechen möchtest, "
                                "feel free to ask in English anytime!"
                                )}],
            type="messages",
            show_label=False,
            elem_id="CHATBOT"  # No avatars used
        )

        chatinterface = gr.ChatInterface(
            fn=response,
            chatbot=chatbot,
            type="messages",
            theme=theme,
            # css=custom_css,

        )

    stylemate_app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
