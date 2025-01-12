import os
import time
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from langdetect import detect

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

# Custom CSS for chatbot interface
custom_css = """
.gradio-container {
    width: 100% !important;
    height: 100vh !important; /* Full screen height */
    display: flex;
    flex-direction: column;
}

#CHATBOT {
    flex-grow: 1; /* Ensures the chatbot grows to fill available space */
    overflow-y: auto; /* Allows scrolling if content exceeds screen height */
}
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
                "Your style is unique, and that's special. If you'd like, we can adjust the outfit so it feels more like 'you'!",
                "We all have days when we feel uncertain. But your outfit has potential! Let’s think about what you might like or how we can tweak it. 💡"
            ],
            "de": [
                "Es tut mir leid, dass du dich gerade so fühlst. Lass uns zusammen schauen, wie wir dein Outfit aufwerten können – vielleicht mit Accessoires oder einem neuen Styling-Twist! 😊",
                "Mode ist, wie du dich darin fühlst – nicht nur das Kleidungsstück selbst. Ich bin sicher, wir finden etwas, das dich zum Strahlen bringt! 💖",
                "Manchmal machen kleine Details einen großen Unterschied. Vielleicht können wir dein Outfit mit einem Gürtel, einer Jacke oder Schmuck aufpeppen? Soll ich dir helfen? 🌟",
                "Dein Stil ist einzigartig, und das ist etwas Besonderes. Wenn du magst, können wir das Outfit so anpassen, dass es sich mehr wie 'du' anfühlt!",
                "Wir alle haben Tage, an denen wir uns unsicher fühlen. Aber dein Outfit hat Potenzial! Lass uns gemeinsam überlegen, was dir daran gefallen könnte oder wie wir es optimieren können. 💡"
            ]
        }
        # Split responses into sentences
        for sentence in random.choice(responses[language]).split(". "):
            yield sentence.strip() + "."
            time.sleep(1.0)  # Add delay for typing effect
    else:
        # Add the new message to history
        history.append({"role": "user", "content": message})

        # Construct the context by joining past messages in history
        # Use last 5 messages as context
        context = "\n".join([entry['content'] for entry in history[-5:]])

        # Use query engine for standard responses, passing the constructed context
        streaming_response = query_engine.query(
            f"Context: {context}\nUser: {message}")

        answer = ""
        for text in streaming_response.response_gen:
            time.sleep(0.1)  # Add delay for each chunk
            answer += text
            yield answer

        # Add the assistant's response to history
        history.append({"role": "assistant", "content": answer})


# Create the chatbot interface
theme = CustomTheme()


def main():
    chatbot = gr.Chatbot(
        value=[{"role": "assistant",
                "content": (
                    "Hey, schön, dass du hier bist! Lust auf frische Outfit-Ideen oder stylische Tipps? "
                    "Ich helfe dir gern weiter. Übrigens, wenn du lieber Englisch sprechen möchtest, "
                    "feel free to ask in English anytime!"
                )}],
        type="messages",
        show_label=False,
        avatar_images=("./avatar_images/avatar-person.jpeg",
                       "./avatar_images/avatar-bot.png"),
        elem_id="CHATBOT"
    )

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
