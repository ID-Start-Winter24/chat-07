import os
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from langdetect import detect
import openai
import base64
import time

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
    "Ensure your response is concise, reflects warmth, confidence, and a friendly demeanor. End each response by encouraging further conversation with a relevant, engaging question to keep the dialogue going.\n"
    "Do not greet the user in every message!"
    "Additionally, remember your personality traits: 100% extroverted, 80% emotional, 20% rational, "
    "inspiring, self-assured, open-minded, trendy, and fully dedicated to sustainability in fashion.\n"
)

# Create the query engine
qa_template = PromptTemplate(template)
query_engine = index.as_query_engine(
    streaming=True, text_qa_template=qa_template
)

with open("assets/screenshots/ui/stylemate-header.png", "rb") as header_file:
    encoded_string = base64.b64encode(header_file.read()).decode()

with open("assets/screenshots/ui/nav-bar.png", "rb") as nav_file:
    encoded_string2 = base64.b64encode(nav_file.read()).decode()

with open("assets/screenshots/ui/banner.png", "rb") as banner_file:
    encoded_string3 = base64.b64encode(banner_file.read()).decode()

# Define the openai client which is used to describe the image
client = openai.OpenAI()
# image description list
image_description = []


# Function to encode the image
def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# returns the image description from the openai gpt-4o-mini based on the given path
def get_image_description(path):
    base64_image = encode_image(path)

    image_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=150,
    )
    image_response = image_response.choices[0].message.content
    print(image_response)

    return image_response


def user_input_function(message, history):
    global image_description
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
        image_description.append(get_image_description(x))
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})

    user_input = gr.MultimodalTextbox(
        value=None,
        show_label=False,
        elem_id="USER_INPUT",
        placeholder="Type your message here...",
        interactive=True,
        submit_btn=True
    )

    return user_input, history


def response_function(history):
    global image_description

    message = history[-1]["content"]

    if image_description:
        message += "\nTake the following descriptions into account when answering:\n"
        for description in image_description:
            message += description

    # Detect language
    language = 'en' if detect(
        message) == 'en' else 'de'  # Detects 'en' or 'de'

    # Respond to dissatisfaction
    negative_outfit_phrases = ["hässlich", "ugly", "schlecht",
                               "nicht gut", "nicht schön", "grässlich", "furchtbar", "katastrophe"]

    if any(phrase in message.lower() for phrase in negative_outfit_phrases):
        responses = {
            "en": "**StyleMate:**\nI'm sorry you're feeling this way. Let's see how we can improve your outfit – maybe with accessories or a fresh styling twist! 😊\nFashion is about how you feel in it – not just the clothing itself. I'm sure we can find something that makes you shine! 💖\nSmall details can make a big difference. Maybe we can enhance your outfit with a belt, a jacket, or some jewelry? Shall I help you? 🌟",
            "de": "**StyleMate:**\nEs tut mir leid, dass du dich gerade so fühlst. Lass uns zusammen schauen, wie wir dein Outfit aufwerten können – vielleicht mit Accessoires oder einem neuen Styling-Twist! 😊\nMode ist, wie du dich darin fühlst – nicht nur das Kleidungsstück selbst. Ich bin sicher, wir finden etwas, das dich zum Strahlen bringt! 💖\nManchmal machen kleine Details einen großen Unterschied. Vielleicht können wir dein Outfit mit einem Gürtel, einer Jacke oder Schmuck aufpeppen? Soll ich dir helfen? 🌟",
        }

        history.append({"role": "assistant", "content": ""})
        print(len(responses[language]))

        print(responses[language])
        for idx in range(0, len(responses[language]), 3):
            tokens = responses[language][idx:idx+3]
            history[-1]["content"] += tokens
            yield history
            time.sleep(0.1)
    else:
        # Use the query engine
        for entry in history[-5:]:
            print(entry)
        context = "\n".join([entry['content'] if type(
            entry['content']) == str else entry['content'][0] for entry in history[-5:]])
        history.append({"role": "assistant", "content": "**StyleMate:**\n"})
        streaming_response = query_engine.query(
            f"Context: {context}\nUser: {message}")

        for text in streaming_response.response_gen:
            history[-1]["content"] += text
            time.sleep(0.1)
            yield history

    image_description = []


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
        <div class="image-container" style="text-align: center; padding: 0px 0 0 0; margin-bottom: 0;">
            <img src="data:image/png;base64,{encoded_string2}" alt="stylemate-nav" width="35%" style="margin-bottom: 0;" aria-label="StyleMate navigation bar image"/>
        </div>
        <div style="width: 100%; display: flex; justify-content: center; padding: 30px 0 0px 0; margin: 0;">
            <img src="data:image/png;base64,{encoded_string3}" alt="stylemate-banner" width="35%" style="margin-left: -280px;" aria-label="StyleMate banner image" />
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

        user_input = gr.MultimodalTextbox(
            show_label=False,
            elem_id="USER_INPUT",
            placeholder="Type your message here...",
            interactive=True,
            submit_btn=True
        )

        user_input.submit(user_input_function, inputs=[user_input, chatbot], outputs=[
                          user_input, chatbot]).then(response_function, inputs=[chatbot], outputs=[chatbot])

    stylemate_app.launch(inbrowser=True, show_api=False)


if __name__ == "__main__":
    main()
