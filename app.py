import streamlit as st
import google.generativeai as genai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import requests
import io
from PIL import Image
from pydub import AudioSegment
from IPython.display import Audio
from io import BytesIO
from IPython.display import Audio
import base64


load_dotenv()

os.environ['GOOGLE_API_KEY'] = "Your_API_KEY"
GOOGLE_API_KEY = "Your_API_Key"
genai.configure(api_key = GOOGLE_API_KEY)
    
model = genai.GenerativeModel('gemini-pro')


loader = CSVLoader(file_path="data/newdatatest.csv")
documents = loader.load()

# print(documents[0])

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
print("embeddings created")
db = FAISS.from_documents(documents, embeddings)
print("embeddings done")

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

llm = ChatGoogleGenerativeAI(model="gemini-pro")


template = """
You are a sarcastic bot that tries to be like Kanye West. 
People will add their messages and you have to reply to them in a funny, maybe disrespectful ways.
Adhere to these rules while you generate a response:

1/ Response should be  similar  to the past best practies, 
in terms of  tone of voice, logical arguments, offence and other details
2/ If the best practice are irrelevant, then try to mimic the style of the best practice to message
3/ Do not violate google gemini-pro safety_ratings while being sarcastic

Below is a message I received from the people:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response

"""

image_template = """
You have to write me a prompt based on text that i give, the prompt needs to be describing an image that is funny and sarcastic
and related to the text that I give. The image must contain Kanye West, the image should depict kanye's vision when he describes 
the text(that i have given to you), the vision(and hence the image) must be abstract and funny
Try to adhere to safety ratings.
Below is the text that i receive:
{text}

Write the prompt to be fed to another AI to generate an image as mentioned above.
"""

audio_template = """
Generate a few lines of jingle based on the result that i give to you. Make a 4 line jingle like song prompt that i can 
sing, the lyrics should be higly related to result that i give. Add "♪" at the start and end of the prompt to make it sing.
here's the result:
{lyrical}

Write me a 4 line lyrics for the song as mentioned above.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

image_prompt = PromptTemplate(
    input_variables=["text"],
    template=image_template
)

audio_prompt = PromptTemplate(
    input_variables=["lyrical"],
    template=audio_template
)

chain = LLMChain(llm=llm, prompt=prompt)
chain2 = LLMChain(llm=llm, prompt=image_prompt)
chain3 = LLMChain(llm=llm, prompt=audio_prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice,
        safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        ]
    )
    return response

def generate_image_text(text):
    img_response = chain2.run(text=text,
        safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        ]
    )
    return img_response

def generate_audio_lyrics(lyrical):
    aud_response=chain3.run(lyrical=lyrical,
        safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        ]
    )
    return aud_response


API_URL = "https://api-inference.huggingface.co/models/dataautogpt3/OpenDalleV1.1"
headers = {"Authorization": "Bearer HF_API_Key"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def generate_image(response_text):
    image_bytes = query({"inputs": response_text})
    return image_bytes

    
AUDIO_API = "https://api-inference.huggingface.co/models/suno/bark-small"

def audioquery(payload):
	response = requests.post(AUDIO_API, headers=headers, json=payload)
	return response.content

# message = """
# Honestly this reminds me of Stein's;Gate.
# """

# print(message)

# response = generate_response(message)
# print(response)

def main():
    st.set_page_config(
        page_title="AllYeNeedToKnow", page_icon=":bird:")

    st.header("AllYeNeedToKnow :bird:")
    # st.write("Sample Audio Test:")
    # test_audio = audioquery({
    #         "inputs": "♪ Hey Ye this side! ♪",
    #     })
    # Audio(test_audio)
    # st.audio(test_audio, format="audio/wav")

    #audio_segment = AudioSegment.from_file(BytesIO(test_audio), format="mp3")

    #st.audio(audio_segment.export(format="mp3"), format="audio/mp3")

    st.image(Image.open(io.BytesIO(generate_image("Kanye west + ((AllYeNeedToKnow!)text logo:1),~*~dark~*~"))))

    message = st.text_area("Get some Ye wisdom, say something, I assume it would be stupid looking at you")

    if message:
        st.write("Man wasted my time thinking on this but now i have to...")


        # image_temp_bytes = generate_image("A vivid imaginary library, fusion of greek and modern art with Kanye West reading an album art type book thinking about")

        # image_temp=Image.open(io.BytesIO(image_temp_bytes))

        st.image(Image.open(io.BytesIO(generate_image("A vivid imaginary library, fusion of greek and modern art with Kanye West reading an album art type book thinking about"))))

        result = generate_response(message)

        image_text = generate_image_text(result)

        st.info(result)

        audio_text = generate_audio_lyrics(result)

        st.write("So here's an visual on what I think: ")

        # final_image_bytes = generate_image(image_text)   

        # final_image = Image.open(io.BytesIO(final_image_bytes))       

        st.image(Image.open(io.BytesIO(generate_image(image_text))))

        st.write("Wait lemme sing a song on this real quick: ")
        
        st.write(audio_text)

        audio_bytes = audioquery({
            "inputs": audio_text,
        })

        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))

        st.audio(audio_segment.export(format="mp3"), format="audio/mp3")

        st.write("What you scrolling this far for punk, show's over go home.")


if __name__ == '__main__':
    main()
