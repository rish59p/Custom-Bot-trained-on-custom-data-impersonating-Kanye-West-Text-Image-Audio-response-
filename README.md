# YeBot - Kanye West-inspired AI Bot

YeBot is an AI-powered chatbot inspired by the personality of Kanye West. It responds to user texts based on the custom + reddit data it has been trained on, generates images using Open DALL·E V1.1, outputs Jingle lyrics using Gemini-Pro, and creates an audio file using BART Bark. The bot is designed to be a bit offensive and sarcastic, capturing the essence of Kanye West's outspoken and creative personality.

## Project Overview

### Motivation

YeBot was created as an exploration into the world of AI language models and creative content generation and to work around with vectorized embeddings. The goal was to develop a chatbot with a unique personality, drawing inspiration from the renowned artist Kanye West. The project combines different AI technologies to provide users with an interactive and entertaining experience.

### Technologies Used

- **Vectorised Embeddings**: Used a custom dataset based on reddit sarcastic comments and generated the embeddings with the help of GoogleGenAIEmbeddings model to feed the embeddings into the bot so that it could reply accordingly

- **LangChain** : Used to connect to different APIs

- **Google Gemini Pro API**: Used to create custom data embeddings for YeBot, allowing it to generate responses based on the unique style of Kanye West. Used to generate prompts from results for feeding into Open DALL·E for image generation and also used to generate lyrical prompts based on the response to feed into bark.

- **Open DALL·E V1.1 (Hugging Face API)**: Employs the power of Open DALL·E to generate visual content based on prompts generated by Gemini in response to user inputs.

- **BART (Hugging Face API)**: Utilized for two purposes - generating Jingle lyrics based on YeBot's responses and creating an audio file using the BART Bark function.

- **Streamlit**: The entire system is deployed using Streamlit, providing an easy-to-use interface for users to interact with YeBot.

### How YeBot Works

1. **Text Responses**: YeBot processes user inputs using data embeddings generated by Google Gemini Pro. It responds with texts that reflect Kanye West's personality.

2. **Image Generation**: Open DALL·E V1.1 is employed to generate visual content based on prompts generated by Gemini in response to user inputs. Users can see creative interpretations of their conversations.

3. **Jingle Lyrics**: YeBot generates Jingle lyrics based on its responses using the BART model. The lyrics are designed to be in line with Kanye West's style, adding a musical touch to the interactions.

4. **Audio Output**: The generated Jingle lyrics serve as a prompt for BART Bark, creating an audio file that users can listen to, providing a multi-sensory experience.


https://github.com/rish59p/Custom-Bot-trained-on-custom-data-impersonating-Kanye-West-Text-Image-Audio-response-/assets/63728926/29311b83-b7f3-426a-8e79-de1d5f19ffcb


## Setup

1. **Dependencies Installation**: Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

2. **API Keys**: Obtain API keys for Google Gemini Pro and Hugging Face APIs. Set these keys as environment variables in your project or in a `.env` file.

    ```bash
    export GEMINI_API_KEY="your_gemini_api_key"
    export HUGGINGFACE_API_KEY="your_huggingface_api_key"
    ```

3. **Run the Application**: Execute the following command to run the Streamlit app:

    ```bash
    streamlit run app.py
    ```
