
# Summarizzler: The Final Boss 🦜

Welcome to **Summarizzler: The Final Boss** – a fun and dynamic app that helps you summarize or query content from URLs. Whether you're looking to quickly summarize an article, dive deep into YouTube video transcripts, or ask specific questions about the content, Summarizzler is your trusty companion for all things web content!

## Features

- **Summarize Content:** Instantly generate a concise summary of any URL's content.
- **Query Extracted Text:** Ask specific questions about the extracted text and get relevant answers, powered by AI!
- **YouTube Video Transcript:** Extract and summarize the transcript from any YouTube video or ask questions based on the video content.

## Getting Started 🚀

### Prerequisites

Before you can start using **Summarizzler: The Final Boss**, you need to make sure you have the following installed:

- Python 3.x
- Streamlit
- LangChain
- HuggingFace Transformers
- FAISS
- Groq

You can install these dependencies via pip. Here’s the quick-start guide to set everything up:

```bash
pip install streamlit langchain langchain_groq langchain_community huggingface-hub faiss-cpu beautifulsoup4 requests python-dotenv
```

You will also need API keys for **Groq** and **HuggingFace**. Create a `.env` file in the root directory and add the following:

```plaintext
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### Running the App 🎮

1. Clone this repo to your local machine:
   ```bash
   git clone https://github.com/yourusername/summarizzler.git
   cd summarizzler
   ```

2. Run the app with Streamlit:
   ```bash
   streamlit run app.py
   ```

   You’re all set to go! Open the app in your browser and start summarizing or querying content.

## How It Works 🧙🏻

### 1. **Enter a URL:**
   Paste a URL (article, blog, YouTube video) into the app.

### 2. **Choose Your Operation:**
   - **Summarize Content:** The app will extract the content and generate a short summary.
   - **Query Extracted Text:** Ask the app a question about the content, and it will give you the best answer it can find!
   - **YouTube Video Transcript:** The app can extract and summarize the transcript from YouTube videos or answer questions based on the video content.

### 3. **Under the Hood:**
   The app uses the mighty **ChatGroq** for language modeling, **FAISS** for fast similarity search, and **HuggingFace embeddings** to understand the content better. It can also pull in content from YouTube videos!

### 4. **A Progress Bar of Glory:**
   Watch as the app processes your URL and generates results with a fun progress bar to keep you entertained.

## Example Usage 🤖

### Summarize an article:

- Enter the URL of an article.
- Select **Summarize Content**.
- Get a neat summary in seconds!

### Ask questions:

- Enter the URL of any content.
- Choose **Query Extracted Text**.
- Ask a question, and let Summarizzler fetch the relevant info for you!

### Summarize or query YouTube videos:

- Enter the URL of a YouTube video.
- Choose **Summarize Content** or **Query Extracted Text**.
- Summarize the entire video transcript or ask questions based on it!

## Contributing

Have a fun idea or want to improve the app? Feel free to fork the repo and send in a pull request! We'd love to see your contributions.

## License

This project is licensed under the MIT License.

## Fun Fact 🎉

Did you know the app uses the **LLama-3.3-70b-specdec** model for generating text? It’s like having a digital wizard help you with your content. ⚡

---

**Summarizzler** – Because every URL deserves a little magic! ✨
