# 🤖 YouTube Chatbot using LangChain

An intelligent chatbot powered by LangChain and RAG (Retrieval Augmented Generation) that can answer questions about YouTube video content. Simply provide a YouTube URL, and chat with the video's transcript using AI!

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green?style=flat)](https://www.langchain.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🌟 Features

- 🎥 **YouTube Transcript Extraction**: Automatically fetches and processes video transcripts
- 🧠 **RAG Architecture**: Uses Retrieval Augmented Generation for accurate responses
- 💬 **Natural Conversation**: Chat naturally about video content
- 🔍 **Context-Aware**: Maintains conversation context for follow-up questions
- ⚡ **Fast Retrieval**: Vector-based semantic search for relevant information
- 📊 **Interactive Notebook**: Easy-to-use Jupyter notebook interface

## 📋 Table of Contents

- [How It Works](#-how-it-works)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🔧 How It Works

The chatbot uses a Retrieval Augmented Generation (RAG) pipeline:

1. **Transcript Extraction**: Downloads YouTube video transcript
2. **Text Chunking**: Splits transcript into manageable chunks
3. **Embedding Generation**: Converts text chunks into vector embeddings
4. **Vector Storage**: Stores embeddings in a vector database
5. **Query Processing**: Converts user questions into embeddings
6. **Semantic Search**: Finds most relevant chunks from the transcript
7. **Response Generation**: Uses LLM to generate answers based on retrieved context

```
YouTube URL → Transcript → Chunks → Embeddings → Vector DB
                                                      ↓
User Question → Query Embedding → Semantic Search → Context
                                                      ↓
                                            LLM → Answer
```

## 📋 Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- API keys for:
  - OpenAI API (or other LLM provider)
  - YouTube Data API (optional, for enhanced features)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Devatva24/Youtube_Chatbot.git
cd Youtube_Chatbot
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install langchain
pip install langchain-openai
pip install youtube-transcript-api
pip install chromadb
pip install tiktoken
pip install openai
pip install jupyter
```

Or create a `requirements.txt`:

```txt
langchain>=0.1.0
langchain-openai>=0.0.2
youtube-transcript-api>=0.6.1
chromadb>=0.4.0
tiktoken>=0.5.1
openai>=1.0.0
jupyter>=1.0.0
```

Then install:
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Or set environment variables directly in the notebook:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

## 💻 Usage

### Quick Start

1. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

2. **Open the Notebook**

Navigate to `rag_using_langchain.ipynb`

3. **Run the Cells**

Follow these steps in the notebook:

```python
# Step 1: Import libraries
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Step 2: Load YouTube video
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()

# Step 3: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(transcript)

# Step 4: Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# Step 5: Create QA chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Step 6: Ask questions!
question = "What is the main topic of this video?"
answer = qa_chain.run(question)
print(answer)
```

## 🏗️ Architecture

### Components

1. **Document Loader**
   - `YoutubeLoader`: Fetches video transcripts from YouTube

2. **Text Splitter**
   - `RecursiveCharacterTextSplitter`: Intelligently splits text into chunks
   - Chunk size: 1000 characters
   - Chunk overlap: 200 characters

3. **Embeddings**
   - `OpenAIEmbeddings`: Converts text to vector representations
   - Model: text-embedding-ada-002

4. **Vector Store**
   - `Chroma`: Stores and retrieves embeddings
   - In-memory or persistent storage options

5. **Language Model**
   - `ChatOpenAI`: Generates responses
   - Model: GPT-3.5-turbo or GPT-4

6. **Retrieval Chain**
   - `RetrievalQA`: Combines retrieval and generation

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   YouTube   │────▶│  Transcript  │────▶│   Chunks    │
│     URL     │     │   Extraction │     │  (1000 chr) │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │◀────│  LLM + RAG   │◀────│  Embeddings │
│             │     │   Pipeline   │     │ & VectorDB  │
└─────────────┘     └──────────────┘     └─────────────┘
                           ▲
                           │
                    ┌──────────────┐
                    │ User Question│
                    └──────────────┘
```

## ⚙️ Configuration

### Customizing Chunk Size

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Larger chunks for more context
    chunk_overlap=300     # More overlap for continuity
)
```

### Changing LLM Model

```python
# Use GPT-4 for better responses
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Use GPT-3.5-turbo for faster, cheaper responses
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

### Adjusting Retrieval Parameters

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Return top 5 most relevant chunks
    )
)
```

### Using Different Vector Stores

```python
# Persistent Chroma storage
vectorstore = Chroma.from_documents(
    docs, 
    embeddings,
    persist_directory="./chroma_db"
)

# Or use FAISS
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embeddings)
```

## 💡 Examples

### Example 1: Educational Content

```python
video_url = "https://www.youtube.com/watch?v=educational_video"

# Sample questions:
questions = [
    "What is the main concept explained in this video?",
    "Can you summarize the key points?",
    "What examples were given?",
    "What are the practical applications?"
]

for question in questions:
    answer = qa_chain.run(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Example 2: Tutorial Video

```python
video_url = "https://www.youtube.com/watch?v=tutorial_video"

# Step-by-step questions:
print(qa_chain.run("What tools are needed?"))
print(qa_chain.run("What is step 1?"))
print(qa_chain.run("What are common mistakes to avoid?"))
```

### Example 3: Interview or Podcast

```python
video_url = "https://www.youtube.com/watch?v=podcast_video"

# Extract insights:
print(qa_chain.run("Who are the speakers?"))
print(qa_chain.run("What are the main topics discussed?"))
print(qa_chain.run("What interesting stories were shared?"))
```

## 🐛 Troubleshooting

### Issue: "No transcript available"

**Solution:**
- Video may not have captions/subtitles
- Try videos with auto-generated or manual captions
- Check if video is public and accessible

### Issue: "API key not found"

**Solution:**
```python
import os
os.environ["OPENAI_API_KEY"] = "your-actual-api-key"
```

### Issue: "Rate limit exceeded"

**Solution:**
- Wait a few minutes before retrying
- Reduce the frequency of requests
- Consider upgrading your OpenAI plan

### Issue: "Out of memory"

**Solution:**
- Reduce chunk size
- Process shorter videos
- Use persistent vector store instead of in-memory

### Issue: "Poor answer quality"

**Solution:**
- Increase chunk overlap for better context
- Adjust retrieval parameters (increase k value)
- Use a more powerful model (GPT-4)
- Improve question phrasing

## 📊 Performance Tips

1. **Chunk Size**: Balance between context and performance
   - Smaller chunks (500-800): Better for specific questions
   - Larger chunks (1000-1500): Better for broader questions

2. **Overlap**: Ensures continuity between chunks
   - Recommended: 10-20% of chunk size

3. **Model Selection**:
   - GPT-3.5-turbo: Fast and cost-effective
   - GPT-4: More accurate but slower and expensive

4. **Caching**: Store vector database to avoid re-processing

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution

- [ ] Add support for multiple videos
- [ ] Create a web interface with Streamlit/Gradio
- [ ] Add support for different languages
- [ ] Implement conversation memory
- [ ] Add video timestamp citations
- [ ] Support for YouTube playlists
- [ ] Export chat history

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) - Amazing framework for LLM applications
- [OpenAI](https://openai.com/) - Powerful language models
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) - Easy transcript extraction

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Databases Guide](https://www.pinecone.io/learn/vector-database/)

## 👤 Author

**Devatva Rachit**

- GitHub: [@Devatva24](https://github.com/Devatva24)
- Project Link: [YouTube Chatbot](https://github.com/Devatva24/Youtube_Chatbot)

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Video timestamp references
- [ ] Streamlit web interface
- [ ] Support for multiple LLM providers
- [ ] Conversation history export
- [ ] Video summary generation
- [ ] Topic extraction and tagging

---

**Built with 🧠 and LangChain**
