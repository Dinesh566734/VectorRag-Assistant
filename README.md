# 🤖 VectorRAG Code Assistant

An **AI-powered Retrieval-Augmented Generation (RAG)** chatbot that answers **code-related questions** using **Ollama**, **LangChain**, and **ChromaDB**.  
Built with **Gradio** for an intuitive, interactive web interface.

---

## 🚀 Features

- 💬 **Code-Aware Q&A** — Ask questions about your codebase or documentation  
- 🧠 **Local Embeddings** using Ollama (`nomic-embed-text`)  
- 🗂️ **ChromaDB Vector Store** for persistent context storage  
- ⚙️ **LangChain RAG Pipeline** for retrieval + generation  
- 🌐 **Gradio Interface** for local or web-based chat UI  
- 🪶 Lightweight, private, and runs fully **offline**

---

## 🧩 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Dinesh566734/VectorRag-Assistant.git
cd VectorRag-Assistant
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # On Windows
# OR
source venv/bin/activate # On macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama Models
Make sure [Ollama](https://ollama.ai) is installed and running locally.

Then pull the required models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 5. Run the App
```bash
python app.py
```

Then open your browser at 👉 **http://127.0.0.1:7860**

---

## 📂 Project Structure

```
VectorRag-Assistant/
├── app.py                  # Main Gradio + RAG logic
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── LICENSE
└── .github/
    ├── ISSUE_TEMPLATE.md
    └── PULL_REQUEST_TEMPLATE.md
```

---

## 🧰 Tech Stack

| Component | Tool / Library |
|------------|----------------|
| LLM Backend | **Ollama (Llama 3)** |
| Embeddings | **nomic-embed-text** |
| Vector Store | **ChromaDB** |
| Framework | **LangChain** |
| Interface | **Gradio** |
| Language | **Python 3.10+** |

---

## 💡 Example Usage

| 💬 You Ask | 🤖 Assistant Answers |
|------------|----------------------|
| “Explain this Python recursion code.” | Gives a detailed walkthrough with examples. |
| “What’s the time complexity of this C++ snippet?” | Explains complexity with reasoning. |
| “Refactor this function for readability.” | Suggests improved code with comments. |

---

## 🧪 Environment Variables (Optional)

If you use a `.env` file for configs, include:
```bash
OLLAMA_HOST=http://localhost:11434
VECTOR_DB_PATH=./chroma_store
EMBED_MODEL=nomic-embed-text
LLM_MODEL=llama3
```

---

## 🤝 Contributing

Contributions are welcome!  
1. Fork the repo  
2. Create a new branch (`git checkout -b feature/awesome-feature`)  
3. Commit changes (`git commit -m 'Add awesome feature'`)  
4. Push to your fork (`git push origin feature/awesome-feature`)  
5. Open a Pull Request 🚀  

Please follow the [Issue Template](.github/ISSUE_TEMPLATE.md) and [PR Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## 🪪 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.ai/)
- [Chroma](https://www.trychroma.com/)
- [Gradio](https://gradio.app/)
- [BigCode Dataset](https://huggingface.co/bigcode)

---

## 🚧 Future Enhancements

- [ ] Add Docker support  
- [ ] Add syntax-highlighted code output  
- [ ] Support multiple embedding models  
- [ ] Add conversation memory  
- [ ] Integrate HuggingFace API

---

## 👨‍💻 Author

**Dinesh Kumar Maranani**  
📦 GitHub: [@Dinesh566734](https://github.com/Dinesh566734)  
💬 Building intelligent assistants for developers and researchers.
