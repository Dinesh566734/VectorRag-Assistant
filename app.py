import os
import tempfile
import gradio as gr
from itertools import islice
from datasets import load_dataset
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ------------------ Check Ollama models ------------------
def check_ollama_models():
    required = ["gemma3", "nomic-embed-text"]
    missing = []
    for model in required:
        try:
            Ollama(model=model)
        except Exception:
            missing.append(model)
    if missing:
        print(f"⚠ Missing Ollama models: {missing}. Please download them first inside the container.")
    else:
        print("✅ All required Ollama models are available.")

check_ollama_models()

# ------------------ Loaders ------------------
def load_local_text(filepath="test.txt"):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents([text])
    return docs, f"✅ Loaded {len(docs)} document chunks from file."

def load_bigcode_dataset(sample_size=50):
    try:
        # ✅ Use streaming mode to avoid large downloads
        print("🔍 Loading BigCode Python subset...")
        ds = load_dataset(
            "bigcode/the-stack", 
            data_dir="data/python",  # only load Python subset
            split="train", 
            streaming=True
        )

        # ✅ Stream only the first N samples
        docs = []
        for item in islice(ds, sample_size):
            code = item.get("content") or item.get("text") or ""
            if code.strip():
                docs.append(Document(page_content=code, metadata={"source": "bigcode"}))

        if not docs:
            raise ValueError("No valid code snippets found in dataset.")

        return docs, f"✅ Loaded {len(docs)} Python code documents from BigCode (streaming)."

    except Exception as e:
        print("⚠ BigCode load failed:", e)

        # ✅ Fallback examples so chatbot still works
        fallback = [
            "def add(a, b):\n    return a + b",
            "class Greeter:\n    def greet(self):\n        return 'Hello, world!'",
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"
        ]
        docs = [Document(page_content=code, metadata={"source": "fallback"}) for code in fallback]

        return docs, f"⚠ BigCode unavailable ({e}). Loaded {len(docs)} fallback examples."

# ------------------ Vector Store & RAG ------------------
def create_vectorstore(docs, store_dir="/app/chroma_store"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Truncate overly long documents
    max_len = 7000
    for doc in docs:
        if len(doc.page_content) > max_len:
            doc.page_content = doc.page_content[:max_len]

    Chroma.from_documents(docs, embedding=embeddings, persist_directory=store_dir)
    return "✅ Vector store created successfully."

def build_rag_chain(store_dir="/app/chroma_store"):
    llm = Ollama(model="gemma3")
    vectordb = Chroma(
        persist_directory=store_dir,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain, retriever

# ------------------ Answer Formatter ------------------
def format_friendly_answer(answer_text):
    friendly = ""

    # Detect if the answer contains code or programming patterns
    if any(keyword in answer_text for keyword in ["class ", "def ", "#include", "```", "public ", "int ", "void "]):
        friendly += "✅ Here's how you can do it:\n\n"
        friendly += answer_text.strip() + "\n\n"
        friendly += "💡 Quick explanation: " + " ".join(answer_text.split()[:50]) + "..."
    else:
        # No artificial 500-char limit
        friendly += "💡 Answer:\n\n" + answer_text.strip()
        # Optional: truncate only if it's *extremely* long
        if len(answer_text) > 5000:
            friendly = friendly[:5000] + "\n\n⚠️ (Truncated to first 5000 characters)"
    return friendly



# ------------------ Global State ------------------
class AppState:
    def _init_(self):
        self.qa_chain = None
        self.retriever = None
        self.store_dir = "/app/chroma_store"
        self.initialized = False

state = AppState()

# ------------------ Gradio Functions ------------------
def initialize_system(data_source, file_upload, sample_size):
    try:
        if data_source == "Local File":
            if file_upload is None:
                return "❌ Please upload a file first.", ""

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                if hasattr(file_upload, "read"):
                    tmp_file.write(file_upload.read())
                else:
                    tmp_file.write(file_upload)
                temp_path = tmp_file.name

            docs, load_msg = load_local_text(temp_path)
            store_dir = os.path.join(tempfile.gettempdir(), "chroma_store_local")
            os.makedirs(store_dir, exist_ok=True)
            state.store_dir = store_dir

        elif data_source == "BigCode Dataset":
            docs, load_msg = load_bigcode_dataset(sample_size=int(sample_size))
            store_dir = os.path.join(tempfile.gettempdir(), "chroma_store_bigcode")
            os.makedirs(store_dir, exist_ok=True)
            state.store_dir = store_dir

        else:
            return "❌ Invalid data source selected.", ""

        if not docs:
            return "❌ No documents loaded.", ""

        vector_msg = create_vectorstore(docs, store_dir=state.store_dir)
        state.qa_chain, state.retriever = build_rag_chain(store_dir=state.store_dir)
        state.initialized = True

        return (
            f"{load_msg}\n{vector_msg}\n🤖 RAG system ready!",
            "System initialized successfully! You can now start asking questions.",
        )

    except Exception as e:
        return f"❌ Error during initialization: {str(e)}", ""

def chat_with_rag(message, history):
    if not state.initialized:
        history = history or []
        history.append((message, "⚠ Please initialize the system first using the 'Setup' tab."))
        return history, ""
    
    try:
        history = history or []
        context_history = ""
        if history:
            for q, a in history[-5:]:
                context_history += f"Q: {q}\nA: {a}\n\n"
        query_with_history = f"{context_history}Question: {message}\nAnswer:" if context_history else message
        
        response = state.qa_chain.invoke(query_with_history)
        answer_text = response.get("result", "").strip()
        
        if not answer_text or "i don't know" in answer_text.lower():
            retrieved_docs = state.retriever.get_relevant_documents(message)
            context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])
            llm_direct = Ollama(model="gemma3")
            prompt = f"Answer the question in a simple, user-friendly way. Include code if helpful.\n\nContext:\n{context}\n\nQuestion: {message}\nAnswer:"
            answer_text = llm_direct.invoke(prompt)
        
        friendly_answer = format_friendly_answer(answer_text)
        history.append((message, friendly_answer))
        return history, ""
    
    except Exception as e:
        history = history or []
        history.append((message, f"❌ Error: {str(e)}"))
        return history, ""

# ------------------ Gradio Interface ------------------
with gr.Blocks(title="VectorRAG Code Assistant") as demo:
    gr.Markdown("# 🤖 VectorRAG Code Assistant\n### AI-powered code search & question answering")
    
    with gr.Tab("⚙ Setup"):
        gr.Markdown("### Initialize the RAG System")
        data_source = gr.Radio(["Local File", "BigCode Dataset"], value="Local File", label="Data Source")
        with gr.Row():
            file_upload = gr.File(label="Upload File", file_types=[".txt", ".py", ".js", ".java", ".cpp"])
            sample_size = gr.Slider(10, 200, 50, step=10, label="Sample Size (BigCode)")
        init_button = gr.Button("🚀 Initialize System", variant="primary")
        with gr.Row():
            status_output = gr.Textbox(label="Status", lines=5)
            ready_output = gr.Textbox(label="Ready Status", lines=2)
        init_button.click(fn=initialize_system, inputs=[data_source, file_upload, sample_size], outputs=[status_output, ready_output])
    
    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(label="Conversation", height=600, show_copy_button=True)
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask a question...", label="Your Question", lines=1)
            submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("🗑 Clear Chat")
        submit_btn.click(fn=chat_with_rag, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(fn=chat_with_rag, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    with gr.Tab("ℹ Info"):
        gr.Markdown(
            "## How to Use\n"
            "1. Setup Tab: Initialize with Local File or BigCode Dataset\n"
            "2. Chat Tab: Ask questions about your code\n"
            "✅ Features: Vector-based search, LLM answers, persistent storage"
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)