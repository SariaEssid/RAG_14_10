# 📘 langchain-rag-service

## 🧠 Overview
`langchain-rag-service` is a Retrieval-Augmented Generation (RAG) microservice built using FastAPI and LangChain.

It provides intelligent knowledge-based answers by retrieving information from markdown documents using vector search and Large Language Models (LLMs).

This service is designed to work as an external Knowledge Engine for conversational platforms such as Rasa chatbot systems.

---

## ⭐ Architecture Role
Frontend → Rasa → LangChain RAG Service → Vector Store → LLM

### Responsibilities
- Semantic document retrieval
- Multi-document reasoning
- Knowledge base management
- Metadata-based filtering (future extension)

---

## 🛠 Technology Stack

| Component         | Technology                    |
|-------------------|-------------------------------|
| API Framework     | FastAPI                       |
| LLM Orchestration | LangChain                     |
| Embeddings        | Sentence Transformers         |
| Vector Store      | FAISS                         |
| Language Model    | OpenAI / HuggingFace / Ollama |
| Configuration     | Python + dotenv               |
| Containerization  | Docker                        |

---

## ⚙️ Installation

### 1️⃣ Clone Repository
```bash
git clone <repository-url>
cd langchain-rag-service

### 2️⃣ Create Virtual Environment
python -m .venv .venv

Activate environment:
Windows:
.venv\Scripts\activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt
```

---

## 📈 Future Enhancements

Metadata filtering (domain / grade)
Streaming token responses
Multi-language support
Knowledge admin interface
Hybrid search (keyword + vector)
Conversation memory
Source citation support
Caching layer (Redis)

---

## 🔐 Security Recommendations

Protect API with authentication tokens
Validate user inputs
Implement rate limiting
Monitor prompt injection risks

---

## 📊 Monitoring Recommendations

Log user queries
Track retrieval accuracy
Monitor latency and token usage
Add observability tools (Prometheus / Grafana)

---

## 🧪 Testing

You can test API using:
Swagger UI
Postman
curl

---

## 📜 License
This project is licensed under a Proprietary License:
You cannot use, copy, modify, merge, publish, distribute, sublicense, or sell copies of this Software or its parts outside the scope of internal enterprise use.

Contact [ESSID Saria] or the IT Department for special permissions.

---

## 🙏 Acknowledgments
Thanks to the IT and DEvops teams for the internal support.
Built with love by [ESSID Saria] 💻