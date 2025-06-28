🧠SanikHire Chatbot
📌Overview
The SanikHire Chatbot is an AI-powered assistant built to provide fast, relevant, and personalized answers to Ex-Servicemen. It leverages multiple data sources — scraped job listings, PDF-based knowledge, and MongoDB — to deliver intelligent responses through natural language queries. This module is designed to run locally and can be integrated into broader systems.

✨ About
This chatbot simplifies access to job opportunities and welfare information by intelligently routing queries to the right knowledge source — CSV (scraped jobs), FAISS vector store (PDFs), MongoDB (personalized data), or Google Gemini API (fallback AI).

🚀 Features
🔍 Smart Query Handling – Routes queries based on their type: jobs, salary, location, document facts, or general questions.

📄 PDF-based QA – Uses FAISS vector store with HuggingFace embeddings for document-based answers.

🗂 Job Data Integration – Uses scraped job listings from CSV and filters by title, location, and post date.

🧠 MongoDB Support – Provides personalized responses based on user rank, location, or education.

🔗 Google Gemini API – Handles fallback responses and “What is…” queries with real-time generative answers.

📅 Recent Job Sorting – Always shows the most recently posted jobs.

💬 Formatted Replies – Returns clean and emoji-enhanced Markdown-style answers.

🎨 Frontend Technology
HTML (via render_template in Flask), CSS

🛠️ Backend Technology
Framework: Flask (Python)
Language Model: Google FLAN-T5 Small via HuggingFace
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Store: FAISS (local disk-based)
CSV Source: Scraped job listings (with post date)
Database: MongoDB (for ex-servicemen personalization)
Fallback API: Google Gemini 1.5 Flash
🎯 Outcome
Delivers accurate, smart answers to job- and welfare-related questions for Ex-Servicemen. Offers a modular, extensible backend architecture that integrates structured (CSV, MongoDB) and unstructured (PDF) data. Acts as a demo-ready local assistant, with potential for deployment via web or mobile channels.
