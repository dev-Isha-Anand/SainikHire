import requests
from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import os
import re
from pymongo import MongoClient  # For MongoDB connection and querying

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

app = Flask(__name__)

# === Configuration section ===
VECTOR_STORE_PATH = "vectorstore/db_faiss"
LOCAL_MODEL_NAME = "google/flan-t5-small"
JOBS_CSV_PATH = "data/jobs.csv"
MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DBNAME = "SainikHire"
MONGODB_COLLECTION = "information"

#loading the api key
from dotenv import load_dotenv
load_dotenv()
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Prompt template to guide the language model for ex-servicemen related queries
PROMPT_TEMPLATE = """
You are a highly knowledgeable, supportive, and extremely concise chatbot assistant for ex-servicemen and their families.
Based exclusively on the following context, provide direct, clear, and absolutely non-repetitive information.
*Your response must be unique and contain no redundant words, phrases, sentences, or ideas whatsoever.*
Focus on delivering complete answers without any duplication.

*Key Requirements for Your Answer:*
* **Absolute Non-Repetition:** Do NOT re-state, re-explain, or repeat any information already given in the answer. Rephrase entirely if a concept needs to be revisited, but ensure it's truly distinct.
* **Concise and Direct:** Provide information as directly as possible, avoiding unnecessary words or conversational filler.
* **Strictly Context-Bound:** Use ONLY the provided context. If the context lacks sufficient information, state: "I apologize, but I do not have enough specific information in my knowledge base to answer that question comprehensively." Do NOT invent or use external knowledge.
* **Content Focus:** Prioritize information related to:
    * Benefits and Entitlements (pensions, healthcare, education, welfare schemes).
    * Resettlement and Employment (job opportunities, training, entrepreneurship).
    * Support Services (counseling, legal aid, disability support).
    * Community and Associations (veteran organizations, events).
    * General Information (policies, application processes, required documents).
* **Tone:** Maintain a helpful, respectful, and empathetic tone.
* **No Metadata:** Do NOT include any source document identifiers, page numbers, or file names.

Context: {context}
Question: {question}

Helpful Answer:
"""

# Global variables to hold various resources and clients
our_knowledge_base = None
llm_model = None
job_data_df = None
mongo_client = None
mongo_collection = None

# Function to call Gemini 1.5 Flash API for generative responses
def ask_gemini_flash(prompt, api_key=None):
    api_key = api_key or GEMINI_API_KEY
    if not api_key:
        return "Gemini API key missing."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, params=params, json=body, timeout=10)
        if response.ok:
            data = response.json()
            # Extract the generated text from the response JSON
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        return "Gemini API error: " + response.text
    except Exception as e:
        return f"Gemini request failed: {e}"

# Helper to detect if the query is about salaries
def is_salary_query(prompt):
    keywords = ["salary", "pay", "monthly salary", "ctc", "package", "income", "expected salary", "expected pay"]
    return any(kw in prompt.lower() for kw in keywords)

# Setup connection and indexes for MongoDB
def initialize_mongodb():
    global mongo_client, mongo_collection
    try:
        mongo_client = MongoClient(MONGODB_URI)
        mongo_db = mongo_client[MONGODB_DBNAME]
        mongo_collection = mongo_db[MONGODB_COLLECTION]
        print("✅ MongoDB connected successfully.")
        # Create text index to support full-text search on important fields
        mongo_collection.create_index([
            ("title", "text"),
            ("description", "text"),
            ("location", "text"),
            ("rank", "text"),
            ("education", "text")
        ])
        print("Text index created/verified on MongoDB collection.")
        print(mongo_collection.index_information())
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        mongo_client = None
        mongo_collection = None

# Load the vector store (FAISS) for document retrieval
def initialize_vector_store():
    global our_knowledge_base
    try:
        embedding_engine = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        our_knowledge_base = FAISS.load_local(
            VECTOR_STORE_PATH,
            embedding_engine,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    except Exception as e:
        our_knowledge_base = None
        print(f"Error loading vector store: {e}")

# Initialize the FLAN-T5 small model pipeline
def initialize_flan_t5_model(model_name):
    global llm_model
    if llm_model:
        return llm_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device_index = 0 if torch.cuda.is_available() else -1
        print(f"Running model on {'GPU' if device_index == 0 else 'CPU'}")
        model_pipeline = pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            device=device_index
        )
        llm_model = HuggingFacePipeline(pipeline=model_pipeline)
        print(f"Model '{model_name}' is ready.")
        return llm_model
    except Exception as e:
        llm_model = None
        print(f"Failed to load model: {e}")
        return None

# Load job listings CSV into a DataFrame
def load_jobs_csv():
    global job_data_df
    if os.path.exists(JOBS_CSV_PATH):
        job_data_df = pd.read_csv(JOBS_CSV_PATH)
        if job_data_df.empty:
            print("Warning: Jobs CSV file is empty.")
        else:
            print(f"Loaded {len(job_data_df)} job listings.")
    else:
        job_data_df = pd.DataFrame()
        print("Jobs CSV file not found.")

# Check if query is related to jobs
def is_job_related(prompt):
    job_keywords = [
        "job", "jobs", "employment", "vacancy", "career", "openings", "hiring",
        "description", "skills", "requirements", "key requirement"
    ]
    return any(keyword in prompt.lower() for keyword in job_keywords)

# Check if query pertains to ex-servicemen info
def is_ex_servicemen_info(prompt):
    ex_keywords = ["ex-servicemen", "veteran", "retired", "army", "navy", "air force", "military", "defence"]
    return any(word in prompt.lower() for word in ex_keywords)

# Generate job-related response based on user query
def get_job_response(prompt=None):
    if job_data_df is None or job_data_df.empty:
        return "Sorry, job data is currently unavailable."

    prompt_lower = prompt.lower() if prompt else ""

    # If user requests recent or top jobs, sort and return top 5
    if any(term in prompt_lower for term in ["latest jobs", "top jobs", "recent jobs", "5 jobs"]):
        top_jobs = job_data_df.sort_values(by="Post Date", ascending=False).head(5)
        result_lines = []
        for _, row in top_jobs.iterrows():
            line = f"🔹 **{row.get('Job Title', 'N/A')}** at *{row.get('Company Name', 'N/A')}* in {row.get('Location', 'N/A')}"
            if pd.notna(row.get("Job Link")):
                line += f" → [View Job]({row.get('Job Link')})"
            result_lines.append(line)
        return "\n".join(result_lines)

    # Filter jobs based on matched location and title keywords in prompt
    all_locations = job_data_df['Location'].dropna().unique()
    all_titles = job_data_df['Job Title'].dropna().unique()

    matched_locations = [loc for loc in all_locations if loc.lower() in prompt_lower]
    matched_titles = [title for title in all_titles if title.lower() in prompt_lower]

    matched_jobs = job_data_df.copy()
    if matched_titles:
        matched_jobs = matched_jobs[matched_jobs['Job Title'].apply(
            lambda x: any(t.lower() in str(x).lower() for t in matched_titles)
        )]
    if matched_locations:
        matched_jobs = matched_jobs[matched_jobs['Location'].apply(
            lambda x: any(l.lower() in str(x).lower() for l in matched_locations)
        )]
    # Fallback to Gemini when job match fails
    if matched_jobs.empty:
        return ask_gemini_flash(prompt)

    wants_description = any(k in prompt_lower for k in ["description"])

    result_lines = []
    for _, job in matched_jobs.iterrows():
        title = job.get('Job Title', 'N/A')
        company = job.get('Company Name', 'N/A')
        location = job.get('Location', 'N/A')
        link = job.get('Job Link', 'N/A')
        full_desc = job.get("Job Description", "")
        desc_part = str(full_desc).strip()
        lines = []
        if wants_description:
            lines.append(f"🔹 **{title}** - 📝**Description:** {desc_part}")
        else:
            lines.append(f"🔹**Job Title:** {title}\n 🏢**Company:** {company}\n 📍**Location:** {location}\n 🔗**Link:** {link}")
            if desc_part:
                lines.append(f"📝**Description:** {desc_part}")
        result_lines.append("\n".join(lines))

    return "\n\n".join(result_lines)

# Prepare the prompt template for LangChain QA
def prepare_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

@app.route('/')
def home():
    # Serve the main landing page
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_input = request.json.get('prompt')
    if not user_input:
        return jsonify({"error": "Prompt missing from request."}), 400

    prompt_lower = user_input.lower()

    # Priority 1: If user question starts with "what is", "who is", or "tell me about", use Gemini
    if prompt_lower.startswith(("what is", "who is", "tell me about")):
        gemini_reply = ask_gemini_flash(user_input)
        return jsonify({"response": gemini_reply})


    # Priority 2: If query is salary-related, use specialized Gemini prompt
    if is_salary_query(user_input):
        salary_prompt = (
            "You are an expert on Indian police and defense salaries. "
            "Always begin with an estimated monthly salary range in INR. "
            "Then mention pay structure, allowances, and growth prospects concisely.\n\nQuery: " + user_input
        )
        salary_response = ask_gemini_flash(salary_prompt)
        return jsonify({"response": salary_response})

    # Priority 3: Job-related queries handled with CSV data
    if is_job_related(user_input):
        return jsonify({"response": get_job_response(user_input)})

    # Priority 4: Queries related to ex-servicemen info go to MongoDB text search
    if is_ex_servicemen_info(user_input):
        if mongo_collection is None:
            return jsonify({"error": "MongoDB not connected."}), 500
        try:
            # Prepare text search query
            search_query = {"$text": {"$search": user_input}}
            user_input_lower = user_input.lower()
            possible_locations = [
                "punjab", "delhi", "maharashtra", "uttar pradesh", "kerala", "jamshedpur",
                "haryana", "gujarat", "karnataka", "rajasthan", "bihar", "madhya pradesh",
                "telangana", "andhra pradesh", "tamil nadu", "jammu", "kashmir", "srinagar"
            ]
            possible_ranks = [
                "sepoy", "naik", "havildar", "subedar", "subedar major", 
                "lieutenant", "captain", "major", "colonel", "brigadier"
            ]
            # Add regex filters if location or rank found in query
            matched_location = next((loc for loc in possible_locations if loc in user_input_lower), None)
            matched_rank = next((rk for rk in possible_ranks if rk in user_input_lower), None)
            if matched_location:
                search_query["location"] = {"$regex": matched_location, "$options": "i"}
            if matched_rank:
                search_query["rank"] = {"$regex": matched_rank, "$options": "i"}

            results = mongo_collection.find(
                search_query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(5)

            response_lines = []
            for doc in results:
                title = doc.get("title", "No Title")
                desc = doc.get("description", "No Description")
                rank = doc.get("rank", "Rank not available")
                location = doc.get("location", "Location unknown")
                education = doc.get("education", "Education not listed")

                response_lines.append(
                    f"🔹 **{title}**\n"
                    f"- 🏅 Rank: {rank}\n"
                    f"- 📍 Location: {location}\n"
                    f"- 🎓 Education: {education}\n"
                    f"- 📜 Description: {desc}"
                )

            if response_lines:
                return jsonify({"response": "\n\n".join(response_lines)})
            else:
                # Fallback to Gemini if MongoDB gives no result
                gemini_fallback = ask_gemini_flash(user_input)
                return jsonify({"response": gemini_fallback})

        except Exception as e:
            return jsonify({"error": f"MongoDB query failed: {str(e)}"}), 500

    # Final fallback: Use vector store + language model to answer from indexed documents
    if our_knowledge_base is None:
        return jsonify({"error": "Vector store not loaded."}), 500
    if llm_model is None:
        return jsonify({"error": "LLM model not initialized."}), 500

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=our_knowledge_base.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prepare_prompt(PROMPT_TEMPLATE)}
        )
        response = qa_chain.invoke({'query': user_input})
        answer_text = response.get("result", "").strip().lower()
        generic_failures = [
            "i apologize", "not enough specific", "i do not have enough",
            "sorry", "insufficient", "unable to find", "no relevant information", "helpful"
        ]

        if answer_text and not any(frag in answer_text for frag in generic_failures):
            return jsonify({"response": response["result"]})
        else:
            return jsonify({"response": ask_gemini_flash(user_input)})

    except Exception as e:
        return jsonify({"error": f"Vector store error: {e}"}), 500

if __name__ == "__main__":
    if not os.path.exists('templates'):
        os.makedirs('templates')
    print("Starting server...")
    initialize_vector_store()
    initialize_flan_t5_model(LOCAL_MODEL_NAME)
    load_jobs_csv()
    initialize_mongodb()
    print("Initialization complete. Server is live.")
    app.run(debug=True, port=5000)
