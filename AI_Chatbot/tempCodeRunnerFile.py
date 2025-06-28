from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import os
import re
from pymongo import MongoClient  # MongoDB Integration

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

app = Flask(__name__)

# Configuration
VECTOR_STORE_PATH = "vectorstore/db_faiss"
LOCAL_MODEL_NAME = "google/flan-t5-small"
JOBS_CSV_PATH = "data/jobs.csv"
MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DBNAME = "SainikHire"
MONGODB_COLLECTION = "information"

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

our_knowledge_base = None
llm_model = None
job_data_df = None
mongo_client = None
mongo_collection = None

def initialize_mongodb():
    global mongo_client, mongo_collection
    try:
        mongo_client = MongoClient(MONGODB_URI)
        mongo_db = mongo_client[MONGODB_DBNAME]
        mongo_collection = mongo_db[MONGODB_COLLECTION]
        print("✅ MongoDB connected.")
        mongo_collection.create_index([
            ("title", "text"),
            ("description", "text"),
            ("location", "text"),
            ("rank", "text"),
            ("education", "text")
        ])
        print("🄤 Text index ensured.")
        print(mongo_collection.index_information())
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        mongo_client = None
        mongo_collection = None

def initialize_vector_store():
    global our_knowledge_base
    try:
        embedding_engine = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        our_knowledge_base = FAISS.load_local(
            VECTOR_STORE_PATH,
            embedding_engine,
            allow_dangerous_deserialization=True
        )
        print("Vector store initialized.")
    except Exception as e:
        our_knowledge_base = None
        print(f"Vector store loading failed: {e}")

def initialize_flan_t5_model(model_name):
    global llm_model
    if llm_model:
        return llm_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device_index = 0 if torch.cuda.is_available() else -1
        print(f"Using {'CUDA' if device_index == 0 else 'CPU'}")
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
        print(f"Model '{model_name}' loaded.")
        return llm_model
    except Exception as e:
        llm_model = None
        print(f"Model load error: {e}")
        return None

def load_jobs_csv():
    global job_data_df
    if os.path.exists(JOBS_CSV_PATH):
        job_data_df = pd.read_csv(JOBS_CSV_PATH)
        if job_data_df.empty:
            print("Job CSV is empty.")
        else:
            print(f"Loaded {len(job_data_df)} jobs.")
    else:
        job_data_df = pd.DataFrame()
        print("Job CSV not found.")

def is_job_related(prompt):
    job_keywords = [
        "job", "jobs", "employment", "vacancy", "career", "openings", "hiring",
        "description", "skills", "requirements", "key requirement"
    ]
    return any(keyword in prompt.lower() for keyword in job_keywords)

def is_ex_servicemen_info(prompt):
    ex_keywords = ["ex-servicemen", "veteran", "retired", "army", "navy", "air force", "military", "defence"]
    return any(word in prompt.lower() for word in ex_keywords)

def get_job_response(prompt=None):
    if job_data_df is None or job_data_df.empty:
        return "Sorry, job data is currently unavailable."

    prompt_lower = prompt.lower() if prompt else ""

    if any(term in prompt_lower for term in ["latest jobs", "top jobs", "recent jobs", "5 jobs"]):
        top_jobs = job_data_df.sort_values(by="Post Date", ascending=False).head(5)
        result_lines = []
        for _, row in top_jobs.iterrows():
            line = f"🔹 **{row.get('Job Title', 'N/A')}** at *{row.get('Company Name', 'N/A')}* in {row.get('Location', 'N/A')}"
            if pd.notna(row.get("Job Link")):
                line += f" → [View Job]({row.get('Job Link')})"
            result_lines.append(line)
        return "\n".join(result_lines)

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

    if matched_jobs.empty:
        return "Sorry, I couldn't find any job matching your request. Try using a different or more specific keyword."

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
            lines.append(f"**{title}** - **Description:** {desc_part}")
        else:
            lines.append(f"**Job Title:** {title}\n**Company:** {company}\n**Location:** {location}\n**Link:** {link}")
            if desc_part:
                lines.append(f"**Description:** {desc_part}")
        result_lines.append("\n".join(lines))

    return "\n\n".join(result_lines)

def prepare_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_input = request.json.get('prompt')
    if not user_input:
        return jsonify({"error": "Prompt missing from request."}), 400

    if is_job_related(user_input):
        return jsonify({"response": get_job_response(user_input)})

    if is_ex_servicemen_info(user_input):
        if mongo_collection is None:
            return jsonify({"error": "MongoDB not connected."}), 500
        try:
            search_query = {"$text": {"$search": user_input}}
            user_input_lower = user_input.lower()
            possible_locations = [
                "punjab", "delhi", "maharashtra", "uttar pradesh", "kerala",
                "haryana", "gujarat", "karnataka", "rajasthan", "bihar", "madhya pradesh",
                "telangana", "andhra pradesh", "tamil nadu", "jammu", "kashmir", "srinagar"
            ]
            possible_ranks = [
                "sepoy", "naik", "havildar", "subedar", "subedar major", 
                "lieutenant", "captain", "major", "colonel", "brigadier"
            ]
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
                return jsonify({"response": "No relevant ex-servicemen records found for your query."})

        except Exception as e:
            return jsonify({"error": f"MongoDB query failed: {str(e)}"}), 500

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
        return jsonify({"response": response["result"]})
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    if not os.path.exists('templates'):
        os.makedirs('templates')
    print("Starting server...")
    initialize_vector_store()
    initialize_flan_t5_model(LOCAL_MODEL_NAME)
    load_jobs_csv()
    initialize_mongodb()
    print("Initialization complete. Server running...")
    app.run(debug=True, port=5000)
