from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback
import requests
import json
import re

app = Flask(__name__)
CORS(app)  # ✅ This allows all origins (good for development)

# Load the prompt template from a file
with open("prompt_template.txt", "r") as f:
    template = f.read()

def prompting_cv(data):
    """Generate resume using Ollama deepseek-r1:1.5b locally."""
    # Fill in the prompt template with user data
    prompt_text = template.format(**data)
        # name=data["name"],
        # role=data["role"],
        # email=data["email"],
        # phone=data["phone"],
        # address=data["address"],
        # summary=data["summary"],
        # education=data["education"],
        # experience=data["experience"],
        # skills=data["skills"],
        # projects=data["projects"],
        # awards=data["awards"]
    # )

    # Call Ollama local API
    print(">>> Prompt to Ollama:\n", prompt_text)  # 👈 Add this for debugging

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:1.5b",
            "prompt": prompt_text,
            "stream": False
        }
    )

    print(">>> Ollama response status:", response.status_code)  # 👈 Add this
    print(">>> Ollama response body:", response.text[:500])     # 👈 Add this

    if response.status_code != 200:
        raise Exception(f"Ollama error {response.status_code}: {response.text}")
     # Extract and clean response
    raw_response = response.json()["response"]

    # 🧽 Remove <think>...</think> section (if any)
    clean_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    return clean_response


@app.route('/generate-cv', methods=['POST'])
def generate_cv():
    data = request.json

    required_fields = [
        "name", "role", "email", "phone", "address",
        "summary", "education", "experience",
        "skills", "projects", "awards"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        resume = prompting_cv(data)
        return jsonify({"resume": resume})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# @app.route('/render-resume', methods=['POST'])
# def render_resume():
#     data = request.json
#     generated_resume = prompting_cv(data)
#     structured = json.loads(generated_resume)  # ✅ now it will work
#     data.update(structured)  # ⬅️ map fields into template



#     # return render_template(template_name, **data)
#     template_name = data.get("template", "template1.html")  # default template
#     try:
#         return render_template(template_name, **data)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)






