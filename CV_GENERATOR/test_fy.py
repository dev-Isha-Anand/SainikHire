import requests
import json

url = "http://127.0.0.1:5000/generate-cv"
data = {
    "name": "Justin Smith",
    "role": "Data Engineer",
    "email": "justinsmith@email.com",
    "phone": "(123) 456-7890",
    "address": "123 Main Street, Anytown, USA",
    "summary": "Experienced data engineer with 3 years of experience in creating data pipelines and ensuring data accuracy.",
    "education": "BSc in Computer Science, University X, May 20XX",
    "experience": "Data Engineer, DEF Company\nJune 20XX – Present\n– Built data pipelines...\n– Worked with Python and SQL...",
    "skills": "Python, SQL, Airflow, Power BI",
    "projects": "– ETL pipeline to analyze traffic\n– Real-time dashboard with Power BI",
    "awards": "Employee of the Month – March 2023"
}

try:
    response = requests.post(url, json=data)
    print("STATUS CODE:", response.status_code)
    
    if response.status_code == 200:
        print("✅ CV Output:\n", response.json()["resume"])
    else:
        print("❌ Error Response:")
        print(json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print("🚫 Network Error:", e)
except Exception as e:
    print("🚫 Unexpected Error:", e)






# import requests
# import json

# url = "http://127.0.0.1:5000/generate-cv"  # Use localhost instead of 127.0.0.1
# data = {
#     "name": "John Smith",
#     "role": "Data Engineer",
#     "email": "johnsmith@email.com",
#     "phone": "(123) 456-7890",
#     "address": "123 Main Street, Anytown, USA",
#     "summary": "Experienced data engineer with 3 years of experience in creating data pipelines and ensuring data accuracy.",
#     "education": "BSc in Computer Science, University X, May 20XX",
#     "experience": "Data Engineer, DEF Company\nJune 20XX – Present\n– Built data pipelines...\n– Worked with Python and SQL...",
#     "skills": "Python, SQL, Airflow, Power BI",
#     "projects": "– ETL pipeline to analyze traffic\n– Real-time dashboard with Power BI",
#     "awards": "Employee of the Month – March 2023"
# }

# try:
#     response = requests.post(url, json=data)
#     print("STATUS CODE:", response.status_code)
    
#     if response.status_code == 200:
#         print("✅ CV Output:\n", response.json()["resume"])
#     else:
#         print("❌ Error Response:")
#         print(json.dumps(response.json(), indent=2))
        
# except requests.exceptions.RequestException as e:
#     print("🚫 Network Error:", e)
# except Exception as e:
#     print("🚫 Unexpected Error:", e)







# import requests

# url = "http://127.0.0.1:5000/generate-cv"
# data = {
#   "name": "John Smith",
#   "role": "Data Engineer",
#   "email": "johnsmith@email.com",
#   "phone": "(123) 456-7890",
#   "address": "123 Main Street, Anytown, USA",
#   "summary": "Experienced data engineer with 3 years of experience in creating data pipelines and ensuring data accuracy.",
#   "education": "BSc in Computer Science, University X, May 20XX",
#   "experience": "Data Engineer, DEF Company\nJune 20XX – Present\n– Built data pipelines...\n– Worked with Python and SQL...",
#   "skills": "Python, SQL, Airflow, Power BI",
#   "projects": "– ETL pipeline to analyze traffic\n– Real-time dashboard with Power BI",
#   "awards": "Employee of the Month – March 2023"
# }

# response = requests.post(url, json=data)

# print("STATUS CODE:", response.status_code)
# print("RAW TEXT RESPONSE:", response.text)  # Show what Flask returned

# # Try parsing only if status is OK
# if response.status_code == 200:
#     result = response.json()["resume"]
#     print("✅ CV Output:\n", result)
# else:
#     print("❌ Something went wrong.")

# response = requests.post(url, json=data)
# cv_output = response.json()["resume"]

# # Print the clean CV
# print(cv_output)

# # Optionally, save to .txt
# with open("resume.txt", "w") as f:
#     f.write(cv_output)
