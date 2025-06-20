import os
import requests
from dotenv import load_dotenv
load_dotenv()


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"




HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    "Content-Type": "application/json"
}

# print(HEADERS)  # Uncomment to debug token loading

def explain_performance_with_gpt(employee_profile: dict) -> str:
    """
    Generates a performance explanation using a Hugging Face language model.
    """

    # Build a smart prompt
    prompt = "You are an HR expert. Analyze the following employee information and explain their performance:\n\n"
    for key, value in employee_profile.items():
        prompt += f"- {key.replace('_', ' ')}: {value}\n"
    prompt += "\nProvide a short and insightful assessment of their overall performance and contribution to the company."

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=30
        )
        response.raise_for_status()
        output = response.json()

        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        else:
            return "[GPT Error] Unexpected response format"

    except requests.exceptions.HTTPError as http_err:
        return f"[GPT Error] HTTP error: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"[GPT Error] Request error: {req_err}"
    except Exception as e:
        return f"[GPT Error] Unexpected error: {str(e)}"
