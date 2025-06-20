from llama_cpp import Llama
import os
import re

model_path = "C:/Users/pc/Downloads/tinyllama-1.1b-chat-v1.1.Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=512)

def explain_performance_with_gpt(employee_profile: dict) -> tuple:
    # Prompt with strict format instructions
    prompt = (
    "You are an expert HR analyst AI. Based on the employee profile below, "
    "rate their performance on a scale of 0 to 5 and explain briefly why.\n"
    "- Give a short and realistic explanation.\n"
    "- Be objective. Do not just say 'poor performance', explain why.\n"
    "- If performance is weak, mention specific points like lack of experience, low satisfaction, or missed deadlines.\n"
    "- Format strictly:\n"
    "Score: <number>\n"
    "Explanation: <brief explanation>\n\n"
)

    for key, value in employee_profile.items():
        prompt += f"{key.replace('_', ' ')}: {value}\n"
    prompt += "\nNow respond:\n"

    try:
        output = llm(prompt, max_tokens=150, echo=False)
        text = output["choices"][0]["text"].strip()

        # Extract score and explanation
        score_match = re.search(r'Score:\s*(\d+(\.\d+)?)', text, re.IGNORECASE)
        explanation_match = re.search(r'Explanation:\s*(.*)', text, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else None
        explanation = explanation_match.group(1).strip() if explanation_match else text.strip()

        # Logical correction for contradictory output
        if score is not None:
            if "excellent" in explanation.lower() and score < 3:
                score = 4.5
            elif "poor" in explanation.lower() and score > 3:
                score = 1.5

        return score, explanation

    except Exception as e:
        return None, f"[LLaMA Error] {e}"
