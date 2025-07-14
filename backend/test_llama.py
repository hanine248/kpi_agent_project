from llama_cpp import Llama
import os

print("🔄 Starting TinyLlama test...")

model_path = "C:/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=1024)  # Increase context for better output


# Load the model
print("📦 Loading model...")
llm = Llama(model_path=model_path, n_ctx=512)
print("✅ Model loaded!")

# Test prompt
prompt = "You are an HR assistant. The employee has 4 years of experience and a salary of $7000. Evaluate their performance."

print("🧠 Generating response...")
output = llm(prompt, max_tokens=150, echo=False)

print("✅ Response received:")
print(output["choices"][0]["text"].strip())
