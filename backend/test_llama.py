from llama_cpp import Llama
import os

print("🔄 Starting TinyLlama test...")

# Path to the downloaded TinyLlama model
model_path = "C:/Users/pc/Downloads/tinyllama-1.1b-chat-v1.1.Q4_K_M.gguf"



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
