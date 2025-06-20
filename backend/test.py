import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("HUGGINGFACE_API_KEY"))
print("hi")
