import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Retrieve variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
version = os.getenv("AZURE_OPENAI_API_VERSION")

# Confirmation check
print("--- Azure Environment Variables Loaded ---")
print(f"Endpoint: {endpoint}")
print(f"Deployment: {deployment}")
print(f"API Version: {version}")

# Safety check for the key
if api_key:
    print(f"API Key: {api_key[:4]}**** (Loaded Successfully)")
else:
    print("API Key: NOT FOUND")