import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)


def main():
    res = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    )
    print(res.text)
    print(f"Prompt Tokens: {res.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {res.usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
