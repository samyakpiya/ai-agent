import os
import sys

from dotenv import load_dotenv
from google import genai


def main():
    if not len(sys.argv) == 2:
        raise Exception("Please provide the prompt as the second argument")
        sys.exit(1)

    _ = load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    prompt = sys.argv[1]
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )

    print(response.text)

    assert response.usage_metadata
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
