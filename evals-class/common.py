from dotenv import load_dotenv
import os

from anthropic import Anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def get_response(prompt: str, model: str = "claude-3-5-sonnet-20240620"):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
