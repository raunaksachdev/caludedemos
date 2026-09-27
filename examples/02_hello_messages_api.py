"""Smallest possible Messages API call: one request, no tools, print the text.

Counterpart to 01_hello_query.py. Requires ANTHROPIC_API_KEY with Console credits.
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    messages=[{"role": "user", "content": "Explain what an AI agent is in one sentence."}],
)
for block in response.content:
    if block.type == "text":
        print(block.text)
