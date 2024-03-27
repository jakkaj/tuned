import requests
import json
import textwrap

def generate_text(prompt):
    url = 'http://host.docker.internal:11434/api/generate'
    data = {
      "model": "mistral",
      "prompt": "Instruct: " + prompt + "\nOutput:",
      "options": {
        "stop": ["Instruct:", "Output:"]
      },
      "raw": True,
      "stream": False
    }

    response = requests.post(url, json=data)
    return response.json()

def parse_into_paragraphs(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        paragraphs = content.split('\n\n')  # Splitting on two newlines to get paragraphs
    return paragraphs