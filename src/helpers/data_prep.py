import ftfy
import requests
import json
import textwrap
import pandas as pd
from sklearn.model_selection import train_test_split
import concurrent.futures
import spacy
import aiohttp
import asyncio

# Calls to ollama server https://ollama.com/

async def generate_text_async(prompt, model):
    url = 'http://host.docker.internal:11434/api/generate'
    data = {
        "model": model,
        "prompt": "Instruct: " + prompt + "\nOutput:",
        "options": {
            "stop": ["Instruct:", "Output:"]
        },
        "raw": True,
        "stream": False
    }

    async with aiohttp.ClientSession() as session:
        try:
            #print(f"Sending request for prompt: {prompt[:30]}...")
            async with session.post(url, json=data, timeout=10) as response:  # 10-second timeout
                response_json = await response.json()
                return response_json
        except aiohttp.ClientError as e:
            print(f"Request failed for prompt: {prompt[:30]} with error {e}")
            return None

def generate_text(prompt, model):
    url = 'http://host.docker.internal:11434/api/generate'
    data = {
      "model": model,
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
  
def parse_into_sentences(file_path):
    with open(file_path, 'r') as file:
      content = file.read()
      nlp = spacy.load("en_core_web_sm")  # Load the English model
      
      doc = nlp(content)
      sentences = [sent.text for sent in doc.sents]
      
      fixed_sentences = [ftfy.fix_text(sentence) for sentence in sentences]
      
      # Iterate over both original and fixed sentences simultaneously
      for original, fixed in zip(sentences, fixed_sentences):
          if original != fixed:
              # If a difference is found, print the original and fixed sentence for comparison
              print("Original:", original)
              print("Fixed:", fixed)
              break  # Exit the loop after finding the first difference
      
      return fixed_sentences

def get_preparation_prompt(para):
  
  para = para.replace('\n', ' ').strip()  # Added strip() to remove leading and trailing whitespace


  instructions = """### Instructions for Generating a Prompt from a Sci-Fi Novel Sentence
  Given this sentence from a science fiction novel, generate a terse prompt for creating similar text that will be used to finetune an LLM. 
  First, analyze the paragraph to extract key elements such as the main character(s), setting, and principal actions or events. 
  Then, expand these elements by describing the character’s appearance, emotional state, and any unique sci-fi traits. 
  Detail the setting to emphasize its sci-fi nature, whether it be a futuristic city, an alien landscape, or a space vessel etc. 
  Discuss the action, focusing on its significance and how it unfolds in the sci-fi context. 
  Highlight any futuristic technology or advanced scientific concepts mentioned. 
  Determine the tone and mood of the text, and suggest incorporating sensory details to create a vivid and immersive experience. 
  Integrate all these aspects into a terse prompt that captures the essence and atmosphere of the original paragraph, guiding the generation of a similar sci-fi narrative.
  
  Remember you must output only a terse LLM prompt for the purposes of LLM and nothing else!

  """
  
  prompt = f"You are a sci-fi author. Follow these instructions:\n{instructions}\n  ### Paragraph: \n\"{para}\"\n"
  return prompt
  
async def generate_prompt_from_segments(segments, output_file, model, start_paragraph=20):
    df = pd.DataFrame(columns=['Original Paragraph', 'Prompt'])
    total_paragraphs = len(segments)
    semaphore = asyncio.Semaphore(6)
    print(f"Generating to {output_file}")
    async def process_paragraph(i):
        async with semaphore:
          try:
              print(f"Processing segment {i+1} of {total_paragraphs}...")

              paragraph = segments[i]
              prompt = get_preparation_prompt(paragraph)

              response_json = await generate_text_async(prompt, model)
              if response_json is None:
                  return None

              response = response_json.get("response", "").strip().replace('\n', '')
              response = ftfy.fix_text(response)

              if not response:
                  return None

              return {'Original Paragraph': paragraph, 'Prompt': response}
          except Exception as e:
              print(f"Error processing paragraph {i+1}: {e}")
              return None

    end_paragraph = start_paragraph + 100
    tasks = [process_paragraph(i) for i in range(start_paragraph, min(end_paragraph, total_paragraphs))]

    results = await asyncio.gather(*tasks)

    for result in results:
        if result is not None:
            new_row = pd.DataFrame([result])
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(output_file, index=False)
    return df

def split_training_set(df, file_base):
  
  print(df.iloc[0])
  record_index = 0  # Replace with the desired record index

  prompt = df.loc[record_index, 'Prompt']
  original_paragraph = df.loc[record_index, 'Original Paragraph']

  print("Prompt:")
  print(prompt)
  print("\nOriginal Paragraph:")
  print(original_paragraph)

  # Assuming df has been properly defined and contains the columns 'input' and 'output'
  df_transposed = df.rename(columns={'Prompt': 'input', 'Original Paragraph': 'output'})
  df_filtered = df_transposed[['input', 'output']]
  df_filtered.to_json(file_base + '_output.jsonl', orient='records', lines=True)

  # Load the JSONL file
  with open(file_base + '_output.jsonl', 'r') as file:
      jsonl_data = file.readlines()

  # Split the data into training and validation sets
  train_data, val_data = train_test_split(jsonl_data, test_size=0.2, random_state=42)

  # Save the training and validation sets to separate files
  with open(file_base + '_train.jsonl', 'w') as file:
      file.writelines(train_data)

  with open(file_base + '_val.jsonl', 'w') as file:
      file.writelines(val_data)



