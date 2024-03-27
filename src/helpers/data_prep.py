import requests
import json
import textwrap
import pandas as pd
from sklearn.model_selection import train_test_split

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

def get_preparation_prompt(para):
  
  para = para.replace('\n', ' ').strip()  # Added strip() to remove leading and trailing whitespace


  instructions = """### Instructions for Generating a Prompt from a Sci-Fi Novel Paragraph
  Given this paragraph from a science fiction novel, generate a terse prompt for creating similar text that will be used to finetune an LLM. 
  First, analyze the paragraph to extract key elements such as the main character(s), setting, and principal actions or events. 
  Then, expand these elements by describing the character’s appearance, emotional state, and any unique sci-fi traits. 
  Detail the setting to emphasize its sci-fi nature, whether it be a futuristic city, an alien landscape, or a space vessel etc. 
  Discuss the action, focusing on its significance and how it unfolds in the sci-fi context. 
  Highlight any futuristic technology or advanced scientific concepts mentioned. 
  Determine the tone and mood of the text, and suggest incorporating sensory details to create a vivid and immersive experience. 
  Integrate all these aspects into a terse prompt that captures the essence and atmosphere of the original paragraph, guiding the generation of a similar sci-fi narrative.
  
  Remember you must output only a terse LLM prompt for the purposes of LLM and nothing else!

  """
  
  prompt = f"You are a sci-fi author. Follow these instructions:\n{instructions}\n  ### Paragraph \n\"{para}\"\n ### Prompt: "
  return prompt
  
  
def generate_prompt_from_paragraphs(paragraphs, output_file, model, start_paragraph=20):  

  # Initialize DataFrame
  df = pd.DataFrame(columns=['Original Paragraph', 'Prompt'])
  process_cutoff = len(paragraphs) #limit (e.g. 10 to process only 10)
  total_paragraphs = len(paragraphs)
  for i in range(start_paragraph, min(total_paragraphs, start_paragraph + process_cutoff)):
      print(f"Processing {i+1} of {min(start_paragraph + total_paragraphs, start_paragraph + process_cutoff)} paragraphs.")
      
      paragraph = paragraphs[i]
      prompt = get_preparation_prompt(paragraph)

      response_json = generate_text(prompt, model)
      response = response_json["response"]
      
      response = f"{response.strip().replace('\n', '')}"
      
      if not response:
        continue
      
      # Append to DataFrame using pd.concat
      new_row = pd.DataFrame({'Original Paragraph': [paragraph], 'Prompt': [response]})
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



