import requests
import json
import textwrap
import pandas as pd
from sklearn.model_selection import train_test_split

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

def get_preparation_prompt(para):
  
  para = para.replace('\n', ' ').strip()  # Added strip() to remove leading and trailing whitespace


  instructions = """### Instructions for Generating a Detailed Prompt from a Sci-Fi Novel Sentence
  Given a paragraph from a science fiction novel, generate a detailed prompt for creating similar text. First, analyze the paragraph to extract key elements such as the main character(s), setting, and principal actions or events. 
  Then, expand these elements by describing the character’s appearance, emotional state, and any unique sci-fi traits. Detail the setting to emphasize its sci-fi nature, whether it be a futuristic city, an alien landscape, or a space vessel. 
  Discuss the action in depth, focusing on its significance and how it unfolds in the sci-fi context. 
  Highlight any futuristic technology or advanced scientific concepts mentioned. Determine the tone and mood of the text, and suggest incorporating sensory details to create a vivid and immersive experience. 
  Integrate all these aspects into a comprehensive prompt that captures the essence and atmosphere of the original paragraph, guiding the generation of a similar sci-fi narrative.

  Create a detailed prompt for generating a paragraph in the style of high-concept science fiction, focusing on elements typical of authors known for their complex narratives and richly detailed worlds. 
  The text should feature morally ambiguous characters, intricate plotlines, and a setting in a vast, technologically advanced universe. 
  Emphasize the creation of a multi-layered narrative, where characters navigate through a world filled with advanced technology, artificial intelligences, and sprawling cosmic environments. 
  The tone should be mature, blending cerebral themes with visceral experiences, and incorporating a subtle, dark humor. 
  The prompt should guide the LLM to produce text that explores ethical dilemmas, power dynamics, and the impact of technology on society, all within a deeply immersive and vividly described sci-fi setting.

  1. **Core Elements Identification**
    - Break down the sentence to identify the main components: character(s), setting, action, and any notable technology or sci-fi element.

  2. **Descriptive Details Expansion**
    - **Character**: Describe the character’s physical appearance, emotional state, and actions, highlighting any unique sci-fi characteristics or traits.
    - **Setting**: Detail the environment or location, focusing on elements that emphasize the sci-fi genre, such as futuristic cities, alien planets, or advanced technology settings.
    - **Action**: Elaborate on the central action or event described in the sentence, detailing how it unfolds and its significance in the scene.

  3. **Sci-Fi Elements Incorporation**
    - Emphasize any sci-fi elements present in the sentence, such as futuristic technology, extraterrestrial life forms, or advanced scientific concepts, describing how they feature in the scene.

  4. **Tone and Mood Setting**
    - Determine the sentence's tone and mood (e.g., tense, mysterious, exhilarating, somber) and use this to guide the descriptive language and atmosphere in the prompt.

  5. **Vivid Picture Creation**
    - Use sensory details to create an immersive experience, describing visuals, sounds, and tactile sensations to bring the scene to life.

  6. **Cohesive Elements Combination**
    - Integrate all the identified and expanded details into a cohesive and comprehensive description that captures the essence and atmosphere of the original sentence.

  This structured approach will help in transforming a sentence from a science fiction novel into a detailed and vivid prompt suitable for further exploration or image generation.

  Remember you must output only terse LLM prompt that will generate the text!

  """
  
  prompt = f"Generate a terse prompt for an LLM that will generate the supplied text. Text is a paragraph from a sci-fi novel. **Text** \n\"{para}\"\nFollow these instructions:\n{instructions}\n "
  return prompt
  
  
def generate_prompt_from_paragraphs(paragraphs, start_paragraph=20):  

  # Initialize DataFrame
  df = pd.DataFrame(columns=['Original Paragraph', 'Prompt'])
  process_cutoff = len(paragraphs) #limit (e.g. 10 to process only 10)
  total_paragraphs = len(paragraphs)
  for i in range(start_paragraph, min(total_paragraphs, start_paragraph + process_cutoff)):
      paragraph = paragraphs[i]
      prompt = get_preparation_prompt(paragraph)

      response_json = generate_text(prompt)
      response = response_json["response"]
      
      response = f"{response.strip()}"
      # Append to DataFrame using pd.concat
      new_row = pd.DataFrame({'Original Paragraph': [paragraph], 'Prompt': [response]})
      df = pd.concat([df, new_row], ignore_index=True)
      print(f"Processed {i+1} of {min(start_paragraph + total_paragraphs, start_paragraph + process_cutoff)} paragraphs.")
      df.to_csv('output.csv', index=False)
  
  return df

def split_training_set(df):
  
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
  df_filtered.to_json('output.jsonl', orient='records', lines=True)

  # Load the JSONL file
  with open('output.jsonl', 'r') as file:
      jsonl_data = file.readlines()

  # Split the data into training and validation sets
  train_data, val_data = train_test_split(jsonl_data, test_size=0.2, random_state=42)

  # Save the training and validation sets to separate files
  with open('train.jsonl', 'w') as file:
      file.writelines(train_data)

  with open('val.jsonl', 'w') as file:
      file.writelines(val_data)



