
import pandas as pd
from helpers.data_prep import parse_into_paragraphs, generate_text, generate_prompt_from_paragraphs, split_training_set
import os

model = "mistral"
file_path = './data/exc.txt'

def get_output_file():
    outputfile = os.path.splitext(file_path)[0] + f'_{model}_prep_1.csv'
    ordinal = 1
    while os.path.exists(outputfile):
        outputfile = os.path.splitext(file_path)[0] + f'_{model}_prep_{ordinal}.csv'
        ordinal += 1
    return outputfile

def get_max_output_file():
    ordinal = 1
    outputfile = os.path.splitext(file_path)[0] + f'_{model}_prep_{ordinal}.csv'
    while True:
        next_file = os.path.splitext(file_path)[0] + f'_{model}_prep_{ordinal + 1}.csv'
        if not os.path.exists(next_file):
            break
        ordinal += 1
        outputfile = next_file
    return outputfile

### this is the big test, that builds the paragraphs
def test_big_split():
    
    
    outputfile = get_output_file()
    
    paragraphs = parse_into_paragraphs(file_path)
    assert len(paragraphs) != 0
    ## this will write output.csv which is half prepared
    df = generate_prompt_from_paragraphs(paragraphs, outputfile, model)
    ## this will write output.jsonl, train.jsonl, val.jsonl
    split_training_set(df)

def test_para_split():
    
    
    paragraphs = parse_into_paragraphs(file_path)
    assert len(paragraphs) != 0
    print(paragraphs[20])
    
    
    

def test_split_dataset():
    output_file = get_max_output_file()
    df = pd.read_csv(output_file)
    split_training_set(df, output_file)
    ## this will write output.jsonl, train.jsonl, val.jsonl


def test_generate():
    result = generate_text("The God Next Door")
    assert len(result["response"]) != 0
    print(result["response"])
