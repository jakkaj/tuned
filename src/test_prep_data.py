
import pandas as pd
from helpers.data_prep import parse_into_paragraphs, generate_text, generate_prompt_from_paragraphs, split_training_set



### this is the big test, that builds the paragraphs
def test_big_split():
    file_path = './data/thegodnextdoor.txt'
    paragraphs = parse_into_paragraphs(file_path)
    assert len(paragraphs) != 0
    ## this will write output.csv which is half prepared
    df = generate_prompt_from_paragraphs(paragraphs)
    ## this will write output.jsonl, train.jsonl, val.jsonl
    split_training_set(df)
    

def split_dataset():
    df = pd.read_csv('output.csv')
    split_training_set(df)
    ## this will write output.jsonl, train.jsonl, val.jsonl


def test_generate():
    result = generate_text("The God Next Door")
    assert len(result["response"]) != 0
    print(result["response"])
