# Usage
from src.helpers.data_prep import parse_into_paragraphs


file_path = '../data/thegodnextdoor.txt'
paragraphs = parse_into_paragraphs(file_path)

start_paragraph = 20

print(paragraphs[start_paragraph])