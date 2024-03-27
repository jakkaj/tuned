from helpers.data_prep import parse_into_paragraphs

def test_load_sample_book():
    file_path = './data/thegodnextdoor.txt'
    paragraphs = parse_into_paragraphs(file_path)
    assert len(paragraphs) != 0

