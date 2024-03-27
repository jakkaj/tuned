
from helpers.infer import infer
from helpers.finetune import run_finetune


file_path = './data/runs/phi2-journal-finetune'

def test_train():
    run_finetune(file_path)
    
def test_infer():
    run_number = 950
    run_path = file_path + "/checkpoint-" + str(run_number)
    result = infer(run_path, "The following is generated scifi content: the grey area dropped out of hyperspace very quickly and ")
    print (result)