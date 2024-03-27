
from helpers.finetune_mistral import run_finetune_mistral
from helpers.infer import infer
from helpers.finetune_phi import run_finetune


file_path = './data/runs/phi2-journal-finetune'
train_path = './data/exc_mistral_prep_1.csv'
def test_train_phi():
    run_finetune(train_path)
    
def test_train_mistral():
    run_finetune_mistral(train_path)
    
def test_infer():
    run_number = 950
    run_path = file_path + "/checkpoint-" + str(run_number)
    result = infer(run_path, "The following is generated scifi content: the grey area dropped out of hyperspace very quickly and ")
    print (result)