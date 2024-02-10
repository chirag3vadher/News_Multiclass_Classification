from Utils.Finetune.finetuning import train, validation
from config import tokenizer, training_data, model
from Utils.preprocessing.preprocessing import pre_processor
import torch
import time


EPOCHS = 1
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
MAX_LEN = 512
train_size = 0.8
training_data = training_data
# Creating the dataset and dataloader for the neural network

if __name__ == "__main__":

    #preprocessing section
    training_loader, testing_loader = pre_processor.preprocess_data(tokenizer,MAX_LEN,TRAIN_BATCH_SIZE,VALID_BATCH_SIZE)

    #training section
    for epoch in range(EPOCHS):
        train.train(training_loader,epoch)

    # Validation Section
    print('This is the validation section to print the accuracy and see how it performs')
    print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')
    acc = validation.valid(model, testing_loader)
    print("Accuracy on test data = %0.2f%%" % acc)

    # Save the finetuned model
    output_model_file = f'\Model\\pytorch_distilbert_news_{time.time}.bin'
    output_vocab_file = f'\models\\vocab_distilbert_news{time.time}.bin'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

    print('All files saved')
