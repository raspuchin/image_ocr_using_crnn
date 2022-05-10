import torch
from torch import nn
from torch.utils.data import DataLoader
from helper_classes.helper_functions import *
from helper_classes.model_classes import *

def main():
    alphabet, alphabet_map = create_alphabet_map('input/alphabet.txt')

    print('start process train data')
    pre_process_img_file(label_file='input/data_train.txt',
                         img_dir='input/train_imgs/',
                         new_dir='preprocessed/train/',
                         alphabet=alphabet)

    print('start process test data')
    pre_process_img_file(label_file='input/data_test.txt',
                         img_dir='input/test_imgs/',
                         new_dir='preprocessed/test/',
                         alphabet=alphabet)

    train_set = MyDataset(data_dir='preprocessed/train/', alphabet_map=alphabet_map)
    batch_size = 64
    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # Check if the input and output shapes meet expectations
    for X, y in trainloader:
        break
    print('input shape:', X.shape)
    crnn = CRNN(num_class=len(alphabet))
    preds = crnn(X)
    print('output shape from CRNNnet:', preds.shape)

    use_gpu = True
    num_epoch = 100

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    crnn.train()
    trainer = torch.optim.Adam(crnn.parameters(), lr=0.001)
    loss = nn.CTCLoss(zero_infinity=True)
    crnn = crnn.to(device)
    loss = loss.to(device)

    for epoch in range(num_epoch):
        for X, y in trainloader:
            X = X.to(device)
            trainer.zero_grad()
            preds = crnn(X)
            encoded_text, preds_length, actual_length = get_ctcloss_parameters(y, alphabet_map, preds, batch_size)

            encoded_text = encoded_text.to(device)
            preds_length = preds_length.to(device)
            actual_length = actual_length.to(device)

            l = loss(preds, encoded_text, preds_length, actual_length) / batch_size
            l.backward()
            trainer.step()
        print('epoch', str(epoch + 1).ljust(10), 'loss:', format(l.item(), '.6f'))

    torch.save(crnn, 'model/crnn.pt')


if __name__ == '__main__':
    main()


