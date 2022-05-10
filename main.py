from helper_classes.helper_functions import *
from helper_classes.model_classes import *
import torch
from torch.utils.data import DataLoader
import cv2


def main():
    crnn = torch.load('model/crnn.pt')
    crnn.eval()
    alphabet, alphabet_map = create_alphabet_map('input/alphabet.txt')

    test_set = MyDataset(data_dir='preprocessed/test/', alphabet_map=alphabet_map)

    use_gpu = True
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # predict single image with random index
    idx = random.randint(0, len(test_set) - 1)
    X, y = test_set[idx]
    X = X.unsqueeze(0)  # add dim as batch
    y = [y]
    X = X.to(device)
    predict(crnn, X, y, alphabet)
    print('\n' * 2)
    # predict batch using dataloader
    testloader = DataLoader(test_set, batch_size=8, shuffle=True, drop_last=True)
    X, y = next(iter(testloader))
    X_np = np.array(X, dtype=object)
    X = X.to(device)
    predictions = predict(crnn, X, y, alphabet)

if __name__ == '__main__':
    main()
