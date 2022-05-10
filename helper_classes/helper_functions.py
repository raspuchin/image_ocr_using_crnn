import shutil
import os
import torch
from tqdm import tqdm
import numpy as np


def create_alphabet_map(working_dir):
    with open(working_dir) as f:
        alphabet = f.readline()
    print(alphabet)

    # Map the characters in the alphabet to the index
    alphabet_map = {}
    for i, char in enumerate(alphabet):
        # The index of blank in CTCLoss should be zero.
        # The first one in the alphabet has been left blank,
        # and there is no need for special operation here
        alphabet_map[char] = i
    print(alphabet_map)

    return alphabet, alphabet_map


def pre_process_img_file(label_file, img_dir, new_dir, alphabet):
    """Process image file from img_dir to new_dir

    Args:
        label_file: The path of label file, each line like "xxx.jpg 1 2 3 4..."
        img_dir: Origin image files folder.
        new_dir: New folder for save images after processed.
        alphabet: dict of alphabets
    """

    img_names = []
    labels = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_names.append(line.strip('\n').split(' ')[0].split('/')[1])
            idxs = line.strip('\n').split(' ')[1:]
            labels.append(''.join([alphabet[int(idx)] for idx in idxs]))
    print('image count:', len(img_names), ', label count:', len(labels))

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)

    os.mkdir(new_dir)

    for idx, img_name in enumerate(tqdm(img_names)):
        img_path = os.path.join(img_dir, img_name)
        new_path = os.path.join(new_dir, img_name.split('_')[0] + '_' + labels[idx] + '.jpg')
        shutil.copyfile(img_path, new_path)


def get_ctcloss_parameters(text_batch, alphabet_map, preds, batch_size):
    """Convert the real text batch into three parameters required by ctcloss,
    encoded text/predict length/real length

    Args:
        text_batch: real text batch, like('E-Z-4', 'EMD-6-04')

    Returns:
        encoded_text: encode text by alphabet_map
        preds_length: (time step x batch_size) => (51 * batch_size)
        actual_length: length of text to index，max(len(text)) * batch_size
        alphabet_map:
    """
    actual_length = []
    result = []
    for item in text_batch:
        actual_length.append(len(item))
        r = []
        for char in item:
            index = alphabet_map[char]
            r.append(index)
        result.append(r)

    max_len = 0
    for r in result:
        if len(r) > max_len:
            max_len = len(r)

    result_temp = []
    for r in result:
        for i in range(max_len - len(r)):
            r.append(0)
        result_temp.append(r)

    encoded_text = result_temp
    encoded_text = torch.LongTensor(encoded_text)
    preds_length = torch.LongTensor([preds.size(0)] * batch_size)
    actual_length = torch.LongTensor(actual_length)
    return encoded_text, preds_length, actual_length


def get_final_pred(text):
    """Remove adjacent duplicate characters

    Args:
        text: Do argmax after crnn net ouput

    Returns:
        final_text: Text removed adjacent duplicate characters
    """
    text = list(text)
    for i in range(len(text)):
        for j in range(i + 1, len(text)):
            if text[j] == ' ':
                break
            else:
                if text[j] == text[i]:
                    text[j] = ' '
                else:
                    continue
    final_text = ''.join(text).replace(' ', '')
    return final_text


def predict(net, X, y, alphabet):
    """Predict batch images, print predict result and ground truth.

    Args:
        net: crnn net
        X: batch images
        y: batch actual texts
        alphabet:
    """
    preds = net(X)
    _, preds = preds.max(2)
    idx = 0
    print('crnn net output'.ljust(51), '|', 'final predict'.ljust(20), '|', 'ground truth'.ljust(20))
    print('=' * 99)
    for pred in preds.permute(1, 0):
        pred_text = ''.join([alphabet[i.item()] for i in pred])
        print(pred_text, '|', get_final_pred(pred_text).ljust(20), '|', y[idx].ljust(20))
        print('·' * 99)
        idx += 1

    return preds