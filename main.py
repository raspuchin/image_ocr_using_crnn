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

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    images_shown = 0
    for image, prediction in zip(X_np, predictions):
        if images_shown >= 5:
            break

        print('Original Shape: ' + str(image.shape))
        image = np.array(image).reshape((32, 200, 1))
        print('Image')
        print('Type of image: ' + str(type(image)))
        print(image)

        img = cv2.resize(image, (256, 256))
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
        imgray = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                       blockSize=321, C=28)
        # get the edges in the image
        thresh = cv2.Canny(imgray, 100, 200)
        # get the contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)

            if w * h < 500:
                continue

            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        img = cv2.putText(img, prediction, (256, 20), font, fontScale, color, thickness,
                          cv2.LINE_AA)

        show_image(img)

        images_shown += 1


def show_image(img):
    cv2.imshow('img', img)

    while True:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
