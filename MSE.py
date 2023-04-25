import numpy as np
import cv2


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE value
    return err


# load the images
original = cv2.imread(f"Gt/cotton_gt_disp.png")
pred = cv2.imread(f"OACC-Net/hci/cotton.png")


# convert the images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

print('MSE: %.4f' % mse(original_gray, pred_gray))



