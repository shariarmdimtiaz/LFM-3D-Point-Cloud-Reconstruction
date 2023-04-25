import cv2
from skimage.metrics import structural_similarity as ssim

# Load the two input images
imageA = cv2.imread(f'GT/boxes_gt_disp.png')
imageB = cv2.imread(f'test_result/2023-04-24/boxes_depth_095123.png')

# Convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# Print the ssim score
print("SSIM: {}".format(score))

