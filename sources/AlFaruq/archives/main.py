import cv2
from skimage.metrics import structural_similarity as ssim

reference_img = cv2.imread("images/absdiff/paper0.png")
target_img = cv2.imread("images/absdiff/paper1.png")

gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(gray_ref, gray_target, full=True)
diff = (diff * 255).astype("uint8")

thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.dilate(thresh, kernel, iterations=2)
thresh = cv2.erode(thresh, kernel, iterations=1)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = target_img.copy()
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

width = int(output.shape[1] * 0.5)
height = int(output.shape[0] * 0.5)
output = cv2.resize(output, (width, height))
cv2.imshow("Detected Differences", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
