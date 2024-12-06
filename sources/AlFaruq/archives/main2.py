import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def align_images(reference_img, target_img, max_features=500, good_match_percent=0.15):
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
    keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_ref, descriptors_target, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points_ref = np.zeros((len(matches), 2), dtype=np.float32)
    points_target = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_ref[i, :] = keypoints_ref[match.queryIdx].pt
        points_target[i, :] = keypoints_target[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points_target, points_ref, cv2.RANSAC)

    # Use homography to warp target image to reference
    height, width, channels = reference_img.shape
    aligned_target = cv2.warpPerspective(target_img, h, (width, height))

    return aligned_target


# Load images
reference_img = cv2.imread("images/absdiff/paper0.png")
target_img = cv2.imread("images/absdiff/paper1b.png")

# Align target image to reference image
aligned_target = align_images(reference_img, target_img)

# Convert to grayscale
gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
gray_aligned = cv2.cvtColor(aligned_target, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
(score, diff) = ssim(gray_ref, gray_aligned, full=True)
diff = (diff * 255).astype("uint8")

# Threshold the difference image
thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)[1]

# Perform morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.dilate(thresh, kernel, iterations=2)
thresh = cv2.erode(thresh, kernel, iterations=1)

# Find contours of the differences
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around detected differences
output = aligned_target.copy()
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    # Filter out small contours if necessary
    if cv2.contourArea(contour) > 100:  # Adjust threshold as needed
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Resize images for display (without stretching, just to fit on screen)
scale_percent = 50  # percent of original size
width = int(reference_img.shape[1] * scale_percent / 100)
height = int(reference_img.shape[0] * scale_percent / 100)
dim = (width, height)

ref_resized = cv2.resize(reference_img, dim, interpolation=cv2.INTER_AREA)
target_resized = cv2.resize(target_img, dim, interpolation=cv2.INTER_AREA)
output_resized = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

# Concatenate images horizontally (left to right)
concatenated_images = cv2.hconcat([ref_resized, target_resized, output_resized])
width = int(concatenated_images.shape[1] * 0.5)
height = int(concatenated_images.shape[0] * 0.5)
concatenated_images = cv2.resize(concatenated_images, (width, height))

# Display the result
cv2.imshow("Comparison", concatenated_images)
cv2.waitKey(0)
cv2.destroyAllWindows()
