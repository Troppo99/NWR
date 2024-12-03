import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def align_images_ecc(reference_img, target_img, warp_mode=cv2.MOTION_AFFINE, number_of_iterations=5000, termination_eps=1e-10):
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Define the motion model
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        print("Mulai proses penyelarasan menggunakan metode ECC...")
        (cc, warp_matrix) = cv2.findTransformECC(gray_ref, gray_target, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)
        print(f"Koefisien korelasi: {cc}")
    except cv2.error as e:
        print("Error saat menyelaraskan gambar:", e)
        return None  # Mengembalikan None jika penyelarasan gagal

    # Use warp_matrix to warp the target image to align with the reference
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned_target = cv2.warpPerspective(target_img, warp_matrix, (reference_img.shape[1], reference_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned_target = cv2.warpAffine(target_img, warp_matrix, (reference_img.shape[1], reference_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    print("Penyelarasan gambar selesai.")
    return aligned_target


# Load images
reference_img = cv2.imread("images/absdiff/paper0.png")
target_img = cv2.imread("images/absdiff/paper1.png")

# Periksa apakah gambar berhasil dimuat
if reference_img is None:
    print("Error: Gambar referensi tidak ditemukan atau tidak dapat dibaca.")
    exit(1)
if target_img is None:
    print("Error: Gambar target tidak ditemukan atau tidak dapat dibaca.")
    exit(1)

# Align target image to reference image using ECC
aligned_target = align_images_ecc(reference_img, target_img, warp_mode=cv2.MOTION_AFFINE)

# Periksa apakah penyelarasan berhasil
if aligned_target is None:
    print("Penyelarasan gambar gagal. Memeriksa penyebabnya.")
    exit(1)

# Tampilkan gambar referensi dan gambar yang telah diselaraskan
cv2.imshow("Gambar Referensi", reference_img)
cv2.imshow("Gambar Target yang Diselaraskan", aligned_target)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
gray_aligned = cv2.cvtColor(aligned_target, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
print("Menghitung SSIM antara gambar referensi dan yang diselaraskan...")
(score, diff) = ssim(gray_ref, gray_aligned, full=True)
print(f"Skor SSIM: {score}")

diff = (diff * 255).astype("uint8")

# Threshold the difference image
print("Melakukan thresholding pada gambar perbedaan...")
thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV)[1]

# Perform morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("Melakukan operasi morfologi (dilasi dan erosi)...")
thresh = cv2.dilate(thresh, kernel, iterations=2)
thresh = cv2.erode(thresh, kernel, iterations=1)

# Find contours of the differences
print("Mencari kontur dari area perbedaan...")
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

# Resize concatenated image for display if necessary
display_scale_percent = 80  # Adjust as needed
display_width = int(concatenated_images.shape[1] * display_scale_percent / 100)
display_height = int(concatenated_images.shape[0] * display_scale_percent / 100)
concatenated_images = cv2.resize(concatenated_images, (display_width, display_height))

# Display the result
cv2.imshow("Comparison", concatenated_images)
cv2.waitKey(0)
cv2.destroyAllWindows()
