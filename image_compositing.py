import cv2 as cv

BUMP_MAP = 'outputs/smooth_weave_bump_map.png'
HEIGHT_MAP = "outputs/motif-4_filled_height.png"
ROUGHNESS_MAP = "outputs/weave_roughness_map.png"
BASE_MAP = "outputs/motif-4_filled.png"

def subtract():
    img1 = cv.imread(ROUGHNESS_MAP)
    img2 = cv.imread(HEIGHT_MAP)

    if img1 is None or img2 is None:
        print("Error: Could not load one or both images. Check file paths.")
        return

    print(f"Img1 shape: {img1.shape}, Img2 shape: {img2.shape}")

    # FIX 1: cv.resize expects (width, height), but .shape gives (height, width)
    target_size = (img2.shape[1], img2.shape[0])
    img1 = cv.resize(img1, target_size)

    # FIX 2: Match channels if one is grayscale and the other is color
    # Note: cv.imread loads images as 3-channel BGR by default unless cv.IMREAD_GRAYSCALE is specified.
    if len(img2.shape) != len(img1.shape):
        if len(img1.shape) == 3 and len(img2.shape) == 2:
            # Convert img2 from grayscale to BGR color
            img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            # Convert img2 from color to grayscale
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
            
    # Perform the subtraction (OpenCV automatically clamps values at 0 to prevent underflow)
    subtracted = cv.subtract(img1, img2)

    cv.imwrite("outputs/subtracted_roughness.png", subtracted)
    print("Success: Saved to outputs/subtracted_roughness.png")

if __name__ == "__main__":
    subtract()