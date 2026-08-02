import cv2 as cv

def subtract(img_path1, img_path2, img_out):
    img1 = cv.imread(img_path1)
    img2 = cv.imread(img_path2)

    if img1 is None or img2 is None:
        print("Error: Could not load one or both images. Check file paths.")
        return

    # resize if not the same size
    target_size = (img2.shape[1], img2.shape[0])
    img1 = cv.resize(img1, target_size)

    #Validation checks
    if len(img2.shape) != len(img1.shape):
        if len(img1.shape) == 3 and len(img2.shape) == 2:
            # Convert img2 from grayscale to BGR color
            img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            # Convert img2 from color to grayscale
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
            
    # Perform the subtraction (OpenCV automatically clamps values at 0 to prevent underflow)
    subtracted = cv.subtract(img1, img2)

    cv.imwrite(img_out, subtracted)
    print("Success: Saved to", img_out)
