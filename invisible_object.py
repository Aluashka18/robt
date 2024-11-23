#robt310
#team 9
#Members: Aliya Shalabayeva, Alua Toktassyn, Nazerke Izbassarova, Tomiris Mukanova

import cv2
from ultralytics import YOLO
import numpy as np
import time

# Initialize the video capture and model
cap = cv2.VideoCapture(0)
time.sleep(3)
print("Capturing the background... Stay out of the frame.")

# Capture a background frame
for i in range(30):
    ret, background = cap.read()
background = np.flip(background, axis=1).astype(np.uint8)  # Ensure uint8 type

model = YOLO("yolov8m-seg.pt")  # Load the segmentation model
CONFIDENCE_THRESHOLD = 0.5

# Define the COCO classes if not already defined
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

# Apply CLAHE for brightness and contrast consistency
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
prev_gray = None
prev_points = None

# Apply CLAHE to the background once
background = apply_clahe(background)

# Check that background is valid
if background is None or background.size == 0:
    raise ValueError("Background capture failed. Please try again.")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = np.flip(img, axis=1).astype(np.uint8)  # Ensure uint8 type
    img = apply_clahe(img)  # Apply CLAHE to current frame for consistency

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None and prev_points is not None:
        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        # Update previous frame and points
        prev_gray = gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)

    results = model(img)
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confidences = np.array(result.boxes.conf.cpu(), dtype="float")

    # Check if masks are available before processing
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # Convert mask data to a NumPy array
    else:
        masks = None

    if masks is not None:
        for cls, conf, mask in zip(classes, confidences, masks):
            
            if conf >= CONFIDENCE_THRESHOLD and COCO_CLASSES[cls] == "bottle":
                # Ensure mask is a binary mask in uint8 format
                mask = (mask.squeeze() > 0.5).astype(np.uint8) * 255

                # Dilate the mask to expand the invisible region by 1-2 cm
                kernel = np.ones((15, 15), np.uint8)  # Adjust kernel size to control the dilation amount
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                # Resize the dilated mask to match the full image size
                mask_resized = cv2.resize(dilated_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Blend the background with the dilated mask
                blended_background = cv2.bitwise_and(background, background, mask=mask_resized)
                apple_invisible = cv2.bitwise_and(img, img, mask=255 - mask_resized)
                result_image = cv2.add(blended_background, apple_invisible)

                # Apply bilateral filter to the blended result
                result_image = cv2.bilateralFilter(result_image, d=9, sigmaColor=75, sigmaSpace=75)

                # Apply Gaussian blur to blend the entire frame smoothly
                result_image = cv2.GaussianBlur(result_image, (11, 11), 0)

                # Update the main image with the invisibility effect
                img = result_image

                # Initialize tracking points for optical flow (only once per apple detection)
                if prev_gray is None or prev_points is None:
                    prev_gray = gray.copy()
                    prev_points = cv2.goodFeaturesToTrack(gray, mask=mask_resized, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Display the result
    cv2.imshow('Display', img)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
