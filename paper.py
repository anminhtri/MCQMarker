import cv2
import numpy as np
import sys
import os

image_path = 'C:/Users/stdso/Documents/USTH/intern/Model_main/frame_0001.jpg'

if not os.path.isfile(image_path):
    print(f"Error: The image file was not found at {image_path}")
    sys.exit()

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image. Please check if the file is corrupted.")
    sys.exit()

orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 50, 150)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

screen_contour = None

for contour in contours:
    # Approximate the contour
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # If the approximated contour has four points, we can assume it's our paper
    if len(approx) == 4:
        screen_contour = approx
        break

# If a contour with four points was found
if screen_contour is not None:
    # Draw the contour on the original image
    cv2.drawContours(orig, [screen_contour], -1, (0, 255, 0), 3)

    # Extract the corner points
    pts = screen_contour.reshape(4, 2)
    print("Corner Points:")
    for (x, y) in pts:
        print(f"({x}, {y})")
        cv2.circle(orig, (x, y), 5, (255, 0, 0), -1)

    # Show the final image with corners detected
    cv2.imshow('A4 Paper Detected', orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("A4 paper not detected.")
