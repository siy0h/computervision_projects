import cv2
import numpy as np

cap = cv2.VideoCapture(0)

backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2)
kernel = np.ones((3, 3), np.uint8)
kernel2 = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range you want to make invisible
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    color_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply background subtraction
    fgmask = backgroundObject.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel2, iterations=6)

    # Combine color mask and foreground mask
    combined_mask = cv2.bitwise_and(color_mask, fgmask)

    # Create the invisibility effect
    inverted_mask = cv2.bitwise_not(combined_mask)
    background = backgroundObject.getBackgroundImage()

    if background is not None:
        invisible_part = cv2.bitwise_and(background, background, mask=combined_mask)
        visible_part = cv2.bitwise_and(frame, frame, mask=inverted_mask)
        result = cv2.add(invisible_part, visible_part)

        cv2.imshow('Invisibility Effect', result)
    else:
        cv2.imshow('Waiting for background', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
