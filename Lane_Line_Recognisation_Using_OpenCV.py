import cv2
import numpy as np

def detect_lane_lines(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the kernel size for Gaussian smoothing / blurring
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Define the parameters for Canny edge detection
    low_threshold = 60
    high_threshold = 160
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Define the parameters for the region of interest
    imgshape = frame.shape
    vertices = np.array([[(0,imgshape[0]),(450, 325), (500, 325), (imgshape[1],imgshape[0])]], dtype=np.int32)
    mask_edges = region_of_interest(edges, vertices)

    # Define the parameters for Hough line detection
    rho = 1
    theta = np.pi/180
    threshold = 25
    min_line_length = 50
    max_line_gap = 2
    line_image = np.copy(frame)*0

    # Perform Hough line detection on the masked edge image
    lines = cv2.HoughLinesP(mask_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the Hough lines and draw them on the line image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Combine the line image with the original frame
    lane_lines = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return lane_lines


def region_of_interest(img, vertices):
    # Define a mask and apply it on the input image
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Capture video from camera (0) or from file
cap = cv2.VideoCapture("test_video.mp4")

# Loop over the video frames
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Detect the lane lines in the frame
    result = detect_lane_lines(frame)

    # Show the result
    cv2.imshow("Result", result)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
