import cv2
from fast_alpr import ALPR
import time

# Initialize the ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# Open the webcam (use 0 for the default webcam, or specify the index for another camera)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./demo3.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

# Initialize variables for FPS calculation
fps = 0
frame_count = 0
fps_start_time = time.time()

try:
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Start timing for the current frame
        start_time = time.time()

        # Draw predictions on the frame
        annotated_frame = alpr.draw_predictions(frame)

        # End timing for the current frame
        end_time = time.time()

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()

        # Display FPS on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the result
        cv2.imshow("ALPR Result", annotated_frame)

        # Print execution time (optional)
        print(f"Execution Time: {end_time - start_time:.2f} seconds")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

