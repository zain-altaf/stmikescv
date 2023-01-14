# Install the required libraries
import cv2
import tensorflow as tf
import time
import os

# Load the Lightining model
# Load the TensorFlow Lite model
model_path = "posenet_lightining_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)

# Allocate memory for the model
interpreter.allocate_tensors()

# Get the input and output tensors
input_tensor = interpreter.get_input_details()[0]["index"]
output_tensor = interpreter.get_output_details()[0]["index"]

# Capture the background frame
# Initialize the video capturer and set the resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Wait until there is no detection in the frame and save it as the background image
while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Use TensorFlow's Lightining model to detect keypoints and edges in the frame
    input_data = cv2.resize(frame, (257, 353))
    input_data = input_data[np.newaxis, ...].astype(np.float32)
    interpreter.set_tensor(input_tensor, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_tensor)
    keypoints, edges = posenet_lightining.decode_multiple_poses(output_data)

    # If no keypoints or edges are detected, save the frame as the background image
    if keypoints is None or edges is None:
        background_image = frame
        break

# Detect motion in the frame
# Initialize the motion detector
md = cv2.BackgroundSubtractorMOG2()

# Initialize a variable to keep track of the elapsed time since the last motion was detected
elapsed_time = 0

# Loop indefinitely
while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Use the motion detector to detect movement in the frame
    fgmask = md.apply(frame)

    # If movement is detected, reset the elapsed time and set the start time
    if cv2.countNonZero(fgmask) > 0:
        elapsed_time = 0
        start_time = time.time()

    # If there is no movement detected for 1 minute, recapture the background image
    elif elapsed_time >= 60:
        # Wait until there is no detection in the frame and save it as the new background image
        while True:
            ret, frame = cap.read()
            if not ret:
        # If no keypoints or edges are detected, save the frame as the new background image
                if keypoints is None or edges is None:
                    background_image = frame
                    break

                # Reset the elapsed time
                elapsed_time = 0
            else:
                # If movement is not detected and the elapsed time is less than 1 minute, increment the elapsed time
                elapsed_time += 1

            # If movement is detected, proceed to the next step
            if cv2.countNonZero(fgmask) > 0:
                # Detect keypoints and edges
                # Use TensorFlow's Lightining model to detect keypoints and edges in the frame
                input_data = cv2.resize(frame, (257, 353))
                input_data = input_data[np.newaxis, ...].astype(np.float32)
                interpreter.set_tensor(input_tensor, input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_tensor)
                keypoints, edges = posenet_lightining.decode_multiple_poses(output_data)

                # Overlay keypoints and edges on the background image
                overlay_image = cv2.addWeighted(background_image, 0.5, frame, 0.5, 0)
                for person_keypoints in keypoints:
                    for keypoint in person_keypoints:
                        cv2.circle(overlay_image, tuple(keypoint), 3, (0, 0, 255), -1)
                for person_edges in edges:
                    for edge in person_edges:
                        cv2.line(overlay_image, tuple(edge[0]), tuple(edge[1]), (0, 255, 0), 2)
                # Save the resulting frame
                # Create a folder to store the captured frames
                date_str = time.strftime("%Y-%m-%d")
                start_time_str = time.strftime("%I:%M %p", time.localtime(start_time))
                end_time_str = time.strftime("%I:%M %p", time.localtime(time.time()))
                folder_name = f"{date_str}_{start_time_str}-{end_time_str}"
                os.makedirs(folder_name, exist_ok=True)

                # Save the frame in the folder
                frame_time_str = time.strftime("%I:%M %p", time.localtime(time.time()))
                frame_name = f"{frame_time_str}.jpg"
                cv2.imwrite(os.path.join(folder_name, frame_name), overlay_image)
            else:
                # If movement is not detected, set the end time
                end_time = time.time()

        # Release the video capturer
        cap.release()
