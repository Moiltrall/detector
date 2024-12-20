import sys
import os
import dlib
import glob
import cv2
import numpy as np

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the"
        " python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # Detect faces in the image
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

        # Get the landmarks/parts for the face in box d
        shape = predictor(img, d)

        # Convert the dlib shape to a numpy array for easier processing
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        # Define the indices for eye landmarks
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))

        # Draw rectangles over the eyes
        left_eye_bbox = cv2.boundingRect(landmarks[left_eye_indices])
        right_eye_bbox = cv2.boundingRect(landmarks[right_eye_indices])

        # Add rectangles to mask the eyes
        cv2.rectangle(img_bgr, (left_eye_bbox[0], left_eye_bbox[1]),
                      (left_eye_bbox[0] + left_eye_bbox[2], left_eye_bbox[1] + left_eye_bbox[3]),
                      (0, 0, 0), -1)  # Black rectangle
        cv2.rectangle(img_bgr, (right_eye_bbox[0], right_eye_bbox[1]),
                      (right_eye_bbox[0] + right_eye_bbox[2], right_eye_bbox[1] + right_eye_bbox[3]),
                      (0, 0, 0), -1)  # Black rectangle

        win.add_overlay(shape)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    win.set_image(img_rgb)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

    # Display the result
    # cv2.imshow("Masked Image", img_bgr)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
