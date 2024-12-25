import joblib
from jetson_utils import videoSource, videoOutput
from skimage.feature import hog
import cv2
import numpy as np
import sys

def extract_hog_features(image):
    features, hog_image = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    return features, hog_image

# 初始化输入和输出流
try:
    input_stream = videoSource("csi://0?width=1280&height=720&framerate=30", argv=sys.argv)
    output_stream = videoOutput("", argv=sys.argv)
except Exception as e:
    print(f"Error initializing video streams: {e}")
    sys.exit(1)

knn = joblib.load('./best_kNN_model.joblib')

while True:
    try:
        frame = input_stream.Capture()
        if frame is None:
            print("Timeout or no frame captured. Retrying...")
            continue
        image = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (64, 64))
        features, hog_image = extract_hog_features(frame)
        test = np.array(features)

        label = knn.predict([test])
        print(label)

        cv2.rectangle(image, (0, 0), (200, 40), (0, 0, 0), -1)
        cv2.putText(image, str(label[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output_stream.Render(image)
        output_stream.SetStatus("Camera Test | FPS: {:.2f}".format(input_stream.GetFrameRate()))

    except Exception as e:
        print(f"Error during frame processing: {e}")
        break

    if not input_stream.IsStreaming() or not output_stream.IsStreaming():
        print("Stream ended.")
        break

print("Camera test finished.")
