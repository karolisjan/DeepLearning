import cv2
import imageio
import imutils
import argparse
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '-c',
        '--cascade_path',
        required=True,
        help='path to the face haar cascade'
    )

    ap.add_argument(
        '-m',
        '--model_path',
        required=True,
        help='path to a trained CNN smile model'
    )

    ap.add_argument(
        '-v',
        '--video_path',
        help='path to a video file (optional)'
    )

    ap.add_argument(
        '-o',
        '--output_path',
        help='path to save the output to (optional)'
    )

    args = ap.parse_args()

    face_detector = cv2.CascadeClassifier(args.cascade_path)
    model = load_model(args.model_path)
    camera = cv2.VideoCapture(0) if 'video_path' not in args else cv2.VideoCapture(args.video_path)
    output = []

    while True:
        _, frame = camera.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_copy = frame.copy()

        # Detect faces
        rectangles = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Extract faces, preprocess, and predict
        for x, y, w, h in rectangles:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype(float) / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict and create a label
            predictions = model.predict(roi)[0]
            prediction = predictions.argmax()
            p = 100 * predictions[prediction]
            label = 'Smiling (%.2f%%)' % p if prediction else 'Not smiling (%.2f%%)' % p

            cv2.putText(
                frame_copy,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, 
                (0, 0, 255),
                2
            )

            cv2.rectangle(
                frame_copy,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

        cv2.imshow('Frame', frame_copy)
        output.append(frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if 'output_path' in args:
        with imageio.get_writer(args.output_path, mode='I') as writer:
            for frame in output:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame)

    camera.release()
    cv2.destroyAllWindows()