import cv2
import imutils
import argparse
import numpy as np
from imutils import contours
from keras.models import load_model
from preprocess_images import preprocess
from keras.preprocessing.image import img_to_array


def break_captcha(imgpath, model):
    img = cv2.imread(imgpath)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(
        gray,
        20, 20, 20, 20,
        cv2.BORDER_REPLICATE
    )
    
    # Binarise image to reveal digits
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find the 4 largest contours
    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    output = cv2.merge([gray] * 3)
    predictions = []

    # Predict digit for each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        roi = gray[
            (y-5):(y + h + 5),
            (x - 5):(x + w + 5)
        ]
        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        
        prediction = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(prediction))

        cv2.rectangle(
            output, 
            (x - 2, y - 2),
            (x + w + 4, y + h + 4),
            (0, 255, 0),
            1
        )

        cv2.putText(
            output,
            str(prediction),
            (x - 5, y - 5),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    return output, predictions


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '-i',
        '--image-path',
        required=True,
        help='path to a captcha image'
    )
    ap.add_argument(
        '-m',
        '--model-path',
        required=True,
        help='path to a model'
    )
    ap.add_argument(
        '-o',
        '--output-path',
        help='path to save the output to'
    )

    args = vars(ap.parse_args())

    model = load_model(args['model_path'])
    output, predictions = break_captcha(args['image_path'], model)

    if 'output_path' in args:
        cv2.imwrite(args['output_path'], output)

    print('Captcha: {}'.format(''.join(predictions)))
    cv2.imshow('Output', output)
    cv2.waitKey()