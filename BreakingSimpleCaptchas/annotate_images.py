import os
import sys
import cv2
import imutils
import argparse
from tqdm import tqdm
from glob import glob


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '-i',
        '--input',
        help='path to input directory of images',
        required=True
    )

    ap.add_argument(
        '-o',
        '--output',
        help='path to output directory of annotated images',
        required=True
    )

    args = vars(ap.parse_args())

    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    digit_counts = {}
    imagepaths = glob('/'.join([args['input'], '*']))

    with tqdm(total=len(imagepaths)) as pbar:
        for i, imagepath in enumerate(imagepaths):
            try:
                imagename = os.path.basename(imagepath)

                print('Processing "%s"...' % imagename)

                image = cv2.imread(imagepath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Pads gray image with 8 pixels in every direction.
                # Padding will prevent numbers from touching the borders
                # which simplifies extraction
                gray = cv2.copyMakeBorder(
                    gray,
                    8, 8, 8, 8,
                    cv2.BORDER_REPLICATE
                )

                # Binarise image via Otsu's thresholding method
                threshold = cv2.threshold(
                    gray, 
                    0,
                    255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1]

                # Find the outlines of the image
                contours = cv2.findContours(
                    threshold.copy(),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                contours = contours[0] if imutils.is_cv2() else contours[1]

                # Sort the contours and keep the largerst 4 based on the area,
                # just in case there is noise in the contours
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

                for contour in contours:
                    # Comput bounding box
                    (x, y, w, h) = cv2.boundingRect(contour)

                    # Find the region of interest (ROI)
                    # 5 is for extra padding
                    roi = gray[
                        (y - 5):(y + h + 5), 
                        (x - 5):(x + w + 5)
                    ]

                    cv2.imshow('ROI', imutils.resize(roi, width=28))
                    key = cv2.waitKey(0)

                    # Ignore the extraced digit
                    if key == ord('s') or key == ord('S'):
                        print('\rProcessing "%s", skipping digit...' % imagename)
                        continue

                    if key == ord('q') or key == ord('Q'):
                        print('\rExiting...')
                        raise SystemExit

                    key = chr(key).upper()
                    path = os.path.sep.join([args['output'], key])

                    if not os.path.exists(path):
                        os.mkdir(path)

                    digit_count = digit_counts.get(key, 1)
                    digitpath = os.path.sep.join([
                        path, 
                        '{}.png'.format(
                            str(digit_count).zfill(6)
                        )
                    ])

                    cv2.imwrite(digitpath, roi)
                    digit_counts[key] = digit_count + 1

                    print('\rProcessing "{}", annoted digit {}'.format(imagename, key))

            except KeyboardInterrupt as keyboard_interrupt:
                break
            except SystemExit as sys_exit:
                break
            except BaseException as e:
                tb = sys.exc_info()[-1]
                f = tb.tb_frame
                error_data = {
                    'error': {
                        'message': str(e),
                        'file': str(f.f_code.co_filename),
                        'line': tb.tb_lineno,
                    }
                }
                print('Error "{}", skipping image...'.format(error_data))