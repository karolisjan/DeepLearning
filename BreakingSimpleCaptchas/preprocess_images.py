import cv2
import imutils


def preprocess(img, width, height):
    '''
        Resize and pad images to a fixed size without distorting their aspect ratio.
    '''
    h, w = img.shape[:2]

    if w > h:
        img = imutils.resize(img, width=width)
    else:
        img = imutils.resize(img, height=height)

    pad_w = (width - img.shape[1]) // 2
    pad_h = (height - img.shape[0]) // 2

    img = cv2.copyMakeBorder(
        img, 
        pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_REPLICATE
    )

    img = cv2.resize(img, (width, height))

    return img
