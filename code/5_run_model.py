import tensorflow as tf
import cv2
import numpy as np

def segmentation(frame, threshold=20):
    global background

    difference = cv2.absdiff(background.astype('uint8'), frame)

    _, processed_frame = cv2.threshold(
        difference, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        contours = max(contours, key=cv2.contourArea)

    return processed_frame, contours


cam = cv2.VideoCapture(0)

# roi : region of interest

top = 50

bottom = 300

right = 50

left = 250


background = None

count = 0

digit = 0

sample_number = 0
words = {
    0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
    5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'
}


model2 = tf.keras.models.load_model(
    'SignLanguageModel.hs')


while True:
    value, frame = cam.read()

    frameCopy = frame.copy()

    frameCopy = cv2.flip(frameCopy, 1)

    roi = frameCopy[top:bottom, right:left]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi_blur = cv2.GaussianBlur(roi_gray, (9, 9), 0)

    if background is None:

        background = roi_blur.copy().astype('float')

    cv2.rectangle(frameCopy, (left, top), (right, bottom), (0, 0, 255), 3)

    if count < 30:

        cv2.accumulateWeighted(roi_blur, background, 0.5)

        cv2.putText(frameCopy, 'Loading...', (250, 250),

                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    collection = segmentation(roi_blur)

    k = cv2.waitKey(1)

    if collection is not None:
        roi_processed, contour = collection

        cv2.drawContours(frameCopy, [contour+(right, top)], -1, (0, 255, 0), 2)

        roi_processed = cv2.resize(roi_processed, (64, 64))

        roi_processed = cv2.cvtColor(roi_processed, cv2.COLOR_GRAY2RGB)

        frame = np.reshape(
            roi_processed, (1, roi_processed.shape[0], roi_processed.shape[1], 3))
        value = model2.predict(frame)

        label = words[np.argmax(value)]

        cv2.putText(frameCopy, str(label), (370, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        cv2.imshow('Segmentation', roi_processed)

    count += 1

    cv2.imshow('Camera', frameCopy)

    if k == 27:

        break


cv2.destroyAllWindows()
cam.release()
