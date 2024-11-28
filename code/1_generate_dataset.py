import cv2

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


# captureImage
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
        if k == ord(' '):
            cv2.putText(frameCopy, str(sample_number) + 'Gesture(' + str(digit) + ')',
                        (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.imwrite('gesture/train/'+str(digit) +
                        '/'+str(digit)+'-'+str(sample_number)+'.jpg', roi_processed)
            sample_number += 1

        cv2.imshow('Segmentation', roi_processed)

    count += 1
    cv2.putText(frameCopy, 'Gesture( '+str(digit)+' )', (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Camera', frameCopy)

    if k == 27:
        break
    elif k == 13:
        digit = (digit + 1) % 11
        sample_number = 0

cv2.destroyAllWindows()
cam.release()


captureImage()