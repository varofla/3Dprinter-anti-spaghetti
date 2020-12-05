import cv2
from darkflow.net.build import TFNet

options = {"model": "model/model.cfg", "load": "model/model.weights", "labels": "model/labels.txt","threshold": 0.25}

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
tfnet = TFNet(options)

while True:
    _, frame = capture.read()
    result = tfnet.return_predict(frame)
    print(result)
    for box in result:
        pt1 = (box['topleft']['x'], box['topleft']['y'])
        pt2 = (box['bottomright']['x'], box['bottomright']['y'])
        cv2.rectangle(frame, pt1, pt2, [255, 255, 255], 3)
        cv2.putText(frame, "{0}, {1}%".format(box['label'], int(box['confidence']*100)), pt2, cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0])
    cv2.imshow("VideoFrame", frame)
    # cv2.imshow('result', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()