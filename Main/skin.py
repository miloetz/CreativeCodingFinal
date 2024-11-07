import cv2
import numpy as np

def detect_skin(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

skin_texture = cv2.imread("C:/Users/miloe/CreativeCodingFinal/assets/img/images.png")

if skin_texture is None:
    print("Error: Could not load skin texture image.")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    skin_mask = detect_skin(frame)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        warped_texture = cv2.resize(skin_texture, (w, h))
        roi = frame[y:y+h, x:x+w]
        roi = cv2.bitwise_and(roi, roi, mask=skin_mask[y:y+h, x:x+w])
        roi += cv2.bitwise_and(warped_texture, warped_texture, mask=cv2.bitwise_not(skin_mask[y:y+h, x:x+w]))
        frame[y:y+h, x:x+w] = roi

    cv2.imshow("Skin Mapping", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()