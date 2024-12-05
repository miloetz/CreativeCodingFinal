import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

previous_frame = None
silhouette_persistent = None

phrases = [
    "did you hear what i said?",
    "where did he move to?",
    "just try your best.",
    "you're so annoying.",
    "fruit for breakfast.",
    "i didn't get the job.",
    "couldn't be me.",
    "sounds like fun.",
    "i swear, i can never take you seriously.",
    "colored felt & denim.",
    "i can't.",
    "i'm a bad mom.",
    "yeah, it just wasn't for me.",
    "wanna grab lunch tomorrow?",
    "oh my god, and then what happened?",
    "i haven't heard anything back.",
    "i already told you, i don't know.",
    "are you sure you're good to drive?",
    "it didn't mean anything.",
    "your shoes are old.",
    "yeah i'll go.",
    "it's just what we do.",
    "you need a haircut.",
    "NO SIGNAL DETECTED"
]
text_index = 0
text_opacity = 0
fade_in = True
last_text_time = time.time()

cv2.namedWindow("Fuzzy Silhouette", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Fuzzy Silhouette", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if silhouette_persistent is None:
        silhouette_persistent = np.zeros_like(frame)

    if previous_frame is None:
        previous_frame = blurred
        continue

    frame_delta = cv2.absdiff(previous_frame, blurred)
    motion_mask = cv2.threshold(frame_delta, 1, 255, cv2.THRESH_BINARY)[1]
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    silhouette_frame = np.zeros_like(frame)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(silhouette_frame, [contour], -1, (128, 128, 128), -1)

    silhouette_persistent = cv2.addWeighted(silhouette_persistent, 0.95, silhouette_frame, 0.05, 0)
    silhouette_persistent_blurred = cv2.GaussianBlur(silhouette_persistent, (15, 15), 0)

    current_time = time.time()
    if fade_in:
        text_opacity += 5
        if text_opacity >= 255:
            text_opacity = 255
            fade_in = False
            last_text_time = current_time
    else:
        if current_time - last_text_time > 2:
            text_opacity -= 5
            if text_opacity <= 0:
                text_opacity = 0
                fade_in = True
                text_index = (text_index + 1) % len(phrases)

    overlay = silhouette_persistent_blurred.copy()
    text = phrases[text_index]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2

    while True:
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        if text_size[0] <= overlay.shape[1] * 0.9:
            break
        font_scale -= 0.1

    text_x = (overlay.shape[1] - text_size[0]) // 2
    text_y = (overlay.shape[0] + text_size[1]) // 2
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255, int(text_opacity)), thickness)

    silhouette_with_text = cv2.addWeighted(silhouette_persistent_blurred, 1, overlay, text_opacity / 255.0, 0)

    previous_frame = blurred

    cv2.imshow("Fuzzy Silhouette", silhouette_with_text)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
