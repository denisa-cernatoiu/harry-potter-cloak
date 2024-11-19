import cv2
import numpy as np
import time

def create_background(capture, num_frames=30):
    print('Please capture ONLY the background!')

    backgrounds = []
    for i in range(num_frames):
        ret, frame = capture.read()
        if ret:
            backgrounds.append(frame)
        else:
            print(f"Warning: Could not read frame {i+1}/{num_frames}")
        time.sleep(0.1)
    if backgrounds:
        return np.median(backgrounds, axis = 0).astype(np.uint8)
    else:
        raise ValueError("Could not capture any frame for background")
    
#a function to identify where the cloak is in each frame
def create_mask(frame, lower_color, upper_color):
    #converting the input frame from BGR to HSV - HSV is better for segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #creating the mask
    mask = cv2.inRange(hsv, lower_color, upper_color)
    #removing small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    #dilating the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations=1)
    return mask

def apply_cloak_effect(frame, mask, background):
    mask_inv = cv2.bitwise_not(mask)
    #appling the inverted mask to the frame
    fg = cv2.bitwise_and(frame, frame, mask = mask_inv)
    #appling the original mask to the background
    bg = cv2.bitwise_and(background, background, mask = mask)
    return cv2.add(fg, bg)

def main():
    print("OpenCV version:", cv2.__version__)

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        background = create_background(capture)
    except ValueError as e:
        print(f"Error: {e}")
        capture.release()
        return
    
    # pink hsv values, not bgr
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])

    print("Starting main loop! Press 'q' to quit.")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Could not read frame.")
            time.sleep(1)
            continue

        mask = create_mask(frame, lower_pink, upper_pink)
        result = apply_cloak_effect(frame, mask, background)

        cv2.imshow('Invisible Cloak', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()