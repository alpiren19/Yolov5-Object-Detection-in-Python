import time
import cv2
import argparse
from object_detection import ObjectDetection



def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video","-v_path",type=str,help='Video path to use')
    parser.add_argument("--object_detection","-od",default=False,action="store_true",help='Activates Object Detection(True or False)')
    args = parser.parse_args()
    return args

def demo_function():
    args = parse_argument()

    detect_object = ObjectDetection(args)
    detection_enabled = True

    pause = False
    thickness = 1
    player = cv2.VideoCapture(args.input_video)
    assert player.isOpened()

    cv2.namedWindow("output",cv2.WINDOW_GUI_NORMAL)

    while True:
        k = cv2.waitKey(1)
        ret,frame = player.read()
        if k ==27 or k==ord('q'):
            break
        elif k == 32:
            pause = not pause
        
        if not pause:
            ret,frame = player.read()

            if frame.shape[0] >= 1080:
                thickness = 2
            if frame.shape[0] < 1080:
                thickness = 1

            if not ret:
                break

            if k == ord('d'):
                detection_enabled = not detection_enabled

            if args.object_detection and detection_enabled:
                results = detect_object.detect(frame)
                frame = detect_object.plot_boxes(results,frame,thickness)        

        cv2.imshow("output",frame)


if __name__ == "__main__":
    demo_function()