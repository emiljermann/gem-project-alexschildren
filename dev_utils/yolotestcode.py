import torch
import numpy as np
import cv2


"""
Run this script from the command line with python 
pedestrian_detection.py to test the model with a webcam and 
live output.

Otherwise import model and the function get_pedestrian_box(model, frame) on 
an opencv2 frame object to get the bounding box of the detected
pedestrian.
i.e.
import pedestrian_detection
model = pedestrian_detection.model
...
pedestrian_detection.get_pedestrian_box(model, image)

"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
model.to(device).eval()

def get_pedestrian_box(model, frame):
    """
    Gets pose points from a given YOLO model and frame.
    Args:
        model: YOLO model.
        frame: Frame from the camera.
    Returns:
        bounding_box: bounding box of person with highest confidence, None otherwise
        in format (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
        
        confidence: confidence score that said pedestrian exists
    """
    results = model(frame).pandas().xyxy[0]
    # uncomment to see output of yolo model 
    # print(results) 
    if len(results) == 0:
        return None, 0
    
    xyxy = None
    highest_conf = 0
    for obj in results.itertuples():
        if obj.name == "person":
            boxes = (obj.xmin, obj.ymin, obj.xmax, obj.ymax)
            confidence = obj.confidence
            if confidence > highest_conf:
                highest_conf = confidence
                xyxy = boxes
            

    return xyxy, highest_conf



if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # Works on mac, change to 0 for windows or linux?
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        box, confidence = get_pedestrian_box(model, frame)

        # No frame was found
        if box is None:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Found You! {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Image with Boxes", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    exit()
