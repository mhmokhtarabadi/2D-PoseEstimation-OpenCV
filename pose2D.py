import cv2
from Blazepose import Blazepose

POSE_DETECTION_MODEL = "./models/TF/pose_detection_float32.pb"
LANDMARK_MODEL_FULL = "./models/TF/pose_landmark_full_float32.pb"
LANDMARK_MODEL_LITE = "./models/TF/pose_landmark_lite_float32.pb"
LANDMARK_MODEL_HEAVY = "./models/TF/pose_landmark_heavy_float32.pb"

Blaze1 = Blazepose(pd_model=POSE_DETECTION_MODEL, 
                    pd_score_thresh=0.5, 
                    lm_model=LANDMARK_MODEL_HEAVY, 
                    lm_score_thresh=0.5, 
                    lm_show=True)

Blaze1.set_frame_shape()

cap = cv2.VideoCapture(0)

while True:
    ok, img = cap.read()

    if ok:
        Blaze1.pose_estimation(img)

    annotated_frame = Blaze1.get_annotated_frame()
    cv2.imshow("frame", annotated_frame)

    key = cv2.waitKey(1) 
    if key == ord('q') or key == 27:
        break
    elif key == 32:
        # Pause on space bar
        cv2.waitKey(0)
