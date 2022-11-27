import cv2
import numpy as np
import mediapipe_utils as mpu
from FPS import FPS
import warnings

# LINES_*_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
LINES_FULL_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27], 
                    [23,24],
                    [22,16,18,20,16,14,12], 
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]

class Blazepose:
    def __init__(self, 
                 pd_model=None, 
                 pd_score_thresh=0.5, 
                 lm_model=None, 
                 lm_score_thresh=0.5, 
                 lm_show=True):
        
        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh

        # The full body landmark model predict 39 landmarks.
        # We are interested in the first 35 landmarks 
        # from 1 to 33 correspond to the well documented body parts,
        # 34th (mid hips) and 35th (a point above the head) are used to predict ROI of next frame
        self.nb_lms = 35

        self.points = np.zeros((self.nb_lms, 3))

        self.list_of_lines = LINES_FULL_BODY
        self.lm_show = lm_show

        self.filter = mpu.LowPassFilter(alpha=0.6)

        # Load tflite models
        self.load_models(pd_model_path=pd_model, lm_model_path=lm_model)

        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt

        anchor_options = mpu.SSDAnchorOptions(
                                num_layers=5, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=224,
                                input_size_width=224,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 32, 32, 32],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)

        self.anchors = mpu.generate_anchors(anchor_options)

        self.fps = FPS()

        warnings.filterwarnings('ignore')

    def load_models(self, pd_model_path, lm_model_path):

        # Pose detection model
        # Input blob: input_1 - shape: [1, 3, 224, 224] ---> [0]
        # Output blob: Identity - shape: [1, 2254, 12] ---> [0]
        # Output blob: Identity_1 - shape: [1, 2254, 1] --->[1]

        print("Loading pose detection model")
        self.pd_interpreter = cv2.dnn.readNetFromTensorflow(pd_model_path)

        # self.pd_interpreter.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.pd_interpreter.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        self.pd_h,self.pd_w = (224, 224)
        self.pd_scores = 1
        self.pd_bboxes = 0

        # Landmarks model
        # Input blob: input_1 - shape: [1, 3, 256, 256] ---> [0]
        # Output blob: Identity - shape: [1, 195] --->[0]
        # Output blob: Identity_1 - shape: [1, 1] --->[1]

        print("Loading landmark model")
        self.lm_interpreter = cv2.dnn.readNetFromTensorflow(lm_model_path)

        # self.lm_interpreter.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.lm_interpreter.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.lm_h,self.lm_w = (256, 256)
        self.lm_score = 1
        self.lm_landmarks = 0
    
    def pd_postprocess(self, results):
        scores = np.squeeze(results[self.pd_scores])  # 2254
        bboxes = results[self.pd_bboxes][0] # 2254x12
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=True)
        
        mpu.detections_to_rect(self.regions, kp_pair=[0,1])
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)
    
    def lm_postprocess(self, region, lm_results):
        region.lm_score = np.squeeze(lm_results[self.lm_score])
        if region.lm_score > self.lm_score_thresh:  

            lm_raw = lm_results[self.lm_landmarks].reshape(-1,5)
            # Each keypoint have 5 information:
            # - X,Y coordinates are local to the region of
            # interest and range from [0.0, 255.0].
            # - Z coordinate is measured in "image pixels" like
            # the X and Y coordinates and represents the
            # distance relative to the plane of the subject's
            # hips, which is the origin of the Z axis. Negative
            # values are between the hips and the camera;
            # positive values are behind the hips. Z coordinate
            # scale is similar with X, Y scales but has different
            # nature as obtained not via human annotation, by
            # fitting synthetic data (GHUM model) to the 2D
            # annotation.
            # - Visibility, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame and not occluded by another bigger body
            # part or another object.
            # - Presence, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame.

            # Normalize x,y,z. Here self.lm_w = self.lm_h and scaling in z = scaling in x = 1/self.lm_w
            lm_raw[:,:3] /= self.lm_w
            # Apply sigmoid on visibility and presence (if used later)
            # lm_raw[:,3:5] = 1 / (1 + np.exp(-lm_raw[:,3:5]))
            
            # region.landmarks contains the landmarks normalized 3D coordinates in the relative oriented body bounding box
            region.landmarks = lm_raw[:,:3]
            # Calculate the landmark coordinate in square padded image (region.landmarks_padded)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(region.landmarks[:self.nb_lms,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
            # A segment of length 1 in the coordinates system of body bounding box takes region.rect_w_a pixels in the
            # original image. Then I arbitrarily divide by 4 for a more realistic appearance.
            lm_z = region.landmarks[:self.nb_lms,2:3] * region.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))
            lm_xyz = self.filter.apply(lm_xyz)
            region.landmarks_padded = lm_xyz.astype(int)
            # If we added padding to make the image square, we need to remove this padding from landmark coordinates
            # region.landmarks_abs contains absolute landmark coordinates in the original image (padding removed))
            region.landmarks_abs = region.landmarks_padded.copy()
            if self.pad_h > 0:
                region.landmarks_abs[:,1] -= self.pad_h
            if self.pad_w > 0:
                region.landmarks_abs[:,0] -= self.pad_w
            
            self.points = region.landmarks_abs
    
    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_thresh:
            
            list_connections = self.list_of_lines
            lines = [np.array([region.landmarks_abs[point,:2] for point in line]) for line in list_connections]
            cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
            
            for i,x_y in enumerate(region.landmarks_abs[:self.nb_lms-2,:2]):
                cv2.circle(frame, (x_y[0], x_y[1]), 4, (0,0,255), -11)

    def set_frame_shape(self, frame_width=640, frame_height=480):

        w = frame_width
        h = frame_height

        self.frame_size = max(w, h)
        self.pad_h = int((self.frame_size - h)/2)
        self.pad_w = int((self.frame_size - w)/2)
    
    def get_annotated_frame(self):
        return self.annotated_frame
    
    def get_keypoints(self):
        return self.points
    
    def pose_estimation(self, frame):

        self.annotated_frame = frame.copy()
        squared_frame = cv2.copyMakeBorder(frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

        # Infer pose detection
        # Resize image to NN square input shape
        frame_nn = cv2.resize(squared_frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
        frame_nn = cv2.cvtColor(frame_nn, cv2.COLOR_BGR2RGB)
        # Transpose hxwx3 -> 1x3xhxw
        frame_nn = np.transpose(frame_nn, (2,0,1))[None,]
        frame_nn = frame_nn.astype('float32') / 255.0

        self.pd_interpreter.setInput(frame_nn)
        pd_results = self.pd_interpreter.forward(["Identity", "Identity_1"])
        self.pd_postprocess(pd_results)

        # Infer landmark detection
        if len(self.regions) == 1:
            r = self.regions[0]
            frame_nn = mpu.warp_rect_img(r.rect_points, squared_frame, self.lm_w, self.lm_h)
            frame_nn = cv2.cvtColor(frame_nn, cv2.COLOR_BGR2RGB)
                # Transpose hxwx3 -> 1x3xhxw
            frame_nn = np.transpose(frame_nn, (2,0,1))[None,]
            frame_nn = frame_nn.astype('float32') / 255.0

            # Get landmarks
            self.lm_interpreter.setInput(frame_nn)
            lm_results = self.lm_interpreter.forward(["Identity", "Identity_1"])
            self.lm_postprocess(r, lm_results)

            if self.lm_show:
                self.lm_render(self.annotated_frame, r)
        
        self.fps.update()
        if self.lm_show:
            self.fps.draw(self.annotated_frame, orig=(100,50), size=1, color=(0,0,255))

        