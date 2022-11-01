import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import math

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start streaming
pipeline.start(config)

list_of_position = []
speaker_area = (0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # If depth and color resolutions are different, resize color image to match depth image for display
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        find_hand = mp.solutions.hands.Hands(max_num_hands=1)
        draw = mp.solutions.drawing_utils
        hand = find_hand.process(color_image)
        if hand.multi_hand_landmarks:
            for handLms in hand.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = color_image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    speaker_area = (cx, cy)
                draw.draw_landmarks(color_image, handLms, mp.solutions.hands.HAND_CONNECTIONS)

        faces = face_cascade.detectMultiScale(gray, 1.1, 10)

        nearest = (0,0)
        nearest_x = 0
        nearest_y = 0
        nearest_w = 0
        nearest_h = 0

        for (x, y, w, h) in faces:
            if math.dist(speaker_area, (x, y)) < math.dist(speaker_area, nearest):
                nearest = (x, y)
                nearest_x = x
                nearest_y = y
                nearest_w = w
                nearest_h = h


        cv2.rectangle(color_image, nearest, (nearest_x + nearest_w, nearest_y + nearest_h), (255, 0, 0), 2)
        list_of_position.append(nearest)
        zDepth = depth_frame.get_distance(nearest_x, nearest_y)
        cv2.putText(
            color_image,
            "x:" + str(nearest_x) + ", y:" + str(nearest_y) + ", z:" + str(zDepth),
            (nearest_x + 5, nearest_y - 5),
            font,
            0.7,
            (255, 255, 255),
            2
        )

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()