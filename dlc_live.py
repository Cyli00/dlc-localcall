import cv2
import numpy as np
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow import config, predict_videos
from deeplabcut.pose_estimation_tensorflow.core import predict
from utils.mockcamera import virtualcamera

# 加载配置
cfg = config.load_config("/home/cyli00/gitProjects/dlc-localcall/pupil_resnet/pose_cfg.yaml")

# 设置模拟相机 
mock_cam = virtualcamera("/home/cyli00/gitProjects/dlc-localcall/example/pupil.mp4", loop=True, target_fps=30)

# Set up TensorFlow session and load model
tf.compat.v1.reset_default_graph()

# TensorFlow configuration
# Allow memory growth to avoid GPU memory issues
# Configure TensorFlow to use limited GPU memory
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid taking all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally, you can also limit memory
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        # )
    except RuntimeError as e:
        print("GPU configuration error:", e)

# Setup pose prediction
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Create a window to display the tracking
cv2.namedWindow('DeepLabCut Tracking', cv2.WINDOW_NORMAL)

print("Starting real-time tracking. Press 'q' to quit.")

# Frame processing loop
while True:
    ret, frame = mock_cam.read()  # Use like a real camera
    if not ret:
        break
    
    # Process the frame (similar to what happens in GetPoseS function)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processed = predict_videos.img_as_ubyte(frame_rgb)
    
    # Get pose prediction
    pose = predict.getpose(frame_processed, cfg, sess, inputs, outputs)
    pose_flatten = pose.flatten()
    
    # Create a copy of the frame for visualization
    display_frame = frame.copy()
    
    # Define colors for different body parts
    colors = [
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
    ]
    
    # Extract coordinates and confidence
    num_joints = len(cfg["all_joints_names"])
    for p_idx in range(num_joints):
        x_idx, y_idx, conf_idx = p_idx * 3, p_idx * 3 + 1, p_idx * 3 + 2
        x, y, confidence = pose_flatten[x_idx], pose_flatten[y_idx], pose_flatten[conf_idx]
        
        # Draw only if confidence is high enough
        if confidence > 0.5:  # Adjust threshold as needed
            color = colors[p_idx % len(colors)]
            cv2.circle(display_frame, (int(x), int(y)), 7, color, -1)
            cv2.putText(display_frame, f"{cfg['all_joints_names'][p_idx]}: {confidence:.2f}", 
                        (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
    
    # Add FPS information
    fps = mock_cam.get(cv2.CAP_PROP_FPS)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('DeepLabCut Tracking', display_frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("Shutting down...")
sess.close()
mock_cam.release()
cv2.destroyAllWindows()