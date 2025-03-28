import os
import cv2
import tensorflow as tf

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.util import visualize
from utils.mockcamera import virtualcamera, select_extract_roi
from utils.analyze_live import analyze_single_frame

# 配置路径
config_path = "/home/cyli00/gitProjects/dlc-localcall/pupil_resnet/pose_cfg.yaml"
video_path = "/home/cyli00/gitProjects/dlc-localcall/example/pupil.mp4"

# 读取配置
cfg = load_config(config_path)

# 设置模拟相机
vcam = virtualcamera(video_path, loop=True, target_fps=30)
# ROI
extract_roi = select_extract_roi(video_path)

if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
    del os.environ["TF_CUDNN_USE_AUTOTUNE"]
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
        
# 初始化TensorFlow会话和模型
tf.compat.v1.reset_default_graph()
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

while True:
    ret, frame = vcam.read()
    if not ret:
        print("Failed to read frame")
        break
        
    # 提取ROI区域
    roi_frame = extract_roi(frame)
    
    # 转成RGB图
    frame_RGB = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    
    # 使用预先初始化的会话进行预测
    pose = predict.getpose(frame_RGB, cfg, sess, inputs, outputs)
    visim = visualize.visualize_joints(frame_RGB, pose)

    # 显示结果
    cv2.imshow('DeepLabCut Tracking', visim)
    
    # 添加延时控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 关闭TensorFlow会话
sess.close()

vcam.release()
cv2.destroyAllWindows()

print("Shutting down...")
