import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽info和warning,保留error

import tensorflow as tf
# 禁用tf_cudnn自动调优,防止输出不稳定
if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
    del os.environ["TF_CUDNN_USE_AUTOTUNE"]
# 获取可用的gpu列表
gpus = tf.config.experimental.list_physical_devices('GPU')

import cv2
import numpy as np
from skimage.util import img_as_ubyte

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.util import visualize

from utils.mockcamera import virtualcamera, select_extract_roi
from utils.uidisplay import displayedparts, plot_distance

# 配置路径
config_path = "/home/cyli00/gitProjects/dlc-localcall/pupil_resnet/pose_cfg.yaml"
video_path = "/home/cyli00/gitProjects/dlc-localcall/example/pupil.mp4"

# 读取配置
cfg = load_config(config_path)

# 设置模拟相机
vcam = virtualcamera(video_path, loop=True, target_fps=30)
# ROI
extract_roi = select_extract_roi(video_path)

# 设置GPU内存增长
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU configuration error:", e)

# 初始化TensorFlow会话和模型
tf.compat.v1.reset_default_graph()
sess, inputs, outputs = predict.setup_GPUpose_prediction(cfg, gpus)
pose_tensor = predict.extract_GPUprediction(outputs, cfg)

# 选择displayedparts
pose_index = displayedparts(cfg)

while True:
    ret, frame = vcam.read()
    if not ret:
        print("Failed to read frame")
        break

    # 提取ROI区域
    frame = extract_roi(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = img_as_ubyte(frame)

    pose = sess.run(
        pose_tensor,
        feed_dict={inputs: np.expand_dims(frame, axis=0).astype(float)}
    )
    pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]

    # 只保留需要的行
    pose = pose[pose_index]

    # ==> pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
    visim = visualize.visualize_joints(frame, pose)

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

