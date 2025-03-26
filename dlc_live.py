import cv2
import numpy as np
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.export import load_model
from deeplabcut.utils import auxiliaryfunctions
from mockcamera import virtualcamera

# 加载配置
cfg = auxiliaryfunctions.read_config("/path/to/your/config.yaml")

# 加载模型
sess, input_name, output_names, dlc_cfg = load_model(
    cfg, 
    TFGPUinference=True  # 如果没有 GPU，设置为 False
)

# 设置模拟相机
mock_cam = virtualcamera("your_video.mp4", loop=True, target_fps=30)

# 主处理循环
while True:
    ret, frame = mock_cam.read()  # 使用方式与真实相机完全相同
    if not ret:
        break
        
    # 预处理帧
    image = cv2.resize(frame, (dlc_cfg["net_width"], dlc_cfg["net_height"]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    
    # 运行推理
    outputs = sess.run(output_names, feed_dict={input_name + ":0": image[None]})
    
    # 处理输出获取关键点
    # 这部分取决于你的模型类型（单动物还是多动物）
    
    # 在帧上绘制关键点
    # ...
    
    # 显示结果
    cv2.imshow('DLC Real-Time Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
mock_cam.release()
cv2.destroyAllWindows()