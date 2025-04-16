import cv2
import time

class virtualcamera:
    """模拟相机类，将视频文件作为实时输入源"""
    
    def __init__(self, video_path, loop=True, target_fps=None):
        """
        初始化模拟相机
        
        参数:
            video_path: 视频文件路径
            loop: 是否循环播放视频
            target_fps: 目标帧率 (None 表示使用视频原始帧率)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.loop = loop
        
        # 获取视频原始帧率
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = target_fps if target_fps else self.original_fps
        
        # 计算每帧间隔时间
        self.frame_time = 1.0 / self.target_fps
        self.last_frame_time = 0
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        print(f"Video loaded - FPS: {self.original_fps}, target FPS: {self.target_fps}")
    
    def read(self):
        """
        读取下一帧，模拟相机的实时采集
        
        返回:
            ret: 是否成功读取帧
            frame: 帧图像
        """
        # 模拟实时帧率
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        
        # 如果距离上一帧时间不足，等待
        if time_diff < self.frame_time:
            time.sleep(self.frame_time - time_diff)
        
        # 读取帧
        ret, frame = self.cap.read()
        
        # 如果视频结束且需要循环播放
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        self.last_frame_time = time.time()
        return ret, frame
    
    def release(self):
        """释放视频资源"""
        self.cap.release()
        
    def get(self, propId):
        """获取视频属性，与 cv2.VideoCapture 接口一致"""
        return self.cap.get(propId)
        
    def set(self, propId, value):
        """设置视频属性，与 cv2.VideoCapture 接口一致"""
        return self.cap.set(propId, value)
        
    def isOpened(self):
        """检查视频是否打开，与 cv2.VideoCapture 接口一致"""
        return self.cap.isOpened()


# 用法示例
def demo():
    # 初始化模拟相机
    video_path = "your_video.mp4"  # 替换为你的视频路径
    mock_cam = virtualcamera(video_path, loop=True, target_fps=30)
    
    try:
        while True:
            # 使用与真实相机相同的方式读取帧
            ret, frame = mock_cam.read()
            if not ret:
                print("无法读取视频帧")
                break
                
            # 显示帧
            cv2.imshow('Mock Camera', frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        mock_cam.release()
        cv2.destroyAllWindows()
        
def select_extract_roi(video_path):
    """
    从相机获取一帧用于ROI选择，然后返回一个函数用于后续提取ROI区域
    
    参数:
        camera: OpenCV相机对象（cv2.VideoCapture或virtualcamera）
        
    返回:
        extract_roi: 一个函数，可以从任何帧中提取ROI区域
    """
    # Open video and get first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Could not read video file")
        
    # Select ROI
    # print("Select a ROI and then press SPACE or ENTER button!")
    # print("Cancel the selection process by pressing c button!")
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=False)
    cv2.destroyAllWindows()
    
    # Get ROI coordinates 
    x, y, w, h = roi
    
    # 检查是否选择了有效的ROI
    if w <= 0 or h <= 0:
        print("No ROI was selected, would return original frame.")
        # 返回一个函数，该函数不修改输入帧
        return lambda frame: frame
    
    print(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
    
    # 返回一个函数，用于从任何帧中提取ROI
    def extract_roi(frame):
        """从frame中提取ROI区域"""
        return frame[int(y):int(y+h), int(x):int(x+w)]
    
    return extract_roi


if __name__ == "__main__":
    demo()