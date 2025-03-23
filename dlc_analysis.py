import os
from pathlib import Path
import cv2
import deeplabcut
from tkinter import filedialog
from dlclibrary.dlcmodelzoo.modelzoo_download import MODELOPTIONS
import sys
from deeplabcut.utils import auxiliaryfunctions
import shutil

def select_model():
    """Let user select a model from available options."""
    print("\nAvailable models:")
    for i, model in enumerate(MODELOPTIONS, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = int(input("\nPlease select a model (enter the number): "))
            if 1 <= choice <= len(MODELOPTIONS):
                return MODELOPTIONS[choice-1]
            else:
                print(f"Please enter a number between 1 and {len(MODELOPTIONS)}")
        except ValueError:
            print("Please enter a valid number")

def extract_roi_from_video(video_path):
    """Extract ROI from video and create a new video with only the ROI."""
    videotype = os.path.splitext(video_path)[-1].lstrip('.')
    
    # Open video and get first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Could not read video file")
        
    # Select ROI
    print("Select a ROI and then press SPACE or ENTER button!")
    print("Cancel the selection process by pressing c button!")
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=False)
    cv2.destroyAllWindows()
    
    # Get ROI coordinates 
    x, y, w, h = roi
    
    # Create new video with ROI
    roi_video_path = os.path.splitext(video_path)[0] + '_roi' + os.path.splitext(video_path)[1]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(roi_video_path, fourcc, fps, (int(w), int(h)))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Crop frame to ROI
            roi_frame = frame[int(y):int(y+h), int(x):int(x+w)]
            out.write(roi_frame)
        else:
            break
            
    cap.release()
    out.release()
    
    return roi_video_path, videotype

def setup_dlc_project(video_path, model_name='mouse_pupil_vclose', project_name='myDLC_modelZoo', experimenter_name='teamDLC'):
    """Setup DLC project with pretrained model, avoiding redundant downloads."""
    
    # First extract ROI from video
    roi_video_path, videotype = extract_roi_from_video(video_path)
    
    # Create project with model
    if model_name in MODELOPTIONS:
        cwd = os.getcwd()
        
        # 创建新项目
        cfg = deeplabcut.create_new_project(
            project_name,
            experimenter_name,
            [roi_video_path],
            working_directory=None,
            copy_videos=False,
            videotype=videotype
        )
        
        config = auxiliaryfunctions.read_config(cfg)
        
        # 设置模型目录
        model_folder = auxiliaryfunctions.get_model_folder(
            trainFraction=config["TrainingFraction"][0],
            shuffle=1,
            cfg=config,
        )
        
        train_dir = Path(os.path.join(config["project_path"], str(model_folder), "train"))
        test_dir = Path(os.path.join(config["project_path"], str(model_folder), "test"))
        
        # 创建目录
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 直接使用本地模型文件
        weight_folder = os.path.join("/home/cyli00/gitRepo/dlc_test_code/models/pretrained", model_name)
        if not os.path.exists(weight_folder):
            raise Exception(f"Local model not found in {weight_folder}")
            
        print("Using local model from:", weight_folder)
        
        # 复制模型文件到训练目录
        for root, dirs, files in os.walk(weight_folder):
            for file in files:
                src_path = os.path.join(root, file)
                # 获取相对路径
                rel_path = os.path.relpath(src_path, weight_folder)
                dst_path = os.path.join(train_dir, rel_path)
                # 确保目标目录存在
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
        
        # 读取并更新配置
        pose_cfg_file = os.path.join(train_dir, "pose_cfg.yaml")
        pose_cfg = auxiliaryfunctions.read_plainconfig(pose_cfg_file)
        
        # 更新项目配置
        dict_ = {
            "default_net_type": pose_cfg["net_type"],
            "default_augmenter": "imgaug",
            "bodyparts": pose_cfg["all_joints_names"],
            "dotsize": 6,
        }
        auxiliaryfunctions.edit_config(cfg, dict_)
        
        # 更新训练配置
        train_cfg_path = str(train_dir / "pose_cfg.yaml")
        test_cfg_path = str(test_dir / "pose_cfg.yaml")
        
        # 获取模型文件名
        model_files = [f for f in os.listdir(train_dir) if f.endswith('.meta')]
        if not model_files:
            raise Exception("No model file found in the train directory")
        model_name = model_files[0].replace('.meta', '')
        
        # 更新训练配置
        dict2change = {
            "init_weights": str(train_dir / model_name),
            "project_path": str(config["project_path"]),
        }
        pose_cfg.update(dict2change)
        auxiliaryfunctions.write_plainconfig(train_cfg_path, pose_cfg)
        
        # 创建测试配置
        test_cfg = {
            "dataset": pose_cfg["dataset"],
            "dataset_type": pose_cfg["dataset_type"],
            "num_joints": pose_cfg["num_joints"],
            "all_joints": pose_cfg["all_joints"],
            "all_joints_names": pose_cfg["all_joints_names"],
            "net_type": pose_cfg["net_type"],
            "init_weights": str(train_dir / model_name),
            "global_scale": 1.0,
            "location_refinement": pose_cfg["location_refinement"],
            "locref_stdev": pose_cfg["locref_stdev"]
        }
        auxiliaryfunctions.write_plainconfig(test_cfg_path, test_cfg)
        
        # 分析视频
        video_dir = os.path.join(config["project_path"], "videos")
        print("Analyzing video...")
        deeplabcut.analyze_videos(cfg, [video_dir], videotype, save_as_csv=True)
        
        # 创建标记视频
        print("Creating labeled video...")
        deeplabcut.filterpredictions(cfg, [video_dir], videotype)
        deeplabcut.create_labeled_video(
            cfg, [video_dir], videotype, draw_skeleton=True, filtered=True,displayedbodyparts=["Lpupil","Dpupil","Rpupil","Vpupil"]
        )
        deeplabcut.plot_trajectories(cfg, [video_dir], videotype, filtered=True,displayedbodyparts=["Lpupil","Dpupil","Rpupil","Vpupil"])
        
        os.chdir(cwd)
        return cfg, pose_cfg_file
    else:
        return "N/A", "N/A"

if __name__ == "__main__":
    # pop out a dialog to help user select the video path
    video_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if not video_path:
        print("No video selected. Exiting...")
        sys.exit()
    # Let user select model
    model_name = select_model()
    print(f"\nSelected model: {model_name}")
    
    # Setup project with selected model
    config_path, train_config_path = setup_dlc_project(video_path, model_name=model_name)
    print("Project created with config at:", config_path) 