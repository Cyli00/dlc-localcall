from matplotlib import pyplot as plt


def displayedparts(cfg):
    bodyparts = set(cfg['bodyparts'])  # 转为集合加速查找
    all_joints_names = cfg['all_joints_names']

    # 创建一个列表存储要保留的列索引
    pose_index = []

    # 遍历 all_joints_names，寻找匹配的关节
    for i, joint_name in enumerate(all_joints_names):
        if joint_name in bodyparts:
            # 添加该关节对应的x,y,z三列
            pose_index.append(i)
            
    return pose_index


def plot_distance(dist, ax, time_vector):
    """
    绘制距离随时间变化的图形
    :param dist: 距离数据
    :param fig: matplotlib Figure 对象
    :param time_vector: 时间向量
    """
    ax.plot(time_vector, dist, label='Distance')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (pixels)')
    ax.set_title('Distance over Time')
    ax.legend()
    plt.show()
