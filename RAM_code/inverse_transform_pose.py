import numpy as np
from scipy.spatial.transform import Rotation as R
import json

def matrix_to_pose7(T):
    """
    4x4 齐次矩阵 → 相机坐标系下的 7D 姿态 [x, y, z, qx, qy, qz, qw]
    """
    x, y, z = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return [x, y, z, *quat]

def inverse_transform_from_matrix(T_base_grasp, T_base_cam, apply_spin_correction=True):
    T_cam_base = np.linalg.inv(T_base_cam)

    if apply_spin_correction:
        R_z_minus_90 = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        T_spin_inv = np.eye(4)
        T_spin_inv[:3, :3] = np.linalg.inv(R_z_minus_90)
    else:
        T_spin_inv = np.eye(4)

    T_cam_grasp = T_cam_base @ T_base_grasp @ T_spin_inv
    print(">>> base_grasp pos =", T_base_grasp[:3, 3])
    print(">>> cam_grasp pos  =", T_cam_grasp[:3, 3])
    return matrix_to_pose7(T_cam_grasp)

def load_col_major_matrix_from_json(path):
    """
    从 JSON 文件读取列优先展开的 4×4 齐次矩阵。
    支持两种格式：
    - 正常 JSON 数组 [v0, ..., v15]
    - 错误格式 {v0,v1,...,v15}，自动解析
    """
    with open(path, "r") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                flat = data
            else:
                raise ValueError
        except:
            # fallback: 解析如 {0.1,0.2,...} 的字符串
            f.seek(0)
            text = f.read().strip().strip("{}[] \n")
            flat = [float(x) for x in text.split(",") if x.strip()]
    
    assert len(flat) == 16, f"❌ 读取失败，列优先矩阵必须有 16 个元素，当前为 {len(flat)}"
    print("读取的列优先矩阵 flat = ", flat)
    return np.array(flat).reshape((4, 4), order='F')

if __name__ == "__main__":
    # === 从 JSON 文件读取 base 坐标系下 4×4 位姿矩阵（列优先） ===
    input_path = "run_realworld/gym_outputs/object_grasp/10/empty_record.json"
    T_base_grasp = load_col_major_matrix_from_json(input_path)

    # === 固定的 base_T_cam 外参 ===
    T_base_cam = np.array([
        [-0.998944,  0.020487, -0.041116, 0.401694357],
        [ 0.042548,  0.075189, -0.996261, 0.802152163],
        [-0.017319, -0.996959, -0.075982, 0.379463994],
        [0, 0, 0, 1]
    ])

    # === 相机坐标系下的 gripper 姿态 ===
    result_7d = inverse_transform_from_matrix(T_base_grasp, T_base_cam, apply_spin_correction=True)

    print("📷 相机坐标系下的抓取姿态 7D：")
    print(json.dumps(result_7d, indent=2))

    # === 保存到 JSON 文件 ===
    output_path = "cam_grasp_pose.json"
    with open(output_path, "w") as f:
        json.dump([result_7d], f, indent=2)
    print(f"✅ 已保存到: {output_path}")