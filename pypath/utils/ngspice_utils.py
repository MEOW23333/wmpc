import numpy as np
import scipy.sparse as sp
from sympy.logic.boolalg import bool_minterm
import torch
import os
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict
import re
import sys
import time
import subprocess
from multiprocessing import Pool, cpu_count
import shutil
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import proposer.config

PARALLEL_EXECUTION = True

NGSPICE_EXECUTABLE = proposer.config.NGSPICE_EXECUTABLE

# ==========================================
#        FOR TRAINING DATA GENERATION
# ==========================================


def extract_uut_only_features(
    node_map: Dict[str, int], 
    uut_instance_name: str,
    pin_to_node_map: Dict[str, str]
):

    uut_node_names = set()
    uut_name_lower = uut_instance_name.lower()

    # 1a. 添加所有外部引脚节点
    for pin, node_name in pin_to_node_map.items():
        # instance_info['pins'] 包含了 VDD/VSS，而 pin_to_node_map 不包含
        # 所以我们不需要额外的过滤
        uut_node_names.add(node_name.lower())

    # 1b. 【核心修正】添加所有内部节点，包括深层节点
    # 我们寻找的模式是 'x_uut.' 或 '.x_uut.' 这样的子字符串
    # 这样可以匹配 'x_uut.internal' 和 'm.x_uut.xx5.m1#drain'
    search_pattern = f".{uut_name_lower}."
    start_pattern = f"{uut_name_lower}."
    
    for node_name in node_map.keys():
        node_name_lower = node_name.lower()
        # 检查是否是内部节点
        # 条件：以 'x_uut.' 开头，或者包含 '.x_uut.' (处理 m.x_uut. 等情况)
        if node_name_lower.startswith(start_pattern) or search_pattern in node_name_lower:
            uut_node_names.add(node_name_lower)

    if not uut_node_names:
        return None

    # 2. 对齐 (Alignment) - 这部分逻辑保持不变，因为它不关心节点名的内容
    uut_nodes_with_indices = []
    for node_name_lower in uut_node_names:
        matrix_index = node_map.get(node_name_lower)
        if matrix_index is not None:
            uut_nodes_with_indices.append((node_name_lower, matrix_index))
            
    sorted_uut_nodes = sorted(uut_nodes_with_indices, key=lambda item: item[1])
    
    uut_indices = [item[1] - 1 for item in sorted_uut_nodes]
    
    if not uut_indices:
        return None
    
    return uut_indices


def read_J_sparse(filepath: str, *, matrix_format: str = "csr") -> sp.spmatrix:
    """Read an ngspice trajectory Jacobian as a SciPy sparse matrix.

    The legacy read_J materializes a dense ndarray. That is suitable for small
    probes but unusable for large PG matrices. This reader keeps ngspice
    triplets sparse and merges duplicate stamps.
    """
    lines = []
    file_found = False
    for attempt in range(5):
        try:
            with open(filepath, "r") as handle:
                lines = handle.readlines()
            file_found = True
            if lines:
                parts = lines[-1].strip().split()
                if len(parts) >= 2 and int(parts[0]) == 0 and int(parts[1]) == 0:
                    break
            print(f"[INFO] File '{filepath}' seems incomplete on attempt {attempt + 1}. Waiting...")
            time.sleep(0.1)
        except FileNotFoundError:
            print(f"[INFO] File '{filepath}' not found on attempt {attempt + 1}. Waiting for it to be created...")
            time.sleep(0.1)
        except (ValueError, IndexError):
            print(f"[INFO] Last line of '{filepath}' is malformed on attempt {attempt + 1}. Waiting...")
            time.sleep(0.1)

    if not file_found or not lines:
        print(f"[WARNING] Sparse Jacobian file not found or empty: {filepath}")
        return sp.csr_matrix((0, 0), dtype=np.float64)
    if len(lines) < 2 or "Circuit Matrix" not in lines[0]:
        print(f"[WARNING] Invalid sparse Jacobian header in: {filepath}")
        return sp.csr_matrix((0, 0), dtype=np.float64)

    try:
        parts = lines[1].strip().split()
        dim = int(parts[0])
        is_complex = "complex" in parts[1].lower() if len(parts) > 1 else False
    except (ValueError, IndexError):
        print(f"[WARNING] Could not parse sparse Jacobian dimension/type from: {filepath}")
        return sp.csr_matrix((0, 0), dtype=np.float64)

    rows = []
    cols = []
    values = []
    for line in lines[2:]:
        parts = line.split()
        try:
            if len(parts) >= 2 and int(parts[0]) == 0 and int(parts[1]) == 0:
                break
            if is_complex and len(parts) >= 4:
                rows.append(int(parts[0]) - 1)
                cols.append(int(parts[1]) - 1)
                values.append(float(parts[2]) + 1j * float(parts[3]))
            elif (not is_complex) and len(parts) >= 3:
                rows.append(int(parts[0]) - 1)
                cols.append(int(parts[1]) - 1)
                values.append(float(parts[2]))
        except (ValueError, IndexError):
            print(f"[WARNING] Skipping malformed sparse Jacobian line in {filepath}: {line.strip()}")

    dtype = np.complex128 if is_complex else np.float64
    matrix = sp.coo_matrix(
        (
            np.asarray(values, dtype=dtype),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(dim, dim),
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix_format == "coo":
        return matrix
    if matrix_format == "csc":
        return matrix.tocsc()
    return matrix.tocsr()

class NgspiceInterface:
    """
    运行一次完整的瞬态分析。
    """
    def run_simulation_trajectory(
        self,
        netlist_path: str,
        traj_dir: str,
        failed_netlist_path: str,
        circuit_id_str: str,
        timeout_s: Optional[float] = None,
    ) -> bool:

        proc_env = os.environ.copy()
        proc_env["TRAJ"] = "1"
        proc_env["VALUE"] = "0"
        proc_env["CKT_ID"] = circuit_id_str
        proc_env["TRAJ_DIR"] = traj_dir

        command = [NGSPICE_EXECUTABLE, "-b", netlist_path]
        try:
            run_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "env": proc_env,
            }
            if timeout_s is not None and float(timeout_s) > 0:
                run_kwargs["timeout"] = float(timeout_s)
            subprocess.run(command, **run_kwargs)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  [NGSPICE FAIL] Simulation failed for {circuit_id_str}.")
            print(f"  Stderr: {e.stderr}")
            
            shutil.copyfile(netlist_path, failed_netlist_path)
            print(f"  Saved failing netlist to: {failed_netlist_path}")

            # 检查轨迹目录是否存在
            if os.path.isdir(traj_dir):
                files_removed_count = 0
                # 构造文件名前缀
                prefix_to_find = f"circuit_{circuit_id_str}"
                
                # 遍历目录中的所有文件
                for filename in os.listdir(traj_dir):
                    # 如果文件名以我们期望的前缀开头
                    if filename.startswith(prefix_to_find):
                        file_to_remove = os.path.join(traj_dir, filename)
                        try:
                            os.remove(file_to_remove)
                            files_removed_count += 1
                        except OSError as remove_error:
                            print(f"    [WARN] Could not remove trajectory file {file_to_remove}: {remove_error}")
                
                if files_removed_count > 0:
                    print(f"    -> Removed {files_removed_count} trajectory files.")
            else:
                print(f"    Trajectory directory '{traj_dir}' not found, nothing to clean.")
            return False


def _process_circuit_trajectories_worker(task):
    ckt_id_str, file_info_list, tra_dir, pin_to_node_map = task
    local_processor = DataProcessor()

    sorted_file_info_list = sorted(file_info_list, key=lambda x: (x[0], -x[1], x[2]))
    trajectories_by_time = defaultdict(list)
    for time_val, gmin_val, iter_id, filename in sorted_file_info_list:
        trajectories_by_time[time_val].append((gmin_val, iter_id, filename))

    for time_val, iter_files in trajectories_by_time.items():
        MIN_TRAJECTORY_LENGTH = 5
        if not iter_files:
            continue
        max_iteration = max(info[1] for info in iter_files)
        if max_iteration < MIN_TRAJECTORY_LENGTH:
            continue

        consistent_files = []
        standard_dim = -1
        for gmin_val, iter_id, filename in iter_files:
            filepath = os.path.join(tra_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                section = None
                count = 0
                for line in lines:
                    if "OLD" in line.strip():
                        section = "old"
                        count = 0
                        continue
                    if section == "old" and line.strip():
                        count += 1
                if standard_dim == -1:
                    standard_dim = count
                if count == standard_dim:
                    consistent_files.append((time_val, gmin_val, iter_id, filename))
            except Exception:
                continue

        for time_val, gmin_val, iter_id, filename in consistent_files:
            filepath = os.path.join(tra_dir, filename)
            local_processor.pair_single_iteration(
                ckt_id_str,
                time_val,
                iter_id,
                gmin_val,
                filepath,
                pin_to_node_map,
            )

    return ckt_id_str, local_processor.training_samples

class DataProcessor:
    """
    独立的后处理器，用于解析轨迹并生成样本。
    """
    def __init__(self):
        self.training_samples = []
        self.dimension_map = {}

    def process_and_pair_all_trajectories(self, tra_dir, pin_to_node_map=None, num_workers=None):
        if pin_to_node_map is None:
            pin_to_node_map = {}
        
        # 扫描轨迹文件
        grouped_files = defaultdict(list)
        traj_pattern = re.compile(r"circuit_(.+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)\.txt")
        for filename in os.listdir(tra_dir):
            match = traj_pattern.match(filename)
            if match:
                ckt_id_str, time_val_str, gmin_val_str, iter_id_str = match.groups()
                grouped_files[ckt_id_str].append((float(time_val_str), float(gmin_val_str), int(iter_id_str), filename))
        
        # # 扫描雅可比矩阵文件
        # jac_files_dict = {}  # 用于存储雅可比矩阵文件路径，键为 (ckt_id, time_val, gmin_val, iter_id)
        # jac_pattern = re.compile(r"circuit_(.+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)_jac\.txt")
        # for filename in os.listdir(tra_dir):
        #     match = jac_pattern.match(filename)
        #     if match:
        #         ckt_id_str, time_val_str, gmin_val_str, iter_id_str = match.groups()
        #         key = (ckt_id_str, float(time_val_str), float(gmin_val_str), int(iter_id_str))
        #         jac_files_dict[key] = os.path.join(tra_dir, filename)
        
        if not grouped_files:
            print("[ERROR] No valid trajectory files found.")
            return

        circuit_tasks = list(grouped_files.items())
        if num_workers is None:
            num_workers = 1

        if num_workers > 1 and len(circuit_tasks) > 1:
            worker_count = max(1, min(int(num_workers), len(circuit_tasks)))
            print(f"[training-json] using {worker_count} parallel workers for {len(circuit_tasks)} circuits")
            tasks = [
                (ckt_id_str, file_info_list, tra_dir, pin_to_node_map)
                for ckt_id_str, file_info_list in circuit_tasks
            ]
            completed = 0
            with Pool(processes=worker_count) as pool:
                for ckt_id_str, circuit_samples in pool.imap_unordered(_process_circuit_trajectories_worker, tasks):
                    completed += 1
                    self.training_samples.extend(circuit_samples)
                    print(
                        f"[training-json] {completed}/{len(circuit_tasks)} "
                        f"circuit={ckt_id_str} samples={len(circuit_samples)} "
                        f"total={len(self.training_samples)}"
                    )
            print(f"In total 0 ckt are bad.")
            return

        bad_ckt_count = 0
        for ckt_id_str, file_info_list in circuit_tasks:

            # 按照要求的顺序排序：ckt_id_str, time_val_str, gmin_val_str(降序), iter_id_str
            # 注意：ckt_id_str 已经通过 grouped_files 分组，所以这里只需要对 file_info_list 排序
            # 排序键：(time_val, -gmin_val, iter_id) - 注意 gmin_val 用负号实现降序
            sorted_file_info_list = sorted(file_info_list, key=lambda x: (x[0], -x[1], x[2]))
            
            trajectories_by_time = defaultdict(list)
            for time_val, gmin_val, iter_id, filename in sorted_file_info_list:
                trajectories_by_time[time_val].append((gmin_val, iter_id, filename))
            
            for time_val, iter_files in trajectories_by_time.items():
                
                # --- 修改点: 轨迹级别的长度检查 ---
                MIN_TRAJECTORY_LENGTH = 5
                if not iter_files:
                    continue
                max_iteration = max(info[1] for info in iter_files)  # iter_id 是第2个元素（索引1）
                if max_iteration < MIN_TRAJECTORY_LENGTH:
                    print(f"  [DATA CLEANING] Trajectory for ckt '{ckt_id_str}' at time {time_val} is too short (length={max_iteration}). Discarding.")
                    continue # 只跳过这个时间点的轨迹，不影响其他时间点

                # 文件已经按 gmin_val(降序), iter_id 排序，这里保持顺序
                sorted_iter_files = iter_files

                # --- 严格的维度一致性校验 ---
                standard_dim = -1
                consistent_files = []
                for gmin_val, iter_id, filename in sorted_iter_files:
                    filepath = os.path.join(tra_dir, filename)
                    try:
                        with open(filepath, 'r') as f: lines = f.readlines()
                        section = None
                        count = 0
                        for line in lines:
                            if "OLD" in line.strip(): section = "old"; count = 0; continue
                            if section == "old" and line.strip(): count += 1
                        
                        if standard_dim == -1: standard_dim = count
                        
                        if count == standard_dim:
                            consistent_files.append((time_val, gmin_val, iter_id, filename))
                        else:
                            print(f"    [DIMENSION MISMATCH] Skipping {filename} (dim={count}, expected={standard_dim})")
                    except Exception as e:
                        print(f"    [FILE READ ERROR] Skipping {filename}: {e}")
            
                # --- 处理维度一致的文件 ---
                for time_val, gmin_val, iter_id, filename in consistent_files:
                    filepath = os.path.join(tra_dir, filename)
                    # # 查找对应的雅可比矩阵文件
                    # jac_key = (ckt_id_str, time_val, gmin_val, iter_id)
                    # jac_filepath = jac_files_dict.get(jac_key, None)
                    self.pair_single_iteration(ckt_id_str, time_val, iter_id, gmin_val, filepath, pin_to_node_map)
                    # self.pair_single_iteration(ckt_id_str, time_val, iter_id, gmin_val, filepath, pin_to_node_map, jac_filepath)
        print(f"In total {bad_ckt_count} ckt are bad.")
        
    def pair_single_iteration(self, ckt_id, time_val, iter_num, gmin_val, traj_filepath, pin_to_node_map, jac_filepath=None):
        with open(traj_filepath, 'r') as f: lines = f.readlines()
        residuals, rhsold, rhsnew, node_to_index =[], [], [], {}
        section = None
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue
            if "OLD" in line_stripped: section = "old"; continue
            if "NEW" in line_stripped: section = "new"; continue
            if "NODE_MAP" in line_stripped: section = "map"; continue
            if "RES" in line_stripped: section = "res"; continue
            try:
                if section == "old": rhsold.append(float(line_stripped))
                elif section == "new": rhsnew.append(float(line_stripped))
                elif section == "res": residuals.append(float(line_stripped))
                elif section == "map":
                    parts = line_stripped.split()
                    if len(parts) == 2: node_to_index[parts[0].lower()] = int(parts[1])
            except (ValueError, IndexError): continue
            
        # 提取 UUT 索引（如果需要）
        uut_indices = None
        if proposer.config.LOCAL_ONLY:
            rhsold, rhsnew, residuals = np.array(rhsold), np.array(rhsnew), np.array(residuals)
        else:
            uut_indices = extract_uut_only_features(node_to_index, "X_UUT", pin_to_node_map)
            rhsold, rhsnew, residuals = np.array(rhsold)[uut_indices], np.array(rhsnew)[uut_indices], np.array(residuals)[uut_indices]
        
        # 读取雅可比矩阵（如果提供了文件路径）
        jacobian_matrix = None
        if jac_filepath and os.path.exists(jac_filepath):
            jacobian_matrix = read_J(jac_filepath)
            # 如果矩阵为空，设置为 None
            if jacobian_matrix.size == 0:
                jacobian_matrix = None
                print(f"    [WARNING] Jacobian matrix is empty for {jac_filepath}")
            else:
                if jacobian_matrix.shape[0] != len(rhsold[-1]):
                    print(f"jacobian_matrix.shape[0]: {jacobian_matrix.shape[0]}, len(rhsold): {len(rhsold[-1])}")
                    jacobian_matrix = None
                    print(f"    [WARNING] Jacobian matrix dimension mismatch for {jac_filepath} (expected {len(rhsold[-1])}, got {jacobian_matrix.shape[0]})")
                

        
        # 转换为列表以便 JSON 序列化
        sample_dict = {
            "circuit_id": ckt_id, "time": time_val, 
            "iteration": iter_num, "gmin_val": gmin_val,
            "node_map": node_to_index,
            "rhsold": rhsold.tolist(), "rhsnew": rhsnew.tolist(), "pure_residuals": residuals.tolist()
        }
        
        # 如果成功读取了雅可比矩阵，添加到样本中
        if jacobian_matrix is not None:
            # 将复数矩阵转换为列表格式（如果是复数，转换为 [real, imag] 的列表）
            if np.iscomplexobj(jacobian_matrix):
                sample_dict["jacobian"] = {
                    "real": np.real(jacobian_matrix).tolist(),
                    "imag": np.imag(jacobian_matrix).tolist()
                }
            else:
                sample_dict["jacobian"] = jacobian_matrix.tolist()
        
        self.training_samples.append(sample_dict)


    def save_samples_to_file(self, output_path):
        if not self.training_samples:
            print("No training samples were generated.")
            return

        with open(output_path, 'w') as f:
            json.dump(self.training_samples, f)
        
        print(f"  Successfully generated {len(self.training_samples)} samples.")

# ==========================================
#              FOR EVALUATION
# ==========================================
def read_F(filepath: str) -> list:
    with open(filepath, 'r') as f: lines = f.readlines()
    res = []
    section = None
    dim = 0
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped: continue
        if "RES" in line_stripped: section = "res"; continue
        try:
            if section == "res": 
                res.append(float(line_stripped))
                dim += 1
        except (ValueError, IndexError): continue
    if dim != len(res):
        print("[ERROR!dim != len(res)]")
    return res

def read_J(filepath: str) -> np.ndarray:
    lines = []
    file_found = False
    
    # --- 步骤 1: 轮询循环，等待文件出现并看起来完整 ---
    for attempt in range(5):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            file_found = True
            
            # 检查文件是否至少有内容，并且最后一行是否是终止符
            if lines:
                last_line = lines[-1].strip()
                parts = last_line.split()
                if len(parts) >= 2 and int(parts[0]) == 0 and int(parts[1]) == 0:
                    # 文件看起来是完整的，可以退出等待循环
                    break
            
            # 如果文件不完整，打印信息并等待
            print(f"[INFO] File '{filepath}' seems incomplete on attempt {attempt+1}. Waiting...")
            time.sleep(0.1)

        except FileNotFoundError:
            # 如果文件还不存在，等待它被创建
            print(f"[INFO] File '{filepath}' not found on attempt {attempt+1}. Waiting for it to be created...")
            time.sleep(0.1)
            continue # 继续下一次循环
        
        except (ValueError, IndexError):
             # 如果最后一行格式错误导致int()转换失败，也等待
             print(f"[INFO] Last line of '{filepath}' is malformed on attempt {attempt+1}. Waiting...")
             time.sleep(0.1)


    if not file_found:
        print(f"[WARNING] Jacobian file not found after {5} retries: {filepath}")
        return np.zeros((0, 0))
    
    if not lines:
        print(f"[WARNING] Jacobian file is empty: {filepath}")
        return np.zeros((0, 0))

    if len(lines) < 2:
        print(f"[WARNING] Jacobian file is too short to be valid: {filepath}")
        return np.zeros((0, 0))

    # 1. 验证文件头
    if "Circuit Matrix" not in lines[0]:
        print(f"[WARNING] Invalid header in Jacobian file: {filepath}")
        return np.zeros((0, 0))

    # 2. 从第二行解析维度和数据类型
    try:
        parts = lines[1].strip().split()
        dim = int(parts[0])
        is_complex = "complex" in parts[1].lower() if len(parts) > 1 else False
    except (ValueError, IndexError):
        print(f"[WARNING] Could not parse dimension/type from line 2 in: {filepath}")
        return np.zeros((0, 0))

    # 3. 根据解析出的信息初始化矩阵
    dtype = np.complex128 if is_complex else np.float64
    dense_matrix = np.zeros((dim, dim), dtype=dtype)

    # 4. 从第三行开始解析矩阵的非零元素
    for i, line in enumerate(lines[2:], start=3):
        parts = line.split()
        
        try:
            if len(parts) >= 2 and int(parts[0]) == 0 and int(parts[1]) == 0:
                # 遇到了终止符，解析结束
                break

            # 根据是否为复数来决定读取多少列
            if is_complex and len(parts) >= 4:
                # 行和列的索引是从1开始的，所以需要减1
                row, col = int(parts[0]) - 1, int(parts[1]) - 1
                real_part = float(parts[2])
                imag_part = float(parts[3])
                
                # 检查索引是否越界
                if 0 <= row < dim and 0 <= col < dim:
                    dense_matrix[row, col] = real_part + 1j * imag_part
                else:
                    print(f"[WARNING] Index ({row+1}, {col+1}) out of bounds for dim={dim} in {filepath}")

            elif not is_complex and len(parts) >= 3:
                row, col = int(parts[0]) - 1, int(parts[1]) - 1
                val = float(parts[2])
                if 0 <= row < dim and 0 <= col < dim:
                    dense_matrix[row, col] = val
                else:
                    print(f"[WARNING] Index ({row+1}, {col+1}) out of bounds for dim={dim} in {filepath}")

        except (ValueError, IndexError):
            # 忽略任何无法解析的行
            print(f"[WARNING] Could not parse line {i} in {filepath}: '{line.strip()}'")
            continue
            
    return dense_matrix

def read_wp_value(filepath: str) -> list:
    with open(filepath, 'r') as f: lines = f.readlines()
    wp = []
    section = None
    dim = 0
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped: continue
        if "WP_OUT" in line_stripped: section = "wp_out"; continue
        try:
            if section == "wp_out": 
                wp.append(float(line_stripped))
                dim += 1
        except (ValueError, IndexError): continue
    if dim != len(wp):
        print("[ERROR!dim != len(res)]")
    return wp


def read_continuation_step(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        lines = f.readlines()

    payload = {
        "rhsold": [],
        "state0_in": [],
        "rhsnew": [],
        "state0_out": [],
        "residual": [],
        "wp_out": [],
        "node_map": {},
    }
    section = None
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if "OLD" in line_stripped:
            section = "old"
            continue
        if "STATE0_IN" in line_stripped:
            section = "state0_in"
            continue
        if "NEW" in line_stripped:
            section = "new"
            continue
        if "STATE0_OUT" in line_stripped:
            section = "state0_out"
            continue
        if "NODE_MAP" in line_stripped:
            section = "map"
            continue
        if "RES" in line_stripped:
            section = "res"
            continue
        if "WP_OUT" in line_stripped:
            section = "wp_out"
            continue
        try:
            if section == "old":
                payload["rhsold"].append(float(line_stripped))
            elif section == "state0_in":
                payload["state0_in"].append(float(line_stripped))
            elif section == "new":
                payload["rhsnew"].append(float(line_stripped))
            elif section == "state0_out":
                payload["state0_out"].append(float(line_stripped))
            elif section == "res":
                payload["residual"].append(float(line_stripped))
            elif section == "wp_out":
                payload["wp_out"].append(float(line_stripped))
            elif section == "map":
                parts = line_stripped.split()
                if len(parts) == 2:
                    payload["node_map"][parts[0].lower()] = int(parts[1])
        except (ValueError, IndexError):
            continue

    return payload


def read_linear_system_corpus_step(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        lines = f.readlines()

    payload = {
        "meta": {},
        "rhsold": [],
        "rhs": [],
        "state0": [],
        "raw_residual": [],
        "raw_residual_norm": None,
        "node_map": {},
    }
    section = None
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if "META" in line_stripped:
            section = "meta"
            continue
        if "RHSOLD" in line_stripped:
            section = "rhsold"
            continue
        if "RHS" in line_stripped:
            section = "rhs"
            continue
        if "STATE0" in line_stripped:
            section = "state0"
            continue
        if "RAW_RESIDUAL_NORM" in line_stripped:
            section = "raw_residual_norm"
            continue
        if "RAW_RESIDUAL" in line_stripped:
            section = "raw_residual"
            continue
        if "NODE_MAP" in line_stripped:
            section = "map"
            continue
        try:
            if section == "meta":
                parts = line_stripped.split(None, 1)
                if len(parts) != 2:
                    continue
                key, value = parts
                if key in {"time", "gmin"}:
                    payload["meta"][key] = float(value)
                elif key in {"iteration", "matrix_size"}:
                    payload["meta"][key] = int(value)
                else:
                    payload["meta"][key] = value
            elif section == "rhsold":
                payload["rhsold"].append(float(line_stripped))
            elif section == "rhs":
                payload["rhs"].append(float(line_stripped))
            elif section == "state0":
                payload["state0"].append(float(line_stripped))
            elif section == "raw_residual":
                payload["raw_residual"].append(float(line_stripped))
            elif section == "raw_residual_norm":
                payload["raw_residual_norm"] = float(line_stripped)
            elif section == "map":
                parts = line_stripped.split()
                if len(parts) == 2:
                    payload["node_map"][parts[0].lower()] = int(parts[1])
        except (ValueError, IndexError):
            continue

    return payload


def run_ngspice_linear_system_corpus(
    val_dir: str,
    netlist_dir: str,
    real_ckt_id: int,
    case: str,
    *,
    timeout: int = 10000,
) -> Dict[str, Any]:
    task_dir = os.path.join(
        val_dir,
        f"linear_system_task_{os.getpid()}_{int(time.time() * 1000000) % 1000000}_{case}",
    )
    os.makedirs(task_dir, exist_ok=True)

    netlist_path = os.path.join(netlist_dir, f"{real_ckt_id}.sp")
    proc_env = os.environ.copy()
    proc_env["TRAJ"] = "0"
    proc_env["VALUE"] = "0"
    proc_env["CKT_ID"] = str(real_ckt_id)
    proc_env["LINEAR_SYSTEM_CORPUS_MODE"] = "1"
    proc_env["LINEAR_SYSTEM_CORPUS_DIR"] = task_dir

    if not os.path.exists(netlist_path):
        return {
            "success": False,
            "reason": f"netlist_not_found:{netlist_path}",
            "steps": [],
            "task_dir": task_dir,
        }

    command = [NGSPICE_EXECUTABLE, "-b", netlist_path]
    completed = None
    process_error_reason = None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            env=proc_env,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        completed = e
        stderr_obj = e.stderr if hasattr(e, "stderr") else None
        if isinstance(stderr_obj, bytes):
            stderr_text = stderr_obj.decode("utf-8", errors="replace")
        else:
            stderr_text = stderr_obj or ""
        stderr = stderr_text[:400] if stderr_text else "N/A"
        process_error_reason = f"ngspice_linear_system_failed:{stderr}"

    step_pattern = re.compile(
        r"linear_system_circuit_(.+?)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)\.txt$"
    )
    steps = []
    for filename in os.listdir(task_dir):
        match = step_pattern.match(filename)
        if not match:
            continue
        _ckt_id_str, time_val_str, gmin_val_str, iter_id_str = match.groups()
        filepath = os.path.join(task_dir, filename)
        payload = read_linear_system_corpus_step(filepath)
        jac_filepath = filepath[:-4] + "_jac.txt"
        jacobian = read_J(jac_filepath) if os.path.exists(jac_filepath) else np.zeros((0, 0))
        raw_residual = np.asarray(payload["raw_residual"], dtype=np.float64)
        raw_residual_norm = payload["raw_residual_norm"]
        if raw_residual_norm is None and raw_residual.size:
            raw_residual_norm = float(np.linalg.norm(raw_residual))
        steps.append(
            {
                "iteration": int(iter_id_str),
                "time": float(time_val_str),
                "gmin_val": float(gmin_val_str),
                "rhsold": np.asarray(payload["rhsold"], dtype=np.float64),
                "rhs": np.asarray(payload["rhs"], dtype=np.float64),
                "state0": np.asarray(payload["state0"], dtype=np.float64),
                "raw_residual": raw_residual,
                "raw_residual_norm": raw_residual_norm,
                "node_map": payload["node_map"],
                "meta": payload["meta"],
                "jacobian": jacobian,
                "filepath": filepath,
                "jacobian_filepath": jac_filepath,
            }
        )

    steps.sort(key=lambda item: (item["time"], item["gmin_val"], item["iteration"]))

    if not steps:
        stderr_obj = (completed.stderr if completed is not None else "") or ""
        if isinstance(stderr_obj, bytes):
            stderr_text = stderr_obj.decode("utf-8", errors="replace")
        else:
            stderr_text = stderr_obj
        if "LINEAR_SYSTEM_CORPUS export currently supports Sparse only" in stderr_text:
            reason = "linear_system_export_skipped_klu_active"
        elif process_error_reason is not None:
            reason = process_error_reason
        else:
            reason = "linear_system_export_no_steps"
        return {
            "success": False,
            "reason": reason,
            "steps": [],
            "task_dir": task_dir,
            "stderr": stderr_text,
            "stdout": completed.stdout or "",
        }

    return {
        "success": process_error_reason is None,
        "reason": process_error_reason,
        "steps": steps,
        "task_dir": task_dir,
        "stderr": (
            completed.stderr.decode("utf-8", errors="replace")
            if completed is not None and isinstance(completed.stderr, bytes)
            else ((completed.stderr or "") if completed is not None else "")
        ),
        "stdout": (
            completed.stdout.decode("utf-8", errors="replace")
            if completed is not None and isinstance(completed.stdout, bytes)
            else ((completed.stdout or "") if completed is not None else "")
        ),
    }


def run_ngspice_true_continuation(
    val_dir: str,
    netlist_dir: str,
    real_ckt_id: int,
    workpoint: np.ndarray,
    case: str,
    *,
    start_iteration: int,
    max_steps: int,
    gmin_val: "Optional[float]" = None,
    state0_in: Optional[np.ndarray] = None,
    reapply_wp_steps: int = 0,
) -> Dict[str, Any]:
    task_dir = os.path.join(
        val_dir,
        f"continuation_task_{os.getpid()}_{int(time.time() * 1000000) % 1000000}_{case}",
    )
    os.makedirs(task_dir, exist_ok=True)

    input_filepath = os.path.join(task_dir, f"{case}_wp_in.txt")
    np.savetxt(input_filepath, np.asarray(workpoint, dtype=np.float64), fmt="%.17e")
    state0_input_filepath = None
    if state0_in is not None:
        state0_input_filepath = os.path.join(task_dir, f"{case}_state0_in.txt")
        np.savetxt(
            state0_input_filepath,
            np.asarray(state0_in, dtype=np.float64),
            fmt="%.17e",
        )

    netlist_path = os.path.join(netlist_dir, f"{real_ckt_id}.sp")
    proc_env = os.environ.copy()
    proc_env["TRAJ"] = "0"
    proc_env["VALUE"] = "0"
    proc_env["CKT_ID"] = str(real_ckt_id)
    proc_env["WP_IN_PATH"] = input_filepath
    proc_env["CONTINUATION_MODE"] = "1"
    proc_env["CONTINUATION_START_ITER"] = str(int(start_iteration))
    proc_env["CONTINUATION_MAX_STEPS"] = str(int(max_steps))
    proc_env["CONTINUATION_DIR"] = task_dir
    if int(reapply_wp_steps) > 0:
        proc_env["CONTINUATION_REAPPLY_WP_STEPS"] = str(int(reapply_wp_steps))
    if gmin_val is not None:
        proc_env["CONTINUATION_GMIN"] = f"{float(gmin_val):.17e}"
    if state0_input_filepath is not None:
        proc_env["CONTINUATION_STATE0_PATH"] = state0_input_filepath

    if not os.path.exists(netlist_path):
        return {
            "success": False,
            "reason": f"netlist_not_found:{netlist_path}",
            "steps": [],
        }

    command = [NGSPICE_EXECUTABLE, "-b", netlist_path]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=10000, env=proc_env)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr[:200] if hasattr(e, "stderr") and e.stderr else "N/A"
        return {
            "success": False,
            "reason": f"ngspice_continuation_failed:{stderr}",
            "steps": [],
        }

    step_pattern = re.compile(r"continuation_circuit_(.+?)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)\.txt$")
    steps = []
    for filename in os.listdir(task_dir):
        match = step_pattern.match(filename)
        if not match:
            continue
        _ckt_id_str, time_val_str, gmin_val_str, iter_id_str = match.groups()
        payload = read_continuation_step(os.path.join(task_dir, filename))
        residual_vec = np.asarray(payload["residual"], dtype=np.float64)
        steps.append({
            "iteration": int(iter_id_str),
            "time": float(time_val_str),
            "gmin_val": float(gmin_val_str),
            "rhsold": np.asarray(payload["rhsold"], dtype=np.float64),
            "state0_in": np.asarray(payload["state0_in"], dtype=np.float64),
            "rhsnew": np.asarray(payload["rhsnew"], dtype=np.float64),
            "state0_out": np.asarray(payload["state0_out"], dtype=np.float64),
            "residual": residual_vec,
            "residual_norm": float(np.linalg.norm(residual_vec)) if residual_vec.size else None,
            "wp_out": np.asarray(payload["wp_out"], dtype=np.float64),
            "node_map": payload["node_map"],
            "filepath": os.path.join(task_dir, filename),
        })

    steps.sort(key=lambda item: item["iteration"])
    return {
        "success": True,
        "reason": None,
        "steps": steps,
        "task_dir": task_dir,
    }


def run_ngspice_segment_entry_rhsold_once(
    val_dir: str,
    netlist_dir: str,
    real_ckt_id: int,
    workpoint: np.ndarray,
    case: str,
    *,
    start_iteration: int,
    max_steps: int,
    gmin_val: "Optional[float]" = None,
) -> Dict[str, Any]:
    """Run one continuation segment with a single entry rhsold replacement.

    This is the conservative warmup path we want as the default semantics:
    replace only the segment-entry rhsold before the first NIiter load, then
    leave the rest of the native ngspice continuation logic unchanged.
    """

    return run_ngspice_true_continuation(
        val_dir,
        netlist_dir,
        real_ckt_id,
        workpoint,
        case,
        start_iteration=start_iteration,
        max_steps=max_steps,
        gmin_val=gmin_val,
        state0_in=None,
        reapply_wp_steps=0,
    )


def _subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_native_gmin_stage_outputs(
    capture_path: str,
    residual_path: str,
) -> List[Dict[str, Any]]:
    residuals_by_key = defaultdict(list)
    if os.path.exists(residual_path):
        with open(residual_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 5:
                    continue
                ckt_id_str, time_val_str, gmin_val_str, iterno_str, residual_norm_str = parts
                try:
                    key = (f"{float(time_val_str):.17e}", f"{float(gmin_val_str):.17e}")
                    residuals_by_key[key].append(
                        {
                            "circuit_id": ckt_id_str,
                            "time": float(time_val_str),
                            "gmin_val": float(gmin_val_str),
                            "iterno": int(iterno_str),
                            "residual_norm": float(residual_norm_str),
                        }
                    )
                except ValueError:
                    continue

    stages = []
    if os.path.exists(capture_path):
        with open(capture_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 7:
                    continue
                ckt_id_str, time_val_str, gmin_val_str, iters_str, injected_str, converged_str, method_str = parts
                try:
                    time_val = float(time_val_str)
                    gmin_val = float(gmin_val_str)
                    key = (f"{time_val:.17e}", f"{gmin_val:.17e}")
                    stage_residuals = residuals_by_key.get(key, [])
                    stages.append(
                        {
                            "circuit_id": ckt_id_str,
                            "time": time_val,
                            "gmin_val": gmin_val,
                            "iters": int(iters_str),
                            "injected": bool(int(injected_str)),
                            "converged": bool(int(converged_str)),
                            "method": method_str,
                            "residuals": [item["residual_norm"] for item in stage_residuals],
                            "residual_steps": stage_residuals,
                        }
                    )
                except ValueError:
                    continue
    stages.sort(key=lambda item: (item["time"], -item["gmin_val"]))
    return stages


def run_ngspice_joint_case(
    val_dir: str,
    netlist_dir: str,
    real_ckt_id: int,
    warmup_inputs: Dict[tuple[float, float], np.ndarray],
    case: str,
    *,
    gmres_env: Optional[Dict[str, Any]] = None,
    extra_env: Optional[Dict[str, Any]] = None,
    timeout: int = 10000,
    task_dir: Optional[str] = None,
    inherit_gmres_env: bool = False,
) -> Dict[str, Any]:
    """Run warmup injection and an explicitly configured native GMRES mode together.

    The caller owns all method choices through ``gmres_env``. By default any
    inherited ``NGSPICE_GMRES_*`` variables are removed first so parallel arms
    cannot leak configuration into one another. ``extra_env`` is reserved for
    additional explicit ngspice instrumentation variables.
    """

    case_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case)).strip("_") or "joint"
    if task_dir is None:
        task_dir = os.path.join(
            val_dir,
            f"native_gmin_task_{os.getpid()}_{int(time.time() * 1000000) % 1000000}_{case_tag}",
        )
    task_dir = os.path.abspath(task_dir)
    os.makedirs(task_dir, exist_ok=True)

    normalized_warmup_inputs = dict(warmup_inputs or {})
    for (time_val, gmin_val), workpoint in normalized_warmup_inputs.items():
        input_filepath = os.path.join(
            task_dir,
            f"segment_warmup_circuit_{real_ckt_id}_time_{float(time_val):.17e}_gmin_{float(gmin_val):.17e}_rhsold.txt",
        )
        np.savetxt(input_filepath, np.asarray(workpoint, dtype=np.float64), fmt="%.17e")

    capture_path = os.path.join(task_dir, "segment_stage_stats.tsv")
    residual_path = os.path.join(task_dir, "segment_stage_residuals.tsv")
    netlist_path = os.path.join(netlist_dir, f"{real_ckt_id}.sp")
    proc_env = os.environ.copy()
    if not inherit_gmres_env:
        for key in list(proc_env):
            if key.startswith("NGSPICE_GMRES_"):
                proc_env.pop(key, None)
    for key in list(proc_env):
        if key.startswith("PALS_ONLINE_SCHWARZ_") or key.startswith(
            "LINEAR_SYSTEM_CORPUS_"
        ):
            proc_env.pop(key, None)
    for key in (
        "SEGMENT_WARMUP_INPUT_DIR",
        "SEGMENT_WARMUP_CAPTURE_PATH",
        "SEGMENT_WARMUP_RESIDUAL_PATH",
    ):
        proc_env.pop(key, None)
    proc_env["TRAJ"] = "0"
    proc_env["VALUE"] = "0"
    proc_env["CKT_ID"] = str(real_ckt_id)
    proc_env["SEGMENT_WARMUP_INPUT_DIR"] = task_dir
    proc_env["SEGMENT_WARMUP_CAPTURE_PATH"] = capture_path
    proc_env["SEGMENT_WARMUP_RESIDUAL_PATH"] = residual_path

    explicit_env: Dict[str, str] = {}
    for updates, require_gmres_prefix in (
        (gmres_env or {}, True),
        (extra_env or {}, False),
    ):
        for raw_key, raw_value in updates.items():
            key = str(raw_key)
            if require_gmres_prefix and not key.startswith("NGSPICE_GMRES_"):
                raise ValueError(f"not a GMRES environment variable: {key}")
            if raw_value is None:
                proc_env.pop(key, None)
                continue
            if isinstance(raw_value, bool):
                value = "1" if raw_value else "0"
            else:
                value = str(raw_value)
            proc_env[key] = value
            explicit_env[key] = value

    if not os.path.exists(netlist_path):
        return {
            "success": False,
            "reason": f"netlist_not_found:{netlist_path}",
            "stages": [],
            "task_dir": task_dir,
            "netlist_path": netlist_path,
            "warmup_input_count": len(normalized_warmup_inputs),
            "explicit_env": explicit_env,
        }

    command = [NGSPICE_EXECUTABLE, "-b", netlist_path]
    started = time.perf_counter()
    timed_out = False
    returncode = None
    stdout_text = ""
    stderr_text = ""
    failure_reason = None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=int(timeout),
            env=proc_env,
        )
        returncode = int(completed.returncode)
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        if returncode != 0:
            stderr_excerpt = stderr_text[:400] if stderr_text else "N/A"
            failure_reason = f"ngspice_joint_failed:returncode={returncode}:{stderr_excerpt}"
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = _subprocess_text(exc.stdout)
        stderr_text = _subprocess_text(exc.stderr)
        failure_reason = f"ngspice_joint_timeout:{int(timeout)}s"

    stages = _read_native_gmin_stage_outputs(capture_path, residual_path)
    success = bool((not timed_out) and returncode == 0)
    return {
        "success": success,
        "reason": failure_reason,
        "stages": stages,
        "task_dir": task_dir,
        "netlist_path": netlist_path,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_sec": float(time.perf_counter() - started),
        "stdout": stdout_text,
        "stderr": stderr_text,
        "warmup_input_count": len(normalized_warmup_inputs),
        "explicit_env": explicit_env,
        "stage_capture_path": capture_path if os.path.exists(capture_path) else None,
        "stage_residual_path": residual_path if os.path.exists(residual_path) else None,
    }


def run_ngspice_native_gmin_segment_warmup(
    val_dir: str,
    netlist_dir: str,
    real_ckt_id: int,
    warmup_inputs: Dict[tuple[float, float], np.ndarray],
    case: str,
) -> Dict[str, Any]:
    """Compatibility wrapper for the historical warmup-only entry point."""

    return run_ngspice_joint_case(
        val_dir,
        netlist_dir,
        real_ckt_id,
        warmup_inputs,
        case,
        timeout=10000,
        inherit_gmres_env=True,
    )


def get_F_and_J(val_dir: str, netlist_dir: str, pred_x_padded: torch.Tensor, ckt_id: torch.Tensor, case: str, pool: Pool, local_use: bool = True, wp_out: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """将模型输出传递给ngspice并获取F和J"""
    x_physical = pred_x_padded.detach().cpu().numpy()
    ckt_ids_np = ckt_id.cpu().numpy()


    F, J, _ = run_ngspice_batch(val_dir, netlist_dir, ckt_ids_np, x_physical, case, pool, local_use, wp_out)
    
    # 检查是否有失败的情况
    if F is None:
        return None, None
    
    # 检查 F 和 J 是否是列表且包含有效数据
    if isinstance(F, list):
        # 如果列表为空，返回 None
        if len(F) == 0:
            return None, None
        # 如果列表中包含 None 或字符串，说明有失败的情况
        if any(item is None or isinstance(item, str) for item in F):
            print(f"  [ERROR] F 列表包含无效元素 (None 或字符串)")
            return None, None
        # 确保所有元素都是数值类型
        try:
            F_np = np.array(F, dtype=np.float64)
        except (ValueError, TypeError) as e:
            print(f"  [ERROR] 无法将 F 转换为数组: {e}, F 类型: {type(F)}, F 第一个元素类型: {type(F[0]) if F else 'N/A'}")
            return None, None
    elif isinstance(F, str):
        print(f"  [ERROR] F 是字符串而不是列表: {F[:100]}")
        return None, None
    else:
        try:
            F_np = np.array(F, dtype=np.float64)
        except (ValueError, TypeError) as e:
            print(f"  [ERROR] 无法将 F 转换为数组: {e}, F 类型: {type(F)}")
            return None, None
    
    if isinstance(J, list):
        if len(J) == 0:
            return None, None
        if any(item is None or isinstance(item, str) for item in J):
            print(f"  [ERROR] J 列表包含无效元素 (None 或字符串)")
            return None, None
        try:
            J_np = np.array(J, dtype=np.float64)
        except (ValueError, TypeError) as e:
            print(f"  [ERROR] 无法将 J 转换为数组: {e}, J 类型: {type(J)}, J 第一个元素类型: {type(J[0]) if J else 'N/A'}")
            return None, None
    elif isinstance(J, str):
        print(f"  [ERROR] J 是字符串而不是列表: {J[:100]}")
        return None, None
    else:
        try:
            J_np = np.array(J, dtype=np.float64)
        except (ValueError, TypeError) as e:
            print(f"  [ERROR] 无法将 J 转换为数组: {e}, J 类型: {type(J)}")
            return None, None
    
    # 2. 再从这个单一的NumPy数组高效地创建PyTorch张量
    try:
        F_val = torch.from_numpy(F_np).to(pred_x_padded.device)
        J_val = torch.from_numpy(J_np).to(pred_x_padded.device)
        return F_val, J_val
    except Exception as e:
        print(f"  [ERROR] 创建 PyTorch 张量失败: {e}")
        return None, None

def get_F_J_WPout(val_dir: str, netlist_dir: str, pred_x_padded: torch.Tensor, ckt_id: torch.Tensor, case: str, pool: Pool):
    """将模型输出传递给ngspice并获取F和J"""
    x_physical = pred_x_padded.detach().cpu().numpy()
    ckt_ids_np = ckt_id.cpu().numpy()


    F, J, WP_OUT = run_ngspice_batch(val_dir, netlist_dir, ckt_ids_np, x_physical, case, pool, local_use=True, wp_out=True)
    
    F_np = np.array(F, dtype=np.float64)
    J_np = np.array(J, dtype=np.float64)
    WP_OUT_np = np.array(WP_OUT, dtype=np.float64)
    
    # 2. 再从这个单一的NumPy数组高效地创建PyTorch张量
    F_val = torch.from_numpy(F_np).to(pred_x_padded.device)
    J_val = torch.from_numpy(J_np).to(pred_x_padded.device)
    WP_OUT_val = torch.from_numpy(WP_OUT_np).to(pred_x_padded.device)
    return F_val, J_val, WP_OUT_val

def run_ngspice_batch(val_dir:str, netlist_dir:str, ckt_ids: np.ndarray, x_batch: np.ndarray, case: str, pool: Pool, local_use: bool, wp_out: bool) -> tuple[list, list, list]:
    """
    并行版：为批次中的每个预测工作点调用ngspice，以获取残差和雅可比。
    """
    pid = os.getpid()
    tasks = []
    for i in range(len(x_batch)):
        task_args = (
                val_dir,
                netlist_dir,
                int(ckt_ids[i]),
            x_batch[i],
            case,
            pid,
            local_use,
            wp_out
        )
        tasks.append(task_args)
    if PARALLEL_EXECUTION and pool:
        results = pool.map(run_single_ngspice_task, tasks)

    else:
        # --- 串行模式 (用于调试) ---
        results = []
        # 使用一个简单的 for 循环，逐个执行任务
        for i, task_args in enumerate(tasks):
            # 直接在主进程中调用任务函数
            result = run_single_ngspice_task(task_args)
            results.append(result)
            
    res_list = []
    jac_list = []
    wp_out_list = []
    for result in results:
        if result is None:
            # ngspice 失败，返回 None
            res_list.append(None)
            jac_list.append(None)
            if wp_out:
                wp_out_list.append(None)
        else:
            res, jac, wp_out = result
            res_list.append(res)
            jac_list.append(jac)
            if wp_out:
                wp_out_list.append(wp_out)
    
    if wp_out:
        return res_list, jac_list, wp_out_list
    else:
        return res_list, jac_list, None

def run_single_ngspice_task(args):
    """
    一个独立的、可被并行调用的函数，负责执行单次ngspice仿真。
    """
    # 1. 从元组中解包所有参数
    val_dir, netlist_dir, real_ckt_id, x_predicted_padded, case_str, pid, local_use, wp_out = args
    
    # 为每个任务创建唯一的子目录，避免并行时的文件冲突
    # 使用 pid、case_str 和时间戳确保唯一性
    import time
    # 在进程池中，每个任务都在独立进程中，pid已经足够唯一
    # 但添加时间戳和case_str可以进一步确保唯一性，并便于调试
    timestamp = int(time.time() * 1000000) % 1000000  # 微秒级时间戳的后6位
    task_unique_id = f"task_{pid}_{timestamp}_{case_str}"
    task_val_dir = os.path.join(val_dir, task_unique_id)
    
    # 确保任务目录存在（如果已存在则先删除，避免残留文件）
    if os.path.exists(task_val_dir):
        shutil.rmtree(task_val_dir)
    os.makedirs(task_val_dir, exist_ok=True)
    
    netlist_path = os.path.join(netlist_dir, f"{real_ckt_id}.sp")
    # 2. 【核心逻辑】(与旧的run_ngspice_batch循环体内部几乎完全相同)
    
    # 唯一文件名
    input_filename = f"{case_str}_pid_{pid}_batchidx_{real_ckt_id}_wp_in.txt"
    output_filename_F = f"{case_str}_pid_{pid}_batchidx_{real_ckt_id}_F.txt"
    output_filename_J = f"{case_str}_pid_{pid}_batchidx_{real_ckt_id}_J.txt"
    wp_output_filename = f"{case_str}_pid_{pid}_batchidx_{real_ckt_id}_WP_OUT.txt"
    input_filepath = os.path.join(val_dir, input_filename)
    output_filepath_F = os.path.join(val_dir, output_filename_F)
    output_filepath_J = os.path.join(val_dir, output_filename_J)
    wp_output_filepath = os.path.join(val_dir, wp_output_filename)
    
    # 环境变量
    proc_env = os.environ.copy()
    proc_env["VALUE"] = "1"; 
    proc_env["TRAJ"] = "0"
    proc_env["CKT_ID"] = str(real_ckt_id)
    proc_env["F_PATH"] = output_filepath_F
    proc_env["J_PATH"] = output_filepath_J
    proc_env["WP_IN_PATH"] = input_filepath

    if wp_out:
        proc_env["CLOSE_LOOP_TRAIN"] = "1"
        proc_env["CLOSE_LOOP_PATH"] = wp_output_filepath
    
    # 写入输入文件
    np.savetxt(input_filepath, x_predicted_padded, fmt="%.17e")
    
    # # 网表路径
    # if local_use:
    #     uut_dir = os.path.dirname(model_dir)
    #     netlist_dir = os.path.join(uut_dir, config.NETLIST_DIR)
    #     netlist_path = os.path.abspath(os.path.join(netlist_dir, f"{real_ckt_id}.cir"))
    # else:
    #     netlist_dir = os.path.join(model_dir, config.NETLIST_DIR)
    #     netlist_path = os.path.abspath(os.path.join(netlist_dir, f"circuit_{real_ckt_id:03d}.sp"))
    
    
    # 运行ngspice并读取结果，使用try-finally确保清理临时目录
    try:
        if os.path.exists(netlist_path):
            command = [NGSPICE_EXECUTABLE, "-b", netlist_path]
            # print(f"command = {command}")
            try:
                subprocess.run(command, capture_output=True, text=True, check=True, timeout=10000, env=proc_env)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  [NGSPICE FAIL async] Sim failed for ckt_id {real_ckt_id}. Stderr: {e.stderr[:100] if hasattr(e, 'stderr') else 'N/A'}...")
                # 失败时返回None
                return None
        else:
            print(f"  [NGSPICE FAIL] Netlist file not found: {netlist_path}")
            return None

        # 读取结果
        try:
            # 我们在这里捕获异常
            res = read_F(output_filepath_F)
            jac = read_J(output_filepath_J)
            if wp_out:
                wp_out_value = read_wp_value(wp_output_filepath)
            else:
                wp_out_value = None
        except Exception as e:
            # 获取当前进程的 ID
            current_pid = os.getpid()
            # 打印进程号和出问题的文件名
            print(f"[ERROR] Process ID: {current_pid}, File causing error: {output_filepath_J}")
            # 打印原始错误信息，以便调试
            print(f"      Exception: {e}")
            # 捕获到错误后，必须重新抛出异常，否则主进程会认为任务成功执行，从而导致后续逻辑错误
            raise
        
        # 填充并返回结果
        if wp_out:
            return res, jac, wp_out_value
        else:
            return res, jac, None
    
    finally:
        # 清理临时任务目录（可选：如果希望保留文件用于调试，可以注释掉这部分）
        # 注意：在并行执行时，清理目录是安全的，因为每个任务都有独立的子目录
        try:
            if os.path.exists(task_val_dir):
                shutil.rmtree(task_val_dir)
        except Exception as e:
            # 如果清理失败，只打印警告，不影响主流程
            print(f"  [WARNING] Failed to cleanup task directory {task_val_dir}: {e}")
