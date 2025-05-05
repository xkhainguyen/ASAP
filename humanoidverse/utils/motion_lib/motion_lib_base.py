import glob
import gc
import os
import os.path as osp
import numpy as np
import joblib
import torch
import random

from humanoidverse.utils.motion_lib.motion_utils.flags import flags
from enum import Enum
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from easydict import EasyDict
from loguru import logger
from rich.progress import track

from isaac_utils.rotations import(
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
)

from concurrent.futures import ThreadPoolExecutor

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)
    
def indexing_tensor(tensor: torch.Tensor, indices: torch.Tensor):
    return tensor[indices]

class MotionLibBase():
    def __init__(self, motion_lib_cfg, num_envs, device, is_training=True):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants(fix_height = False,  multi_thread = False)
        if flags.real_traj:
            self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        if is_training:
            self.start_stream_idx = 0
        else:
            self.start_stream_idx = 8
        print(self.start_stream_idx)
        self.num_cuda_streams = 16
        self.cuda_streams = [torch.cuda.Stream(device=self._device) for _ in range(self.num_cuda_streams)]
        self.experiment = False
        self.offload_dir = "/workspace/data/ASAP/tensors"
        if self.experiment:
            os.makedirs(self.offload_dir, exist_ok=True)

        return
        
    def load_data(self, motion_file, min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        # import ipdb; ipdb.set_trace()
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        action = self._motion_actions[f0l]
        return action
    
    def has_tensor(self, tensor_name: str) -> bool:
        if isinstance(getattr(self, tensor_name, None), torch.Tensor):
            return True
        if osp.isfile(osp.join(self.offload_dir, f"{tensor_name}.pt")):
            return True
        return False
    
    def onload_tensor(self, tensor_name: str, device: str) -> None:
        if self.experiment:
            if isinstance(getattr(self, tensor_name, None), torch.Tensor):
                setattr(self, tensor_name, getattr(self, tensor_name).to(device))
            else:
                tensor_path = osp.join(self.offload_dir, f"{tensor_name}.pt")
                setattr(self, tensor_name, torch.load(tensor_path, map_location=device))
        
    def offload_tensor(self, tensor_name: str, device: str) -> None:
        if self.experiment:
            if isinstance(getattr(self, tensor_name, None), torch.Tensor):
                if device == "cpu":
                    setattr(self, tensor_name, getattr(self, tensor_name).cpu())
                elif device == "disk":
                    tensor_path = osp.join(self.offload_dir, f"{tensor_name}.pt")
                    torch.save(getattr(self, tensor_name), tensor_path)
                else:
                    raise ValueError(f"Invalid device: {device}. Must be one of [\"cpu\", \"disk\"]")
            
    def free_tensor(self, tensor_name: str) -> None:
        if self.experiment:
            delattr(self, tensor_name)
            gc.collect()
            torch.cuda.empty_cache()

    def index_cpu_tensor(self, tensor_name: str, indices: torch.Tensor) -> torch.Tensor:
        tensor = getattr(self, tensor_name)
        return tensor[indices].pin_memory().to(self._device, non_blocking=True)
    
    def index_gpu_tensor(self, tensor_name: str, indices: torch.Tensor, stream_idx: int) -> torch.Tensor:
        tensor = getattr(self, tensor_name)

        with torch.cuda.stream(self.cuda_streams[stream_idx]):
            if tensor_name == "gts":
                return tensor[indices, :]
            else:
                return tensor[indices]

    def index_and_blend_gpu_tensor(self, tensor_name: str, index0: torch.Tensor, index1: torch.Tensor, blend_exp: torch.Tensor, offset: torch.Tensor = None, stream_idx: int = 0) -> torch.Tensor:
        tensor = getattr(self, tensor_name)
        
        with torch.cuda.stream(self.cuda_streams[stream_idx]):
            if tensor_name == "gts":
                tensor0, tensor1 = tensor[index0, :], tensor[index1, :]
            else:
                tensor0, tensor1 = tensor[index0], tensor[index1]
            
            if offset is None:
                return (1.0 - blend_exp) * tensor0 + blend_exp * tensor1
            else:
                return (1.0 - blend_exp) * tensor0 + blend_exp * tensor1 + offset[..., None, :]  # ZL: apply offset

    def get_motion_state_pcoc(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        indices = torch.stack([f0l, f1l], dim=0).cpu().contiguous()

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)
        torch.cuda.synchronize(self._device)
        dof_pos = self.index_and_blend_gpu_tensor("dof_pos", f0l, f1l, blend, None, 0 + self.start_stream_idx)
        dof_vel = self.index_and_blend_gpu_tensor("dvs", f0l, f1l, blend, None, 1 + self.start_stream_idx)
        tensor_names = ["gts_t", "grs_t", "gvs_t", "gavs_t"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            indexed_tensors = list(
                executor.map(
                    lambda x: self.index_cpu_tensor(x, indices),
                    tensor_names
                )
            )
        torch.cuda.synchronize(self._device)
        rg_pos_t, rg_rot_t, body_vel_t, body_ang_vel_t = indexed_tensors

        rg_pos_t0, rg_pos_t1 = rg_pos_t[0], rg_pos_t[1]
        rg_rot_t0, rg_rot_t1 = rg_rot_t[0], rg_rot_t[1]
        body_vel_t0, body_vel_t1 = body_vel_t[0], body_vel_t[1]
        body_ang_vel_t0, body_ang_vel_t1 = body_ang_vel_t[0], body_ang_vel_t[1]

        if offset is None:
            rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
        else:
            rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
        rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
        body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
        body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        
        return_dict = {}

        return_dict.update({
            "dof_pos": dof_pos.clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "rg_pos_t": rg_pos_t,
            "rg_rot_t": rg_rot_t,
            "body_vel_t": body_vel_t,
            "body_ang_vel_t": body_ang_vel_t,
        })

        return return_dict
    
    def get_motion_state_rrs(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        indices = torch.stack([f0l, f1l], dim=0)

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        torch.cuda.synchronize(self._device)
        body_vel = self.index_and_blend_gpu_tensor("gvs", f0l, f1l, blend_exp, None, 2 + self.start_stream_idx)
        body_ang_vel = self.index_and_blend_gpu_tensor("gavs", f0l, f1l, blend_exp, None, 3 +  self.start_stream_idx)
        rg_pos = self.index_and_blend_gpu_tensor("gts", f0l, f1l, blend_exp, offset, 4 + self.start_stream_idx)
        rb_rot = self.index_gpu_tensor("grs", indices, 5)
        torch.cuda.synchronize(self._device)
        rb_rot0, rb_rot1 = rb_rot[0], rb_rot[1]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        
        return_dict = {}        

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
        })

        return return_dict
    
    def get_motion_state_rd(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        blend = blend.unsqueeze(-1)
        torch.cuda.synchronize(self._device)
        dof_pos = self.index_and_blend_gpu_tensor("dof_pos", f0l, f1l, blend, None, 6 + self.start_stream_idx)
        dof_vel = self.index_and_blend_gpu_tensor("dvs", f0l, f1l, blend, None, 7 + self.start_stream_idx)
        torch.cuda.synchronize(self._device)

        return_dict = {}

        return_dict.update({
            "dof_pos": dof_pos.clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
        })

        return return_dict

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = (frame_idx0 + self.length_starts[motion_ids])
        f1l = (frame_idx1 + self.length_starts[motion_ids])

        f0l_ = f0l.cpu().contiguous()
        f1l_ = f1l.cpu().contiguous()

        if self.has_tensor("dof_pos"):
            self.onload_tensor("dof_pos", self._device)
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
            self.free_tensor("dof_pos")
        else:
            self.lrs = self.onload_tensor("lrs", self._device)
            local_rot0 = self.lrs[f0l_].pin_memory().to(self._device, non_blocking=True)
            local_rot1 = self.lrs[f1l_].pin_memory().to(self._device, non_blocking=True)
            self.free_tensor("lrs")
        
        self.onload_tensor("gvs", self._device)
        body_vel0 = self.gvs[f0l_].pin_memory().to(self._device, non_blocking=True)
        body_vel1 = self.gvs[f1l_].pin_memory().to(self._device, non_blocking=True)
        self.free_tensor("gvs")

        self.onload_tensor("gavs", self._device)
        body_ang_vel0 = self.gavs[f0l_].pin_memory().to(self._device, non_blocking=True)
        body_ang_vel1 = self.gavs[f1l_].pin_memory().to(self._device, non_blocking=True)
        self.free_tensor("gavs")

        self.onload_tensor("gts", self._device)         
        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]
        self.free_tensor("gts")

        self.onload_tensor("dvs", self._device)
        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]
        self.free_tensor("dvs")

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if self.has_tensor("dof_pos"): # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)
        
        self.onload_tensor("grs", self._device)
        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        self.free_tensor("grs")

        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if self.has_tensor("gts_t"):
            self.onload_tensor("gts_t", self._device)
            rg_pos_t0 = self.gts_t[f0l_].pin_memory().to(self._device, non_blocking=True)
            rg_pos_t1 = self.gts_t[f1l_].pin_memory().to(self._device, non_blocking=True)
            self.free_tensor("gts_t")
            
            self.onload_tensor("grs_t", self._device)
            rg_rot_t0 = self.grs_t[f0l_].pin_memory().to(self._device, non_blocking=True)
            rg_rot_t1 = self.grs_t[f1l_].pin_memory().to(self._device, non_blocking=True)
            self.free_tensor("grs_t")
            
            self.onload_tensor("gvs_t", self._device)
            body_vel_t0 = self.gvs_t[f0l_].pin_memory().to(self._device, non_blocking=True)
            body_vel_t1 = self.gvs_t[f1l_].pin_memory().to(self._device, non_blocking=True)
            self.free_tensor("gvs_t")
            
            self.onload_tensor("gavs_t", self._device)
            body_ang_vel_t0 = self.gavs_t[f0l_].pin_memory().to(self._device, non_blocking=True)
            body_ang_vel_t1 = self.gavs_t[f1l_].pin_memory().to(self._device, non_blocking=True)
            self.free_tensor("gavs_t")

            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        
        if flags.real_traj:
            self.onload_tensor("q_gavs", self._device)
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            self.free_tensor("q_gavs")

            self.onload_tensor("q_grs", self._device)
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            self.free_tensor("q_grs")

            self.onload_tensor("q_gts", self._device)
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            self.free_tensor("q_gts")

            self.onload_tensor("q_gvs", self._device)
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]
            self.free_tensor("q_gvs")

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel

        self.onload_tensor("_motion_aa", self._device)
        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "motion_bodies": self._motion_bodies[motion_ids],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "rg_pos_t": rg_pos_t,
            "rg_rot_t": rg_rot_t,
            "body_vel_t": body_vel_t,
            "body_ang_vel_t": body_ang_vel_t,
        })
        self.free_tensor("_motion_aa")

        return return_dict
    
    def load_motions(self, 
                     random_sample=True, 
                     start_idx=0, 
                     max_len=-1, 
                     target_heading = None):
        # import ipdb; ipdb.set_trace()

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False
        _motion_actions = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        num_motion_to_load = self.num_envs

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device) # (4096,)
        else:
            sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device) # (4096,)

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]
        # self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:5]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:5]}, ....")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)
        for f in track(range(len(res_acc)), description="Loading motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
            del curr_motion
        
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32).contiguous()       # (4096,)
        logger.info("self._motion_lengths:", self._motion_lengths.device, "-", self._motion_lengths.shape)

        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32).contiguous()               # (4096,)
        logger.info("self._motion_fps:", self._motion_fps.device, "-", self._motion_fps.shape)

        self._motion_bodies = torch.stack(_motion_bodies).to(self._device, torch.float32).contiguous()                    # (4096, 17)
        logger.info("self._motion_bodies:", self._motion_bodies.device, "-", self._motion_bodies.shape)

        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), dtype=torch.float32).contiguous().pin_memory()         # (4743168, 72)
        logger.info("self._motion_aa:", self._motion_aa.device, "-", self._motion_aa.shape)
        self.offload_tensor("_motion_aa", "disk")
        self.free_tensor("_motion_aa")

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32).contiguous()                 # (4096,)
        logger.info("self._motion_dt:", self._motion_dt.device, "-", self._motion_dt.shape)

        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device).contiguous()                      # (4096,)
        logger.info("self._motion_num_frames:", self._motion_num_frames.device, "-", self._motion_num_frames.shape)

        # import ipdb; ipdb.set_trace()
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).to(self._device, torch.float32).contiguous()
            logger.info("self._motion_actions:", self._motion_actions.device, "-", self._motion_actions.shape)
            self.offload_tensor("_motion_actions", "disk")
            self.free_tensor("_motion_actions")

        self._num_motions = len(motions)
        
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).to(self._device, torch.float32).contiguous()             # (4743168, 24, 3)
        _num_frames = self.gts.shape[0]
        logger.info("self.gts:", self.gts.device, "-", self.gts.shape)
        self.offload_tensor("gts", "disk")
        self.free_tensor("gts")

        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).to(self._device, torch.float32).contiguous()                # (4743168, 24, 4)
        logger.info("self.grs:", self.grs.device, "-", self.grs.shape)
        self.offload_tensor("grs", "disk")
        self.free_tensor("grs")

        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().contiguous().pin_memory()                            # (4743168, 27, 4)
        logger.info("self.lrs:", self.lrs.device, "-", self.lrs.shape)
        self.offload_tensor("lrs", "disk")
        self.free_tensor("lrs")
        
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).to(self._device, torch.float32).contiguous()          # (4743168, 3)
        logger.info("self.grvs:", self.grvs.device, "-", self.grvs.shape)
        self.offload_tensor("grvs", "disk")
        self.free_tensor("grvs")

        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).to(self._device, torch.float32).contiguous() # (4743168, 3)
        logger.info("self.gravs:", self.gravs.device, "-", self.gravs.shape)
        self.offload_tensor("gravs", "disk")
        self.free_tensor("gravs")
        
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).to(self._device, torch.float32).contiguous()       # (4743168, 24, 3)
        logger.info("self.gavs:", self.gavs.device, "-", self.gavs.shape)
        self.offload_tensor("gavs", "disk")
        self.free_tensor("gavs")

        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).to(self._device, torch.float32).contiguous()                # (4743168, 24, 3)
        logger.info("self.gvs:", self.gvs.device, "-", self.gvs.shape)
        self.offload_tensor("gvs", "disk")
        self.free_tensor("gvs")
        
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).to(self._device, torch.float32).contiguous()                       # (4743168, 23)
        logger.info("self.dvs:", self.dvs.device, "-", self.dvs.shape)
        self.offload_tensor("dvs", "disk")
        self.free_tensor("dvs")
        
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().contiguous().pin_memory()           # (4743168, 27, 3)
            logger.info("self.gts_t:", self.gts_t.device, "-", self.gts_t.shape)
            # self.offload_tensor("gts_t", "disk")
            # self.free_tensor("gts_t")

            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().contiguous().pin_memory()              # (4743168, 27, 4)
            logger.info("self.grs_t:", self.grs_t.device, "-", self.grs_t.shape)
            # self.offload_tensor("grs_t", "disk")
            # self.free_tensor("grs_t")

            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().contiguous().pin_memory()              # (4743168, 27, 3)
            logger.info("self.gvs_t:", self.gvs_t.device, "-", self.gvs_t.shape)
            # self.offload_tensor("gvs_t", "disk")
            # self.free_tensor("gvs_t")

            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().contiguous().pin_memory()     # (4743168, 27, 3)
            logger.info("self.gavs_t:", self.gavs_t.device, "-", self.gavs_t.shape)
            # self.offload_tensor("gavs_t", "disk")
            # self.free_tensor("gavs_t")
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).to(self._device, torch.float32).contiguous() # (4743168, 23)
            logger.info("self.dof_pos:", self.dof_pos.device, "-", self.dof_pos.shape)
            # self.offload_tensor("dof_pos", "disk")
            # self.free_tensor("dof_pos")

        # import ipdb; ipdb.set_trace()
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).to(self._device, torch.float32).contiguous()
            logger.info("self.q_gts:", self.q_gts.device, "-", self.q_gts.shape)
            self.offload_tensor("q_gts", "disk")
            self.free_tensor("q_gts")

            self.q_grs = torch.cat(self.q_grs, dim=0).to(self._device, torch.float32).contiguous()
            logger.info("self.q_grs:", self.q_grs.device, "-", self.q_grs.shape)
            self.offload_tensor("q_grs", "disk")
            self.free_tensor("q_grs")
            
            self.q_gavs = torch.cat(self.q_gavs, dim=0).to(self._device, torch.float32).contiguous()
            logger.info("self.q_gavs:", self.q_gavs.device, "-", self.q_gavs.shape)
            self.offload_tensor("q_gavs", "disk")
            self.free_tensor("q_gavs")
            
            self.q_gvs = torch.cat(self.q_gvs, dim=0).to(self._device, torch.float32).contiguous()
            logger.info("self.q_gvs:", self.q_gvs.device, "-", self.q_gvs.shape)
            self.offload_tensor("q_gvs", "disk")
            self.free_tensor("q_gvs")
        
        lengths = self._motion_num_frames # (4096,)
        lengths_shifted = lengths.roll(1) # (4096,)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0) #(4096,)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device).contiguous() # (4096,)
        # motion = motions[0]
        self.num_bodies = self.num_joints
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {_num_frames} frames.")
        return motions

    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():
            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff
            
            return trans, height_diff

    def load_motion_with_skeleton(self,
                                  motion_data_list,
                                  fix_height,
                                  target_heading,
                                  max_len):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():
                self.has_action = True
            
            dt = 1/curr_file['fps']

            B, J, N = pose_aa.shape

            if not target_heading is None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ])))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))

            if self.mesh_parsers is not None:
                # trans, trans_fix = MotionLibRobot.fix_trans_height(pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
                curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
                # add "action" to curr_motion
                if self.has_action:
                    curr_motion.action = to_torch(curr_file['action']).clone()[start:end]
                res[f] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")
        return res
    

    def num_motions(self):
        return self._num_motions


    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]


    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend


    def _get_num_bodies(self):
        return self.num_bodies


    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)