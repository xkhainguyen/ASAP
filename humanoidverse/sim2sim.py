import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import onnx
import onnxruntime

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":

    SINGLE_FRAME = False
    LINER_VELOCITY = False

    # policy_path = "logs/MotionTracking/20250412_195040-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/exported/model_6600.onnx"

    policy_path = "logs/MotionTracking/20250412_215612-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/exported/model_37000.onnx"

    # simple
    # policy_path = "logs/MotionTracking/20250412_194908-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/exported/model_2500.onnx"

    # policy_path = "logs/MotionTracking/20250412_194908-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/exported/model_50000.onnx"

    xml_path = "humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_deploy.xml"
    print("policy_path: ", policy_path)
    print("xml_path   : ", xml_path)

    # define context variables
    simulation_dt = 0.001
    control_dt = 1 / 50
    control_decimation = int(control_dt / simulation_dt)

    simulation_duration = 60
    counter = 0
    num_actions = 23

    action = np.zeros(num_actions, dtype=np.float32)

    # default_angles =  np.array([ -0.1, 0.0, 0.0, 0.3, -0.2, 0., 
    #                              -0.1, 0.0, 0.0, 0.3, -0.2, 0., 
    #                               0.0, 0.0, 0.0, 
    #                               0.0, 0.0, 0.0, 0.0, 
    #                               0.0, 0.0, 0.0, 0.0 ], dtype=np.float32)
    
    default_angles =  np.array([ -0.1, 0.0, 0.0, 0.3, -0.2, -0.0, 
                                 -0.1, 0.0, 0.0, 0.3, -0.2, -0.0, 
                                  0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0 ], dtype=np.float32)  #TODO: offset the ankles instead of delta action
    
    kps = np.array([ 100, 100, 100, 200, 20, 20, 
                     100, 100, 100, 200, 20, 20, 
                     400, 400, 400, 
                     90,   60,  20, 60, 
                     90,   60,  20, 60 ], dtype=np.float32)
    kds = np.array([ 2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 
                     2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 
                     5.0, 5.0, 5.0, 
                     2.0, 1.0, 0.4, 1.0, 
                     2.0, 1.0, 0.4, 1.0 ], dtype=np.float32)
    
    tau_limit = np.array([88, 88, 88, 139, 50, 50, 
                          88, 88, 88, 139, 50, 50,
                          88, 50, 50, 
                          25, 25, 25, 25, 
                          25, 25, 25, 25], dtype=np.float32)

    dof_upper_limit = np.array([
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
        2.618,  0.52,   0.52,
        2.6704, 2.2515, 2.618, 2.0944,
        2.6704, 1.5882, 2.618, 2.0944
    ], dtype=np.float32) * 0.95

    dof_lower_limit = np.array([
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, 
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618,  -0.52,   -0.52,
    -3.0892, -1.5882, -2.618, -1.0472,
    -3.0892, -2.2515, -2.618, -1.0472
    ], dtype=np.float32) * 0.95

    target_dof_pos = default_angles.copy()
    ref_motion_phase = 0

    history_length = 4  
    lin_vel_buf = np.zeros(3 * history_length, dtype=np.float32)
    ang_vel_buf = np.zeros(3 * history_length, dtype=np.float32)
    proj_g_buf = np.zeros(3 * history_length, dtype=np.float32)
    dof_pos_buf = np.zeros(23 * history_length, dtype=np.float32)
    dof_vel_buf = np.zeros(23 * history_length, dtype=np.float32)
    action_buf = np.zeros(23 * history_length, dtype=np.float32)
    ref_motion_phase_buf = np.zeros(1 * history_length, dtype=np.float32)

    # load onnx model
    onnx_model = onnx.load(policy_path)
    ort_session = onnxruntime.InferenceSession(policy_path)
    input_name = ort_session.get_inputs()[0].name

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    # Set contact parameters per geom
    for i in range(m.ngeom):
        m.geom_solref[i] = np.array([0.001, 1.0])                    # Stiff contact
        m.geom_solimp[i] = np.array([0.9, 0.95, 0.001, 0.5, 1.0])     # Less softness

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            tau = np.clip(tau, -tau_limit, tau_limit)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]     
                dqj = d.qvel[6:]    
                quat = d.qpos[3:7]        
                lin_vel = d.qvel[:3]
                ang_vel = d.qvel[3:6]

                projected_gravity = get_gravity_orientation(quat)
                dof_pos = qj * 1.0
                dof_vel = dqj * 0.05
                base_ang_vel = ang_vel* 0.25
                base_lin_vel = lin_vel * 2.0
                
                if (ref_motion_phase < 1.0): # always in [0, 1]
                    ref_motion_phase += 0.0141  #TODO: compute the phase based on motion length and episode length
                else:
                    ref_motion_phase = 1.0
                
                """_summary_
                    curr_obs:
                            base_lin_vel        3
                            base_ang_vel         3
                            projected_gravity    3
                            dof_pos              23
                            dof_vel              23
                            actions              23
                            ref_motion_phase     1
                """

                if SINGLE_FRAME:
                    # 1 single frame  
                    current_obs = np.concatenate((
                                                action,
                                                base_ang_vel,
                                                base_lin_vel,
                                                dof_pos,
                                                dof_vel,
                                                projected_gravity,
                                                np.array([ref_motion_phase])
                                ), axis=-1, dtype=np.float32)
                    obs_buf = current_obs
                
                elif not SINGLE_FRAME:
                    if LINER_VELOCITY:
                        # 2 history frames with liner velocity
                        history_obs_buf = np.concatenate((action_buf, ang_vel_buf, lin_vel_buf, dof_pos_buf, dof_vel_buf, proj_g_buf, ref_motion_phase_buf), axis=-1, dtype=np.float32)
                        obs_buf = np.concatenate((action, base_ang_vel, base_lin_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, np.array([ref_motion_phase])), axis=-1, dtype=np.float32)
                    elif not LINER_VELOCITY:
                        # 3 history frames without liner velocity
                        history_obs_buf = np.concatenate((action_buf, ang_vel_buf, dof_pos_buf, dof_vel_buf, proj_g_buf, ref_motion_phase_buf), axis=-1, dtype=np.float32)
                        obs_buf = np.concatenate((action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, np.array([ref_motion_phase])), axis=-1, dtype=np.float32)
                    else:
                        assert False
                else: assert False

                obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
                action = np.squeeze(ort_session.run(None, {input_name: obs_tensor})[0])
                # action = np.clip(action, -100, 100)
                print(action)
                # transform action to target_dof_pos
                target_dof_pos = action * 0.25 + default_angles 
                
                # target_dof_pos = np.clip(target_dof_pos, dof_lower_limit, dof_upper_limit)

                # update history
                ang_vel_buf = np.concatenate((base_ang_vel, ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
                lin_vel_buf = np.concatenate((base_lin_vel, lin_vel_buf[:-3]), axis=-1, dtype=np.float32)
                proj_g_buf = np.concatenate((projected_gravity, proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
                dof_pos_buf = np.concatenate((dof_pos, dof_pos_buf[:-23] ), axis=-1, dtype=np.float32)
                dof_vel_buf = np.concatenate((dof_vel, dof_vel_buf[:-23] ), axis=-1, dtype=np.float32)
                action_buf = np.concatenate((action, action_buf[:-23] ), axis=-1, dtype=np.float32)
                ref_motion_phase_buf = np.concatenate((np.array([ref_motion_phase]), ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)   

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
