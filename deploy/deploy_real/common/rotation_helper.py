import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    # get directional vector or projection of gravity vector in a local frame
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    # input:
    # imu/torso_quat in some global frame (Initial frame == World frame)
    # imu/torso_omega in some frame (Initial frame == World frame)
    # print("imu_quat: ", imu_quat)
    # print("imu_omega: ", imu_omega)
    # print("waist_yaw: ", waist_yaw)
    # print("waist_yaw_omega: ", waist_yaw_omega)

    R_torso_pelvis = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()  # which frame? initial powered IMU frame?
    # compared to mujoco, R_torso wrt to some global frame
    # R_pelvis = np.dot(R_torso, R_torso_pelvis.T)  # in some global frame, why not R_torso_pelvis.T @ R_torso? #TODO: printing both
    R_pelvis = np.dot(R_torso_pelvis.T, R_torso)  # in some global frame, why not R_torso_pelvis.T @ R_torso? #TODO: printing both

    # print("R_torso x R_torso_pelvis.T: ", np.dot(R_torso, R_torso_pelvis.T))
    # print("R_torso_pelvis.T x R_torso: ", np.dot(R_torso_pelvis.T, R_torso))

    w_pelvis = np.dot(R_torso_pelvis, imu_omega) - np.array([0, 0, waist_yaw_omega]) # why minus?
    # w_pelvis = np.dot(R_torso_pelvis, imu_omega[0]) - np.array([0, 0, waist_yaw_omega]) # why minus?

    # print("imu_omega: ", imu_omega)
    # print("imu_omega[0]: ", imu_omega[0])
    # output:
    # R_pelvis in some global frame (Initial frame == World frame)
    # w_pelvis in pelvis frame
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w_pelvis

