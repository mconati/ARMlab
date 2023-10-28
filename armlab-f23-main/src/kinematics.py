"""!
Implements Forward and Inverse kinematics with DH parameters and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
from scipy.spatial.transform import Rotation


def clamp(angle: float) -> float:
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def vec3_to_skew_matrix(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0],
        ]
    )


def skew_matrix_to_vec3(mat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            mat[2, 1],
            mat[0, 2],
            mat[1, 0],
        ]
    )


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    pass


def get_euler_angles_from_T(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    return Rotation.from_matrix(T[:3, :3]).as_euler("zyz")  # type: ignore


def get_pose_from_T(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    return np.concatenate((T[:3, 3], get_euler_angles_from_T(T)))  # type: ignore


def to_s_matrix(
    w: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """!
    @brief      Convert to s matrix.

    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     A unit vector describing the axis of rotation.
    @param      v     The linear velocity of the base link with respect to this rotation.

    @return     The "S" matrix, before matrix exponentiation.
    """
    S = np.zeros((4, 4))
    S[:3, :3] = vec3_to_skew_matrix(w)
    S[:3, 3] = v
    return S


def vec6_to_s_matrix(screw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return to_s_matrix(screw[:3], screw[3:])


def from_s_matrix(
    S: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    returns (w, v): axis of rotation, linear velocity respectively as column vectors
    """
    w = skew_matrix_to_vec3(S[:3, :3])
    v = S[:3, 3]

    return w.reshape(3, 1), v.reshape(3, 1)


def FK_pox(
    joint_angles: Union[list[float], npt.NDArray[np.float64]],
    m_mat: npt.NDArray[np.float64],
    s_lst: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rxarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    M: npt.NDArray[np.float64] = np.copy(m_mat)

    for i in reversed(range(len(joint_angles))):
        angle = joint_angles[i]

        # TODO: our screw vectors use a different convention, so this is a
        # temporary fix
        if 1 <= i <= 3:
            angle = -angle

        screw_vec = s_lst[i]

        S = to_s_matrix(screw_vec[:3], screw_vec[3:])
        M = scipy.linalg.expm(S * angle) @ M

    return M


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass


def AdjMatrix(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    input: 4x4 transformation matrix
    output: 6x6 matrix
    """

    R = T[:3, :3]
    p = T[:3, 3]

    # Create the following matrix:
    #       R | 0
    # [p] @ R | R

    Adj = np.zeros((6, 6))
    Adj[0:3, 0:3] = R
    Adj[3:6, 0:3] = vec3_to_skew_matrix(p) @ R
    Adj[3:6, 3:6] = R

    return Adj


def GetNewtonJacobian(joint_angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    joint_angles = joint_angles.copy()
    joint_angles = -joint_angles

    screw_vectors = np.array(
        [
            [0, 0, 1, -0.42415, 0, 0],
            [1, 0, 0, 0, -0.2, 0.42415],
            [1, 0, 0, 0, 0, 0.37415],
            [1, 0, 0, 0, 0, 0.17415],
            [0, 1, 0, 0, 0, 0],
        ]
    )

    # As in func. FK_pox, the convention that we used to calculate on paper
    # negates the direction of the z-axis of joints 2, 3, and 4 compared to the
    # convention on the robot. Here we want to negate the screw vectors instead
    # of the joint angles (no justification given, empirically this works).
    for i in range(1, 4):
        screw_vectors[i] = -screw_vectors[i]

    # exponentiated S matrices for each of the skew vectors
    M = []
    for i in range(5):
        S = vec6_to_s_matrix(screw_vectors[i])
        M.append(scipy.linalg.expm(S * joint_angles[i]))

    return np.vstack(
        (
            AdjMatrix(M[4] @ M[3] @ M[2] @ M[1]) @ screw_vectors[0],
            AdjMatrix(M[4] @ M[3] @ M[2]) @ screw_vectors[1],
            AdjMatrix(M[4] @ M[3]) @ screw_vectors[2],
            AdjMatrix(M[4]) @ screw_vectors[3],
            screw_vectors[4],
        )
    ).T


def FindRootsNewton(
    joint_angles: list[float],
    T_world_desired: npt.NDArray[np.float64],
    m_mat: npt.NDArray[np.float64],
    s_lst: list[npt.NDArray[np.float64]],
) -> list[float]:
    # TODO: parameterize these?
    MAX_ITERS = 30
    MAX_ROT_ERROR = 1e-2
    MAX_LIN_ERROR = 1e-3

    joint_angles: npt.NDArray[np.float64] = np.array(joint_angles)

    T_ee_world = np.linalg.inv(FK_pox(joint_angles, m_mat, s_lst))
    skewed_twist_ee_desired = scipy.linalg.logm(T_ee_world @ T_world_desired)
    twist_ee_desired = np.concatenate(from_s_matrix(skewed_twist_ee_desired))

    it = 0
    while (
        np.linalg.norm(twist_ee_desired[0:3]) > MAX_ROT_ERROR
        or np.linalg.norm(twist_ee_desired[3:]) > MAX_LIN_ERROR
    ) and it < MAX_ITERS:
        jacobian = GetNewtonJacobian(joint_angles)
        d_theta = (np.linalg.pinv(jacobian) @ twist_ee_desired).squeeze()
        joint_angles += d_theta

        T_ee_world = np.linalg.inv(FK_pox(joint_angles, m_mat, s_lst))
        skewed_twist_ee_desired = scipy.linalg.logm(T_ee_world @ T_world_desired)
        twist_ee_desired = np.concatenate(from_s_matrix(skewed_twist_ee_desired))

        it += 1

    return joint_angles.tolist()  # type: ignore


def IK_numerical_pox(
    T_world_desired: npt.NDArray[np.float64],
    m_mat: npt.NDArray[np.float64],
    s_lst: list[npt.NDArray[np.float64]],
) -> list[float]:
    x = T_world_desired[0, 3]
    y = T_world_desired[1, 3]

    initial_joint_angles = [-np.arctan2(x, y), 0.0, 0.0, 0.0, -np.arctan2(x, y)]
    new_joint_angles = FindRootsNewton(
        initial_joint_angles, T_world_desired, m_mat, s_lst
    )

    min_joint_angles = np.array(
        [-7 * np.pi / 6, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi]
    )
    max_joint_angles = np.array([7 * np.pi / 6, np.pi / 2, np.pi / 2, np.pi / 2, np.pi])

    if (
        (min_joint_angles < new_joint_angles) & (new_joint_angles < max_joint_angles)
    ).all() and sum(new_joint_angles[1:4]) < np.pi / 2 + 0.1:
        return new_joint_angles
    else:
        return [0.0] * 5


if __name__ == "__main__":
    # proof-of-concept test
    M = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.35415],
            [0.0, 0.0, 1.0, 0.30391],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    S = [
        [0, 0, 1, -0.07, 0, 0],
        [1, 0, 0, 0, 0.10319, 0.07],
        [1, 0, 0, 0, 0.30319, 0.02],
        [1, 0, 0, 0, 0.30319, -0.18],
        [0, 1, 0, -0.30319, 0, 0],
    ]

    S = list(map(np.array, S))

    joint_angles = np.arange(1, 6, dtype=np.float64) * 0.1
    T_world_ee = FK_pox(joint_angles, M, S)

    ik_joint_angles = IK_numerical_pox(T_world_ee, M, S)

    print("expected:", joint_angles)
    print("calculated:", ik_joint_angles)

    print("expected:\n")
    print(T_world_ee)
    print()
    print("calculated:")
    print(FK_pox(ik_joint_angles, M, S))
