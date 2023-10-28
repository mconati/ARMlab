from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from apriltag_msgs.msg import AprilTagDetectionArray, AprilTagDetection


class AprilTagCorrespondence:
    def __init__(self) -> None:
        self.world_coordinate: npt.NDArray[np.float64] = np.empty(0)
        self.image_coordinate: npt.NDArray[np.float64] = np.empty(0)

    def __bool__(self) -> bool:
        return len(self.world_coordinate) != 0 and len(self.image_coordinate) != 0


def get_apriltag_correspondences(
    msg: AprilTagDetectionArray,
) -> dict[int, AprilTagCorrespondence]:
    # mapping of tag ID numbers to correspondence objects
    tag_correspondences: dict[int, AprilTagCorrespondence] = defaultdict(
        AprilTagCorrespondence
    )

    # Set the known locations of centers of the tags we expect to be on the
    # board in the world frame.
    tag_correspondences[1].world_coordinate = np.array([-0.25, -0.1, 0])
    tag_correspondences[2].world_coordinate = np.array([0.25, -0.1, 0])
    tag_correspondences[3].world_coordinate = np.array([0.25, 0.2, 0])
    tag_correspondences[4].world_coordinate = np.array([-0.25, 0.2, 0])

    # Read the pixel coordinates of the center of each tag detection.
    detection: AprilTagDetection
    for detection in msg.detections:
        tag_id = detection.id
        image_coordinate = np.array([detection.centre.x, detection.centre.y])
        tag_correspondences[tag_id].image_coordinate = image_coordinate

    # Remove correspondence objects that do not have values for both coordinates.
    tag_correspondences = {k: v for k, v in tag_correspondences.items() if v}

    return tag_correspondences


def get_manual_T_world_camera() -> npt.NDArray[np.float64]:
    roll = np.deg2rad(180 - 13)

    T_world_camera = np.identity(4)
    # set the rotation component of the transformation
    T_world_camera[:3, :3] = Rotation.from_euler("X", roll).as_matrix()

    # set the translation component of the transformation
    T_world_camera[:3, 3] = (-0.05, 0.293, 0.98)

    return T_world_camera


# Factory intrinsic matrix for reference (Station 3)
# def get_intrinsic_matrix() -> npt.NDArray[np.float64]:
#     return np.array(
#         [
#             [896.861083984375, 0.0, 660.5230712890625],
#             [0.0, 897.2031860351562, 381.4194030761719],
#             [0.0, 0.0, 1.0],
#         ]
#     )


def compute_extrinsic_matrix(
    intrinsic_matrix: npt.NDArray[np.float64],
    apriltag_detections: AprilTagDetectionArray,
) -> Optional[npt.NDArray[np.float64]]:
    # mapping of tag ID numbers to correspondence objects
    tag_correspondences = get_apriltag_correspondences(apriltag_detections)
    if len(tag_correspondences) < 4:
        return None

    world_coordinates = np.array(
        list(
            correspondence.world_coordinate
            for correspondence in tag_correspondences.values()
        )
    )

    image_coordinates = np.array(
        list(
            correspondence.image_coordinate
            for correspondence in tag_correspondences.values()
        )
    )

    distortion_coeffs = np.zeros(5, dtype=np.float64)
    # distortion_coeffs = np.array(
    #     [
    #         0.1452116072177887,
    #         -0.4892308712005615,
    #         -0.0012472006492316723,
    #         -0.0003476899000816047,
    #         0.45278146862983704,
    #     ]
    # )

    success, rvec, tvec = cv2.solvePnP(
        world_coordinates,
        image_coordinates,
        intrinsic_matrix,
        distortion_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success:
        print("SOLVEPNP DID NOT FIND A SOLUTION", flush=True)

    T_camera_world = np.identity(4)
    cv2.Rodrigues(rvec, T_camera_world[:3, :3])
    T_camera_world[:3, 3] = tvec.squeeze()

    return T_camera_world


def compute_homography_matrix(
    msg: AprilTagDetectionArray,
) -> Optional[npt.NDArray[np.float64]]:
    PIXELS_PER_METER = 1000

    # mapping of tag ID numbers to correspondence objects
    tag_correspondences = get_apriltag_correspondences(msg)
    if len(tag_correspondences) < 4:
        return None

    dst_coords = np.array(
        list(
            correspondence.world_coordinate[:2]  # we only want x and y here
            for correspondence in tag_correspondences.values()
        ),
        dtype=np.float32,  # opencv wants f32
    )

    src_coords = np.array(
        list(
            correspondence.image_coordinate
            for correspondence in tag_correspondences.values()
        ),
        dtype=np.float32,  # opencv wants f32
    )

    center_coord = np.array([0, 0.075], dtype=np.float32)
    dst_coords -= center_coord
    dst_coords *= PIXELS_PER_METER

    # The world y axis and the image y axis are inverted
    dst_coords[:, 1] = -dst_coords[:, 1]

    # TODO: find better output image dimensions
    output_width = 1500
    output_height = 900

    dst_coords[:, 0] += output_width // 2
    dst_coords[:, 1] += output_height // 2

    homography_mat = cv2.getPerspectiveTransform(src_coords, dst_coords)

    return homography_mat  # type: ignore


def apply_affine3d(
    transform: npt.NDArray[np.float64], point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    if point.ndim == 1:
        point = point[:, np.newaxis]

    # convert the point from affine coordinates to homogeneous coordinates
    point = np.vstack((point, np.array([[1]])))

    # transform the point
    point = transform @ point

    # convert the point from homogenous coordinates to affine coordinates
    return point[:3]  # type: ignore
