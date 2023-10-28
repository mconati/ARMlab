#!/usr/bin/env python3

"""!
Class to represent the camera.
"""

import time
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import numpy as np
import numpy.typing as npt
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from cv_bridge import CvBridge, CvBridgeError

# local imports
import transformations


class Camera:
    """!
    @brief      This class describes a camera.
    """

    def __init__(self) -> None:
        """!
        @brief      Constructs a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720, 1280, 3)).astype(np.uint8)

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix: Optional[npt.NDArray[np.float64]] = None
        self.extrinsic_matrix: Optional[npt.NDArray[np.float64]] = None
        self.homography_matrix: Optional[npt.NDArray[np.float64]] = None
        self.depth_correction: npt.NDArray[np.uint16] = np.zeros(720, dtype=np.uint16)
        """Depth correction in millimeters for each row of the registered depth
        image."""

        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))  # type: ignore
        self.tag_detections: Optional[AprilTagDetectionArray] = None
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def processVideoFrame(self) -> None:
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(
            self.VideoFrame, self.block_contours.tolist(), -1, (255, 0, 255), 3
        )

    def ColorizeDepthFrame(self) -> None:
        """!
        @brief Converts frame to colormapped formats in HSV and RGB
        """

        def get_delta_z(y: int) -> np.uint16:
            delta_z = 77.25988700564972 - 0.2128060263653484 * y
            return np.uint16(delta_z)

        corrected_depth_frame = np.copy(self.DepthFrameRaw)

        for y in range(corrected_depth_frame.shape[0]):
            corrected_depth_frame[y] += get_delta_z(y)

        # The mask of the points outside of the board
        binary_mask = np.ones(corrected_depth_frame.shape, dtype=np.bool_)
        binary_mask[140:620, 280:1100] = False
        binary_mask[370:630, 600:800] = True

        corrected_depth_frame[binary_mask] = 1000

        # The board should be around 1000 after correction

        ret, thresh1 = cv2.threshold(
            corrected_depth_frame.astype(np.float32), 990, 255, cv2.THRESH_BINARY
        )
        thresh1 = thresh1.astype(np.uint8)

        self.DepthFrameRGB = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB)

        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(self.DepthFrameRGB, contours, -1, (0, 255, 0), 3)

        # self.DepthFrameRGB = np.copy(corrected_depth_frame)
        # self.DepthFrameRGB = np.clip(self.DepthFrameRGB, 1000, 1000 + 255)
        # self.DepthFrameRGB -= 1000
        # self.DepthFrameRGB = self.DepthFrameRGB.astype(np.uint8)
        # # self.DepthFrameRGB = np.vstack((self.DepthFrameRGB,) * 3)
        # self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameRGB, cv2.COLOR_GRAY2RGB)

        # self.DepthFrameHSV[..., 0] = corrected_depth_frame >> 1
        # self.DepthFrameHSV[..., 1] = 0xFF
        # self.DepthFrameHSV[..., 2] = 0x9F
        # self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV, cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self) -> None:
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
        )

    def loadDepthFrame(self) -> None:
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png", 0).astype(np.uint16)

    def convertQtVideoFrame(self) -> Optional[QImage]:
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(
                frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888
            )
            return img
        except BaseException:
            return None

    def convertQtGridFrame(self) -> Optional[QImage]:
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(
                frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888
            )
            return img
        except BaseException:
            return None

    def convertQtDepthFrame(self) -> Optional[QImage]:
        """!
        @brief      Converts colormaped depth frame to format suitable for Qt

        @return     QImage
        """
        try:
            img = QImage(
                self.DepthFrameRGB.tobytes(),
                self.DepthFrameRGB.shape[1],
                self.DepthFrameRGB.shape[0],
                QImage.Format.Format_RGB888,
            )
            return img
        except BaseException:
            return None

    def convertQtTagImageFrame(self) -> Optional[QImage]:
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(
                frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888
            )
            return img
        except BaseException:
            return None

    def getAffineTransform(
        self, coord1: npt.NDArray[np.float64], coord2: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)  # type: ignore

    def loadCameraCalibration(self, file: str) -> None:
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        # no implementation; the CameraInfoListener will set the intrinsic
        # matrix to the factory calibration
        pass

    def calibrateExtrinsics(self) -> bool:
        """Returns whether calibration succeeded or failed"""
        if self.intrinsic_matrix is None or self.tag_detections is None:
            return False

        new_extrinsic_matrix = transformations.compute_extrinsic_matrix(
            self.intrinsic_matrix, self.tag_detections
        )
        if new_extrinsic_matrix is None:
            return False

        self.extrinsic_matrix = new_extrinsic_matrix
        self.homography_matrix = transformations.compute_homography_matrix(
            self.tag_detections
        )

        # copied from control_station
        # TODO(elvin): fix this
        def pixelToWorld(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
            # precondition: self.camera.extrinsic_matrix is valid
            K = self.intrinsic_matrix

            homogenous_pixels = np.array([x, y, 1]).reshape((3, 1))
            P_camera = z * np.linalg.inv(K) @ homogenous_pixels  # type: ignore

            T_world_camera = np.linalg.inv(self.extrinsic_matrix)  # type: ignore

            P_world = transformations.apply_affine3d(T_world_camera, P_camera)
            return P_world.squeeze()

        # Calibrate the depth correction
        def find_tag(id: int) -> tuple[int, int]:
            if self.tag_detections is None:
                return (0, 0)
            tag_detection: AprilTagDetection
            for tag_detection in self.tag_detections.detections:
                if tag_detection.id == id:
                    center = tag_detection.centre
                    return (int(center.x), int(center.y))
            return (0, 0)

        # TODO(elvin): fix this if the tags are not detectable
        # these are represented as [x y]
        tag1_detection = find_tag(1)
        tag4_detection = find_tag(4)

        tag1_z = self.DepthFrameRaw[tag1_detection[1], tag1_detection[0]]
        tag4_z = self.DepthFrameRaw[tag4_detection[1], tag4_detection[0]]

        # We want the z coordinate of all points on the board to be zero, so the
        # error is simply the z value that we get from pixelToWorld.
        tag1_error_mm = pixelToWorld(*tag1_detection, tag1_z / 1000)[2] * 1000
        tag4_error_mm = pixelToWorld(*tag4_detection, tag4_z / 1000)[2] * 1000

        tag1_correction_mm = -tag1_error_mm
        tag4_correction_mm = -tag4_error_mm

        # Linear interpolation between the errors of the two tags. We will
        # assume that the correction is independent of the x coordinate (which
        # may not necessarily be true but works reasonably well enough).
        slope = (tag4_correction_mm - tag1_correction_mm) / (
            tag4_detection[1] - tag1_detection[1]
        )
        intercept = tag1_correction_mm - slope * tag1_detection[1]

        self.depth_correction = (slope * np.arange(720) + intercept).astype(np.uint16)

        return True

    def blockDetector(self) -> None:
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self) -> None:
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self) -> None:
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matrices to project the gridpoints
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        if self.extrinsic_matrix is None or self.intrinsic_matrix is None:
            return

        self.GridFrame = np.copy(self.VideoFrame)

        # 14 latitudinal lines
        # 21 longitudinal lines

        # indices in the world frame (+y goes up)
        meridian = 10
        equator = 5

        for y_idx in range(14):
            for x_idx in range(21):
                world_x = (x_idx - meridian) * 0.05
                world_y = (y_idx - equator) * 0.05
                world_z = 0

                P_world = np.array([world_x, world_y, world_z])

                P_camera = transformations.apply_affine3d(
                    self.extrinsic_matrix, P_world
                )

                pixel_coordinate = self.intrinsic_matrix @ P_camera
                pixel_coordinate /= pixel_coordinate[2]
                pixel_coordinate = pixel_coordinate[:2]

                center = pixel_coordinate.astype(int).squeeze().tolist()
                # print(center)

                self.GridFrame = cv2.circle(self.GridFrame, center, 5, (255, 0, 0))

        # if self.camera.homography_matrix is not None:
        #     self.camera.GridFrame = cv2.warpPerspective(
        #         cv_image, self.camera.homography_matrix, (1500, 900)
        #     )

        pass

    def drawTagsInRGBImage(self, msg: AprilTagDetectionArray) -> None:
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        detection: AprilTagDetection
        # Write your code here
        for detection in msg.detections:
            centre_coord = [detection.centre.x, detection.centre.y]
            centre_coord = [int(x) for x in centre_coord]
            modified_image = cv2.drawMarker(
                modified_image,
                centre_coord,
                color=(124, 252, 0),
                markerType=cv2.MARKER_CROSS,
            )
            for i in range(4):
                pt1 = [detection.corners[i - 1].x, detection.corners[i - 1].y]
                pt2 = [detection.corners[i].x, detection.corners[i].y]

                pt1 = [int(x) for x in pt1]
                pt2 = [int(x) for x in pt2]

                modified_image = cv2.line(modified_image, pt1, pt2, color=(124, 252, 0))
            modified_image = cv2.putText(
                modified_image,
                f"ID: {detection.id}",
                pt2,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(124, 252, 0),
            )

        self.TagImageFrame = modified_image


class ImageListener(Node):
    def __init__(self, topic: str, camera: Camera) -> None:
        super().__init__("image_listener")
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic: str, camera: Camera):
        super().__init__("tag_detection_listener")
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray, topic, self.callback, 10
        )
        self.camera = camera

    def callback(self, msg: AprilTagDetectionArray) -> None:
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic: str, camera: Camera) -> None:
        super().__init__("camera_info_listener")
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data: CameraInfo) -> None:
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic: str, camera: Camera) -> None:
        super().__init__("depth_listener")
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data: Image) -> None:
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)

        self.camera.DepthFrameRaw = cv_depth
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera: Camera, parent: Optional[QObject] = None) -> None:
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic, self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic, self.camera)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self) -> None:
        if __name__ == "__main__":
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if (rgb_frame is not None) and (depth_frame is not None):
                    self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once()  # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == "__main__":
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imshow(
                        "Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR),
                    )
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass

        self.executor.shutdown()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
