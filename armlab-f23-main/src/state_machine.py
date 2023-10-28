"""!
The state machine that implements the logic.
"""
from typing import Optional
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer, QObject
import time
import numpy as np
import numpy.typing as npt
import rclpy

from rxarm import RXArm
from camera import Camera
import transformations
import kinematics


class RobotAction:
    ...


class OpenGripper(RobotAction):
    ...


class CloseGripper(RobotAction):
    ...


class GoToWaypoint(RobotAction):
    def __init__(self, waypoints: list[float]):
        self.waypoints = waypoints


class StateMachine:
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm: RXArm, camera: Camera) -> None:
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message: str = "State: Idle"
        self.current_state: str = "idle"
        self.next_state: str = "idle"
        # self.waypoints = [
        #     [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
        #     [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
        #     [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
        #     [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
        #     [0.0,             0.0,       0.0,          0.0,        0.0],
        #     [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
        #     [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
        #     [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
        #     [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
        #     [0.0,             0.0,       0.0,          0.0,        0.0]]

        # overwrite
        # self.waypoints = [
        #     [-0.6427379846572876, 0.2316311001777649, -0.1288543939590454, 1.4879614114761353, -0.5783107876777649],
        #     [-0.62126225233078, 0.32827189564704895, 0.17180585861206055, 1.055378794670105, -0.5890486240386963],
        #     [-0.03528155758976936, -0.06289321184158325, 0.2791845202445984, 1.3775148391723633, -0.08130098134279251],
        #     [-0.03528155758976936, -1.3775148391723633, 0.9633399844169617, 0.5322913527488708, 0.023009711876511574],
        # ]

        T_world_ee = np.identity(4)
        T_world_ee[:3, 3] = 0.0, 0.4, 0.1
        self.actions: list[RobotAction] = [
            GoToWaypoint(kinematics.IK_numerical_pox(T_world_ee, self.rxarm.M_matrix, self.rxarm.S_list))
        ]
        print(kinematics.IK_numerical_pox(T_world_ee, self.rxarm.M_matrix, self.rxarm.S_list))

    def add_waypoint(self) -> None:
        self.actions.append(GoToWaypoint(self.rxarm.get_positions()))

    def add_close_gripper_action(self) -> None:
        self.actions.append(CloseGripper())

    def add_open_gripper_action(self) -> None:
        self.actions.append(OpenGripper())

    def clear_actions(self) -> None:
        self.actions.clear()

    def export_actions(self) -> None:
        filename = "/tmp/actions.csv"

        with open(filename, "w") as f:
            for action in self.actions:
                if type(action) is OpenGripper:
                    f.write("open\n")
                elif type(action) is CloseGripper:
                    f.write("close\n")
                elif type(action) is GoToWaypoint:
                    tokens = [f"{position:.6f}" for position in action.waypoints]
                    line = ",".join(tokens) + "\n"
                    f.write(line)

    def set_next_state(self, state: str) -> None:
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self) -> None:
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

    """Functions run for each state"""

    def manual(self) -> None:
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self) -> None:
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self) -> None:
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self) -> None:
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """

        # TODO: is this the correct way to handle estop?
        if self.current_state == "estop":
            return

        self.status_message = "State: Execute - Executing motion plan"

        for action in self.actions:
            if self.current_state == "estop":
                break

            if type(action) is CloseGripper:
                self.rxarm.gripper.grasp()
            elif type(action) is OpenGripper:
                self.rxarm.gripper.release()
            elif type(action) is GoToWaypoint:
                self.rxarm.set_positions(action.waypoints)

            time.sleep(3)

        self.next_state = "idle"

    def calibrate(self) -> None:
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        if self.camera.calibrateExtrinsics():
            self.status_message = "Calibration - Completed Calibration"
        else:
            self.status_message = "Calibration Failed - Bad AprilTag Detections"

    """ TODO """

    def detect(self) -> None:
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self) -> None:
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print("Failed to initialize the rxarm")
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """

    updateStatusMessage = pyqtSignal(str)

    def __init__(
        self, state_machine: StateMachine, parent: Optional[QObject] = None
    ) -> None:
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm = state_machine

    def run(self) -> None:
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)
