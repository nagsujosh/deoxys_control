import argparse
import time

import numpy as np

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="config/charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    device = None
    robot_interface = None
    controller_cfg = get_default_controller_config(controller_type=args.controller_type)
    try:
        logger.info(
            "Starting SpaceMouse teleoperation with controller=%s interface_cfg=%s",
            args.controller_type,
            args.interface_cfg,
        )
        device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
        device.start_control()

        robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)

        controller_type = args.controller_type
        # controller_cfg = YamlConfig("config/osc-pose-controller.yml").as_easydict()

        robot_interface._state_buffer = []

        for i in range(3000):
            start_time = time.time_ns()

            action, grasp = input2action(
                device=device,
                controller_type=controller_type,
            )
            if action is None:
                logger.info("Received SpaceMouse stop signal, stopping teleoperation cleanly")
                break

            action = np.asarray(action, dtype=np.float64)
            if action.ndim != 1 or action.shape[0] < 7:
                logger.warning(f"Skipping malformed SpaceMouse action with shape {action.shape}")
                continue

            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            end_time = time.time_ns()
            logger.debug(f"Time duration: {((end_time - start_time) / (10**9))}")
    except KeyboardInterrupt:
        logger.info("Stopping SpaceMouse teleoperation on keyboard interrupt")
    finally:
        if robot_interface is not None:
            try:
                robot_interface.control(
                    controller_type=args.controller_type,
                    action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
                    controller_cfg=controller_cfg,
                    termination=True,
                )
            except Exception as exc:
                logger.warning(f"Failed to send teleop termination command: {exc}")
            finally:
                robot_interface.close()
                logger.info("Closed Franka interface")

            # Check if there is any state frame missing
            for (state, next_state) in zip(
                robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]
            ):
                if (next_state.frame - state.frame) > 1:
                    print(state.frame, next_state.frame)
        if device is not None:
            device.close()
            logger.info("Closed SpaceMouse device")


if __name__ == "__main__":
    main()
