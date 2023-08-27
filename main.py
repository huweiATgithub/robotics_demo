from robotics_demo.configs import (
    Q0,
    IIWA14,
    IIWA14_ALPHA,
    Q_GOAL,
    NETWORK_PATH,
)
from robotics_demo.demo import DemoNetworkController


def main():
    robot = IIWA14
    q0 = Q0
    model_path = NETWORK_PATH / "demo_good.pth"

    demo = DemoNetworkController(
        robot,
        model_path=model_path,
        lower_limits=q0 - 0.1,
        upper_limits=q0 + 0.1,
        total_steps=800,
        dt=0.001,
        target_robot=IIWA14_ALPHA,
        target_q=Q_GOAL,
        port=7000,
    )
    demo.run()


if __name__ == "__main__":
    main()
