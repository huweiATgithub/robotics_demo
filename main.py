from robotics_demo.configs import Q0, IIWA14, IIWA14_ALPHA, Q_GOAL
from robotics_demo.demo import Demo


def main():
    robot = IIWA14
    q0 = Q0

    demo = Demo(
        robot,
        lower_limits=q0 - 0.1,
        upper_limits=q0 + 0.1,
        total_steps=800,
        dt=0.001,
        target_robot=IIWA14_ALPHA,
        target_q=Q_GOAL,
    )

    demo.run()


if __name__ == "__main__":
    main()
