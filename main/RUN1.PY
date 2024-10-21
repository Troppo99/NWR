from ALKBR import BroomDetector

detector = BroomDetector(
    BROOM_ABSENCE_THRESHOLD=15,
    BROOM_TOUCH_THRESHOLD=5,
    PERCENTAGE_GREEN_THRESHOLD=60,
    camera_name="10.5.0.182",
)

detector.run()
