from ALKBR import BroomDetector

detector = BroomDetector(
    BROOM_ABSENCE_THRESHOLD=30,
    BROOM_TOUCH_THRESHOLD=0,
    PERCENTAGE_GREEN_THRESHOLD=50,
    camera_name="10.5.0.195",
    new_size=(480, 270),
)

detector.run()
