from ALKBR import BroomDetector

detector = BroomDetector(
    BROOM_ABSENCE_THRESHOLD=10,
    BROOM_TOUCH_THRESHOLD=0,
    PERCENTAGE_GREEN_THRESHOLD=50,
    camera_name="10.5.0.182",
    # new_size=(1280, 720),
)

detector.run()
