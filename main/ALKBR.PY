import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import cvzone
import pymysql
from datetime import datetime, timedelta
import threading
import queue
import math


class BroomDetector:
    def __init__(
        self,
        BROOM_ABSENCE_THRESHOLD=10,
        BROOM_TOUCH_THRESHOLD=0,
        BROOM_PERCENTAGE_GREEN_THRESHOLD=50,
        BROOM_CONFIDENCE_THRESHOLD=0.9,
        broom_model="broom5l_11.pt",
        camera_name=None,
        new_size=(960, 540),
        rtsp_url=None,
        window_size=(540, 360),
    ):
        self.BROOM_CONFIDENCE_THRESHOLD = BROOM_CONFIDENCE_THRESHOLD
        self.BROOM_ABSENCE_THRESHOLD = BROOM_ABSENCE_THRESHOLD
        self.BROOM_TOUCH_THRESHOLD = BROOM_TOUCH_THRESHOLD
        self.BROOM_PERCENTAGE_GREEN_THRESHOLD = BROOM_PERCENTAGE_GREEN_THRESHOLD
        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = new_size
        self.scale_x = self.new_width / 1280
        self.scale_y = self.new_height / 720
        self.scaled_borders = []
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.broom_absence_timer_start = None
        self.prev_frame_time = time.time()
        self.fps = 0
        self.first_green_time = None
        self.is_counting = False
        self.camera_name = camera_name
        self.borders, self.ip_camera, self.idx = self.camera_config()

        # Adjust rtsp_url handling
        if rtsp_url is not None:
            self.rtsp_url = rtsp_url  # Keep the provided rtsp_url
            if os.path.isfile(rtsp_url):
                # It's a local file
                self.is_local_file = True
                cap = cv2.VideoCapture(self.rtsp_url)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25  # Default FPS if unable to get
                cap.release()
                print(f"Local video file detected. FPS: {self.video_fps}")
            else:
                # Assume it's an RTSP stream
                self.is_local_file = False
                print(f"RTSP stream detected. URL: {self.rtsp_url}")
                self.video_fps = None
        else:
            # rtsp_url is None
            self.rtsp_url = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
            self.video_fps = None
            self.is_local_file = False

        self.show_text = True
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        for border in self.borders:
            scaled_border = []
            for x, y in border:
                scaled_x = int(x * self.scale_x)
                scaled_y = int(y * self.scale_y)
                scaled_border.append((scaled_x, scaled_y))
            self.scaled_borders.append(scaled_border)

        self.border_states = {
            idx: {
                "broom_time": None,
                "is_green": False,
                "broom_overlap_time": 0.0,
                "last_broom_overlap_time": None,
            }
            for idx in range(len(self.borders))
        }
        self.borders_pts = [np.array(border, np.int32) for border in self.scaled_borders]
        self.broom_model = YOLO(broom_model).to("cuda")
        self.broom_model.overrides["verbose"] = False
        print(f"Model Broom device: {next(self.broom_model.model.parameters()).device}")

    def camera_config(self):
        config = {
            # Line Cameras
            "SEWING6": {
                "borders": [
                    [(626, 234), (702, 234), (714, 304), (616, 304)],
                    [(616, 304), (714, 304), (725, 361), (610, 360)],
                    [(610, 360), (725, 361), (735, 419), (603, 419)],
                    [(603, 419), (735, 419), (745, 485), (595, 485)],
                    [(595, 485), (745, 485), (760, 570), (584, 573)],
                    [(584, 573), (527, 574), (519, 620), (529, 624), (509, 714), (611, 716), (621, 571), (584, 573)],
                    [(760, 570), (844, 568), (866, 718), (747, 717), (741, 574), (760, 570)],
                    [(621, 571), (741, 574), (747, 717), (611, 716)],
                ],
                "ip": "10.5.0.159",
            },
            "SEWING4": {
                "borders": [
                    [(589, 9), (629, 10), (630, 43), (582, 44)],
                    [(669, 8), (629, 10), (630, 43), (677, 44)],
                    [(582, 44), (630, 43), (630, 73), (575, 71)],
                    [(677, 44), (630, 43), (630, 73), (682, 71)],
                    [(575, 71), (630, 73), (628, 106), (570, 104)],
                    [(682, 71), (630, 73), (628, 106), (687, 104)],
                    [(570, 104), (628, 106), (628, 144), (559, 143)],
                    [(687, 104), (628, 106), (628, 144), (690, 144)],
                    [(559, 143), (628, 144), (626, 185), (553, 185)],
                    [(690, 144), (628, 144), (626, 185), (697, 184)],
                    [(553, 185), (626, 185), (625, 231), (544, 231)],
                    [(697, 184), (626, 185), (625, 231), (709, 230)],
                    [(544, 231), (625, 231), (623, 288), (531, 291)],
                    [(709, 230), (625, 231), (623, 288), (725, 290)],
                    [(531, 291), (623, 288), (622, 342), (522, 342)],
                    [(725, 290), (623, 288), (622, 342), (729, 340)],
                    [(522, 342), (622, 342), (620, 400), (511, 402)],
                    [(729, 340), (622, 342), (620, 400), (737, 398)],
                    [(511, 402), (620, 400), (617, 470), (500, 471)],
                    [(737, 398), (620, 400), (617, 470), (745, 468)],
                    [(500, 471), (617, 470), (619, 553), (490, 554)],
                    [(745, 468), (617, 470), (619, 553), (761, 553)],
                    [(490, 554), (619, 553), (618, 624), (481, 630)],
                    [(761, 553), (619, 553), (618, 624), (769, 625)],
                    [(481, 630), (618, 624), (617, 713), (474, 713)],
                    [(769, 625), (618, 624), (617, 713), (774, 713)],
                ],
                "ip": "10.5.0.146",
            },
            "SEWINGBACK1": {
                "borders": [
                    [(728, 237), (767, 236), (776, 269), (727, 269)],
                    [(798, 238), (767, 236), (776, 269), (815, 272)],
                    [(727, 269), (776, 269), (784, 303), (727, 303)],
                    [(815, 272), (776, 269), (784, 303), (831, 306)],
                    [(727, 303), (784, 303), (790, 340), (724, 339)],
                    [(831, 306), (784, 303), (790, 340), (844, 343)],
                    [(724, 339), (790, 340), (799, 377), (724, 375)],
                    [(844, 343), (790, 340), (799, 377), (863, 380)],
                    [(724, 375), (799, 377), (805, 419), (723, 418)],
                    [(863, 380), (799, 377), (805, 419), (876, 423)],
                    [(723, 418), (805, 419), (816, 469), (721, 464)],
                    [(876, 423), (805, 419), (816, 469), (896, 472)],
                    [(721, 464), (816, 469), (827, 532), (717, 534)],
                    [(896, 472), (816, 469), (827, 532), (914, 536)],
                    [(717, 534), (827, 532), (839, 594), (714, 595)],
                    [(914, 536), (827, 532), (839, 594), (936, 596)],
                    [(714, 595), (839, 594), (856, 708), (706, 708)],
                    [(936, 596), (839, 594), (856, 708), (972, 708)],
                    [(714, 595), (712, 621), (641, 616), (631, 710), (706, 708)],
                    [(641, 616), (631, 710), (543, 711), (569, 609)],
                    [(569, 609), (543, 711), (507, 711), (504, 660), (457, 650), (481, 599)],
                    [(481, 599), (457, 650), (359, 631), (384, 580)],
                    [(384, 580), (359, 631), (286, 613), (239, 681), (194, 667), (273, 567)],
                    [(273, 567), (194, 667), (131, 645), (217, 549)],
                    [(217, 549), (131, 645), (89, 630), (172, 539)],
                ],
                "ip": "10.5.0.183",
            },
            "SEWINGBACK2": {
                "borders": [
                    [(675, 206), (732, 206), (733, 241), (670, 241)],
                    [(794, 205), (732, 206), (733, 241), (805, 243)],
                    [(805, 243), (822, 292), (737, 293), (733, 241)],
                    [(670, 241), (733, 241), (737, 293), (654, 292)],
                    [(654, 292), (737, 293), (738, 334), (641, 334)],
                    [(822, 292), (737, 293), (738, 334), (838, 337)],
                    [(641, 334), (738, 334), (739, 380), (632, 376)],
                    [(838, 337), (738, 334), (739, 380), (852, 378)],
                    [(632, 376), (739, 380), (741, 435), (615, 436)],
                    [(852, 378), (739, 380), (741, 435), (870, 430)],
                    [(870, 430), (741, 435), (741, 501), (891, 495)],
                    [(615, 436), (741, 435), (741, 501), (597, 500)],
                    [(597, 500), (741, 501), (740, 564), (579, 563)],
                    [(891, 495), (741, 501), (740, 564), (907, 555)],
                    [(579, 563), (740, 564), (737, 629), (565, 628)],
                    [(907, 555), (740, 564), (737, 629), (925, 619)],
                    [(565, 628), (737, 629), (737, 711), (538, 711)],
                    [(925, 619), (737, 629), (737, 711), (949, 708)],
                ],
                "ip": "10.5.0.195",
            },
            "SEWING1": {
                "borders": [
                    [(608, 12), (665, 12), (662, 46), (592, 47)],
                    [(665, 12), (726, 13), (731, 48), (662, 46)],
                    [(777, 10), (726, 13), (731, 48), (794, 49)],
                    [(592, 47), (662, 46), (658, 79), (579, 81)],
                    [(731, 48), (662, 46), (658, 79), (735, 79)],
                    [(731, 48), (735, 79), (806, 83), (794, 49)],
                    [(579, 81), (658, 79), (654, 120), (568, 118)],
                    [(654, 120), (658, 79), (735, 79), (740, 121)],
                    [(806, 83), (735, 79), (740, 121), (822, 121)],
                    [(568, 118), (654, 120), (651, 166), (572, 163)],
                    [(651, 166), (654, 120), (740, 121), (743, 165)],
                    [(822, 121), (740, 121), (743, 165), (842, 165)],
                    [(572, 163), (651, 166), (645, 223), (553, 226)],
                    [(645, 223), (651, 166), (743, 165), (747, 223)],
                    [(842, 165), (743, 165), (747, 223), (863, 227)],
                    [(553, 226), (645, 223), (639, 297), (506, 296), (524, 243)],
                    [(639, 297), (645, 223), (747, 223), (752, 296)],
                    [(752, 296), (747, 223), (863, 227), (886, 299)],
                    [(506, 296), (639, 297), (634, 372), (481, 373)],
                    [(634, 372), (639, 297), (752, 296), (758, 375)],
                    [(758, 375), (752, 296), (886, 299), (909, 380)],
                    [(481, 373), (634, 372), (625, 461), (457, 454)],
                    [(625, 461), (634, 372), (758, 375), (765, 464)],
                    [(765, 464), (758, 375), (909, 380), (933, 463)],
                    [(457, 454), (625, 461), (614, 583), (423, 574)],
                    [(614, 583), (625, 461), (765, 464), (766, 583)],
                    [(933, 463), (765, 464), (766, 583), (966, 575)],
                    [(423, 574), (614, 583), (603, 710), (395, 710)],
                    [(603, 710), (614, 583), (766, 583), (767, 708)],
                    [(767, 708), (766, 583), (966, 575), (989, 706)],
                ],
                "ip": "10.5.0.201",
            },
            "LINEMANUAL14": {
                "borders": [
                    [(631, 149), (723, 149), (734, 193), (623, 193)],
                    [(623, 193), (734, 193), (747, 242), (612, 242)],
                    [(612, 242), (747, 242), (765, 314), (596, 314)],
                    [(596, 314), (765, 314), (783, 386), (583, 386)],
                    [(583, 386), (783, 386), (802, 464), (565, 464)],
                    [(565, 464), (802, 464), (821, 571), (549, 575)],
                    [(549, 575), (821, 571), (848, 714), (529, 715)],
                ],
                "ip": "10.5.0.202",
            },
            "LINEMANUAL10": {
                "borders": [
                    [(560, 250), (623, 250), (628, 279), (555, 281)],
                    [(555, 281), (628, 279), (667, 282), (672, 306), (631, 306), (593, 305), (593, 280), (628, 279)],
                    [(555, 281), (521, 281), (520, 308), (593, 305)],
                    [(549, 308), (545, 338), (639, 343), (631, 306), (593, 305)],
                    [(542, 360), (641, 359), (648, 412), (535, 413)],
                    [(535, 413), (648, 412), (659, 467), (529, 467)],
                    [(529, 467), (659, 467), (670, 532), (521, 532)],
                    [(521, 532), (670, 532), (680, 605), (511, 606)],
                    [(511, 606), (680, 605), (699, 717), (507, 717)],
                ],
                "ip": "10.5.0.205",
            },
            "LINEMANUAL15": {
                "borders": [
                    [(190, 551), (172, 439), (65, 565), (164, 586)],
                    [(65, 565), (65, 590), (34, 638), (60, 675), (32, 718), (87, 719), (101, 691)],
                    [(164, 586), (273, 609), (216, 720), (189, 716), (101, 691)],
                    [(273, 609), (385, 622), (353, 718)],
                    [(385, 622), (497, 629), (479, 718)],
                    [(497, 629), (617, 631), (613, 718)],
                    [(613, 718), (617, 631), (712, 631), (713, 717)],
                    [(712, 631), (816, 625), (832, 690), (826, 720)],
                ],
                "ip": "10.5.0.207",
            },
            "SEWING3": {
                "borders": [
                    [(565, 58), (651, 58), (653, 113), (551, 113)],
                    [(551, 113), (653, 113), (655, 181), (531, 179)],
                    [(531, 179), (655, 181), (659, 259), (512, 252)],
                    [(512, 252), (659, 259), (666, 343), (492, 337)],
                    [(492, 337), (666, 343), (671, 474), (468, 461)],
                    [(468, 461), (402, 457), (380, 563), (561, 578), (570, 468)],
                    [(570, 468), (747, 481), (745, 583), (561, 578)],
                    [(450, 569), (433, 714), (683, 714), (681, 581), (561, 578)],
                ],
                "ip": "10.5.0.214",
            },
            "SEWING8": {
                "borders": [
                    [(630, 199), (699, 196), (709, 239), (627, 241)],
                    [(627, 241), (709, 239), (721, 284), (620, 293)],
                    [(620, 293), (721, 284), (731, 338), (616, 343)],
                    [(616, 343), (731, 338), (745, 396), (613, 401)],
                    [(613, 401), (745, 396), (759, 460), (611, 470)],
                    [(611, 470), (759, 460), (776, 548), (603, 553)],
                    [(603, 553), (594, 714), (812, 716), (776, 548)],
                ],
                "ip": "10.5.0.219",
            },
            "SEWING2": {
                "borders": [
                    [(619, 95), (706, 96), (717, 150), (614, 155)],
                    [(614, 155), (717, 150), (730, 211), (607, 215)],
                    [(607, 215), (730, 211), (746, 285), (601, 291)],
                    [(601, 291), (746, 285), (758, 366), (593, 369)],
                    [(593, 369), (758, 366), (773, 437), (589, 442)],
                    [(589, 442), (773, 437), (784, 503), (581, 508)],
                    [(581, 508), (455, 509), (435, 606), (576, 612)],
                    [(581, 508), (689, 506), (692, 610), (576, 612)],
                    [(689, 506), (784, 503), (850, 498), (866, 593), (692, 610)],
                    [(576, 612), (571, 718), (820, 717), (802, 596), (692, 610)],
                ],
                "ip": "10.5.0.217",
            },
            "SEWING5": {
                "borders": [
                    [(623, 159), (705, 155), (716, 208), (619, 211)],
                    [(619, 211), (716, 208), (728, 270), (611, 276)],
                    [(611, 276), (728, 270), (741, 343), (603, 347)],
                    [(603, 347), (741, 343), (757, 445), (597, 450)],
                    [(597, 450), (475, 457), (468, 614), (584, 617), (597, 450)],
                    [(597, 450), (757, 445), (781, 607), (584, 617)],
                    [(757, 445), (876, 438), (900, 536), (860, 573), (860, 592), (781, 607)],
                    [(584, 617), (781, 607), (795, 715), (581, 718)],
                ],
                "ip": "10.5.0.252",
            },
            # Office Cameras
            "OFFICE1": {
                "borders": [
                    [(688, 98), (737, 100), (739, 137), (684, 136)],
                    [(790, 103), (737, 100), (739, 137), (803, 140)],
                    [(684, 136), (739, 137), (743, 173), (679, 170)],
                    [(803, 140), (739, 137), (743, 173), (814, 177)],
                    [(679, 170), (743, 173), (747, 208), (672, 205)],
                    [(814, 177), (743, 173), (747, 208), (826, 214)],
                    [(672, 205), (747, 208), (752, 253), (668, 249)],
                    [(826, 214), (747, 208), (752, 253), (839, 258)],
                    [(668, 249), (752, 253), (755, 302), (662, 299)],
                    [(839, 258), (752, 253), (755, 302), (854, 305)],
                    [(662, 299), (755, 302), (759, 360), (657, 355)],
                    [(854, 305), (755, 302), (759, 360), (869, 362)],
                    [(657, 355), (759, 360), (760, 431), (645, 429)],
                    [(869, 362), (759, 360), (760, 431), (883, 436)],
                    [(645, 429), (760, 431), (760, 526), (631, 520)],
                    [(883, 436), (760, 431), (760, 526), (904, 529)],
                    [(631, 520), (760, 526), (762, 644), (606, 639)],
                    [(904, 529), (760, 526), (762, 644), (923, 644)],
                    [(606, 639), (762, 644), (923, 644), (932, 710), (596, 710)],
                ],
                "ip": "10.5.0.170",
            },
            "OFFICE2": {
                "borders": [
                    [(29, 493), (107, 444), (168, 543), (81, 598)],
                    [(168, 543), (182, 533), (194, 550), (297, 487), (245, 429), (138, 491)],
                    [(194, 550), (297, 487), (390, 581), (269, 654)],
                    [(269, 654), (390, 581), (509, 687), (466, 714), (318, 714)],
                    [(466, 714), (684, 714), (579, 642), (509, 687)],
                    [(509, 687), (579, 642), (646, 595), (518, 502), (390, 581)],
                    [(390, 581), (518, 502), (414, 418), (297, 487)],
                    [(245, 429), (268, 418), (255, 356), (309, 324), (414, 418), (297, 487)],
                    [(579, 642), (646, 595), (710, 550), (843, 637), (758, 713), (684, 714)],
                    [(309, 324), (414, 418), (528, 355), (406, 271)],
                    [(406, 271), (500, 235), (628, 305), (528, 355)],
                    [(518, 502), (414, 418), (528, 355), (641, 428)],
                    [(518, 502), (646, 595), (710, 550), (766, 506), (641, 428)],
                    [(710, 550), (843, 637), (941, 544), (816, 468), (766, 506)],
                    [(758, 713), (843, 637), (975, 714)],
                    [(975, 714), (843, 637), (941, 544), (1056, 616)],
                    [(975, 714), (1114, 713), (1143, 665), (1056, 616)],
                    [(1143, 665), (1056, 616), (1116, 528), (1189, 576)],
                    [(1056, 616), (1116, 528), (1011, 463), (941, 544)],
                    [(816, 468), (941, 544), (1011, 463), (899, 397)],
                    [(528, 355), (641, 428), (764, 349), (662, 290), (628, 305)],
                    [(641, 428), (766, 506), (816, 468), (875, 419), (764, 349)],
                    [(875, 419), (899, 397), (968, 339), (868, 281), (764, 349)],
                    [(764, 349), (868, 281), (777, 235), (662, 290)],
                    [(899, 397), (1011, 463), (1069, 396), (968, 339)],
                    [(1011, 463), (1116, 528), (1160, 451), (1069, 396)],
                    [(1116, 528), (1189, 576), (1228, 492), (1160, 451)],
                ],
                "ip": "10.5.0.182",
            },
            "OFFICE3": {
                "borders": [
                    [(128, 592), (302, 604), (298, 712), (110, 714)],
                    [(466, 709), (298, 712), (302, 604), (466, 609)],
                    [(466, 709), (466, 609), (603, 608), (615, 709)],
                    [(128, 592), (302, 604), (320, 436), (154, 435)],
                    [(302, 604), (466, 609), (463, 432), (320, 436)],
                    [(466, 609), (603, 608), (588, 424), (463, 432)],
                    [(588, 424), (780, 415), (795, 469), (650, 475), (593, 478)],
                    [(795, 469), (780, 415), (925, 405), (942, 455)],
                    [(942, 455), (925, 405), (1043, 396), (1062, 446)],
                    [(1062, 446), (1043, 396), (1158, 354), (1130, 435)],
                    [(463, 432), (476, 415), (477, 335), (612, 328), (622, 422), (588, 424)],
                    [(622, 422), (612, 328), (754, 320), (780, 415)],
                    [(780, 415), (754, 320), (886, 316), (925, 405)],
                    [(925, 405), (886, 316), (1002, 315), (1043, 396)],
                    [(1043, 396), (1002, 315), (1129, 310), (1158, 354)],
                    [(477, 335), (612, 328), (602, 250), (477, 257)],
                    [(602, 250), (612, 328), (754, 320), (730, 244)],
                    [(730, 244), (754, 320), (886, 316), (852, 245)],
                    [(852, 245), (886, 316), (1002, 315), (962, 244)],
                    [(962, 244), (1002, 315), (1129, 310), (1090, 249)],
                    [(482, 193), (477, 257), (602, 250), (594, 188)],
                    [(594, 188), (602, 250), (730, 244), (711, 184)],
                    [(711, 184), (730, 244), (852, 245), (823, 183)],
                    [(823, 183), (852, 245), (962, 244), (925, 185)],
                    [(925, 185), (962, 244), (1090, 249), (1052, 192)],
                    [(486, 142), (482, 193), (594, 188), (587, 135)],
                    [(587, 135), (594, 188), (711, 184), (696, 131)],
                    [(696, 131), (711, 184), (823, 183), (799, 135)],
                    [(799, 135), (823, 183), (925, 185), (892, 136)],
                    [(892, 136), (925, 185), (1052, 192), (1018, 144)],
                    [(492, 81), (486, 142), (587, 135), (581, 75)],
                    [(581, 75), (587, 135), (696, 131), (676, 72)],
                    [(676, 72), (696, 131), (799, 135), (768, 70)],
                    [(768, 70), (799, 135), (892, 136), (853, 77)],
                    [(853, 77), (892, 136), (1018, 144), (965, 87)],
                    [(500, 5), (492, 81), (581, 75), (573, 5)],
                    [(573, 5), (581, 75), (676, 72), (653, 5)],
                    [(653, 5), (676, 72), (768, 70), (730, 5)],
                    [(730, 5), (768, 70), (853, 77), (798, 4)],
                    [(798, 4), (853, 77), (965, 87), (893, 5)],
                    [(932, 6), (893, 5), (965, 87), (1014, 92)],
                    [(1014, 92), (965, 87), (1018, 144), (1063, 149)],
                    [(1063, 149), (1018, 144), (1052, 192), (1090, 249), (1137, 246)],
                    [(1137, 246), (1090, 249), (1129, 310), (1158, 354), (1205, 338)],
                ],
                "ip": "10.5.0.161",
            },
            # Inventory Cameras
            "GUDANGACC1": {
                "borders": [
                    [(632, 9), (616, 58), (664, 69), (674, 6)],
                    [(674, 6), (723, 4), (735, 86), (664, 69)],
                    [(723, 4), (777, 7), (790, 51), (730, 50)],
                    [(735, 86), (730, 50), (790, 51), (808, 104)],
                    [(735, 86), (701, 78), (700, 119), (760, 131), (759, 90)],
                    [(759, 90), (808, 104), (821, 146), (760, 131)],
                    [(700, 119), (760, 131), (765, 177), (700, 165)],
                    [(821, 146), (760, 131), (765, 177), (835, 194)],
                    [(700, 165), (765, 177), (773, 232), (700, 217)],
                    [(773, 232), (765, 177), (835, 194), (853, 252)],
                    [(700, 217), (773, 232), (778, 297), (699, 283)],
                    [(853, 252), (773, 232), (778, 297), (872, 323)],
                    [(699, 283), (778, 297), (784, 366), (697, 353)],
                    [(872, 323), (778, 297), (784, 366), (885, 387)],
                    [(697, 353), (784, 366), (788, 434), (693, 422)],
                    [(898, 456), (788, 434), (784, 366), (885, 387)],
                    [(693, 422), (788, 434), (796, 519), (688, 506)],
                    [(898, 456), (788, 434), (796, 519), (915, 537)],
                    [(688, 506), (796, 519), (801, 617), (688, 600)],
                    [(915, 537), (796, 519), (801, 617), (929, 631)],
                    [(688, 600), (801, 617), (805, 713), (683, 713)],
                    [(929, 631), (801, 617), (805, 713), (936, 713)],
                ],
                "ip": "10.5.0.110",
            },
            "GUDANGACC4": {
                "borders": [
                    [(713, 41), (664, 44), (671, 90), (731, 87)],
                    [(664, 44), (602, 48), (597, 95), (671, 90)],
                    [(534, 49), (602, 48), (597, 95), (515, 103)],
                    [(515, 103), (597, 95), (592, 149), (488, 157)],
                    [(592, 149), (597, 95), (671, 90), (677, 144)],
                    [(731, 87), (671, 90), (677, 144), (752, 143)],
                    [(488, 157), (592, 149), (581, 213), (464, 221)],
                    [(581, 213), (592, 149), (677, 144), (687, 206)],
                    [(752, 143), (677, 144), (687, 206), (778, 201)],
                    [(464, 221), (581, 213), (575, 273), (443, 282)],
                    [(575, 273), (581, 213), (687, 206), (698, 269)],
                    [(778, 201), (687, 206), (698, 269), (799, 262)],
                    [(443, 282), (575, 273), (568, 361), (416, 370)],
                    [(568, 361), (575, 273), (698, 269), (708, 352)],
                    [(799, 262), (698, 269), (708, 352), (826, 346)],
                    [(416, 370), (568, 361), (558, 454), (384, 458)],
                    [(558, 454), (568, 361), (708, 352), (721, 450)],
                    [(826, 346), (708, 352), (721, 450), (854, 441)],
                    [(384, 458), (558, 454), (549, 552), (358, 554)],
                    [(549, 552), (558, 454), (721, 450), (735, 551)],
                    [(854, 441), (721, 450), (735, 551), (886, 539)],
                    [(358, 554), (549, 552), (546, 638), (337, 637)],
                    [(546, 638), (549, 552), (735, 551), (744, 634)],
                    [(886, 539), (735, 551), (744, 634), (902, 621)],
                    [(337, 637), (546, 638), (539, 712), (321, 713)],
                    [(539, 712), (546, 638), (744, 634), (751, 713)],
                    [(902, 621), (744, 634), (751, 713), (922, 713)],
                ],
                "ip": "10.5.0.180",
            },
            "METALDET2": {
                "borders": [
                    [(652, 108), (720, 110), (719, 129), (647, 126)],
                    [(647, 126), (719, 129), (718, 155), (639, 149)],
                    [(639, 149), (718, 155), (717, 185), (624, 180)],
                    [(624, 180), (717, 185), (716, 216), (615, 210)],
                    [(615, 210), (716, 216), (713, 249), (603, 241)],
                    [(603, 241), (713, 249), (711, 287), (587, 281)],
                    [(587, 281), (711, 287), (708, 327), (572, 319)],
                    [(572, 319), (708, 327), (705, 385), (552, 373)],
                    [(552, 373), (705, 385), (701, 443), (534, 427)],
                    [(534, 427), (701, 443), (696, 504), (511, 486)],
                    [(511, 486), (696, 504), (692, 574), (485, 559)],
                    [(442, 554), (415, 636), (683, 649), (692, 574), (485, 559)],
                    [(388, 710), (415, 636), (683, 649), (677, 714)],
                    [(705, 385), (858, 389), (863, 462), (700, 458), (701, 443)],
                    [(700, 458), (863, 462), (882, 576), (692, 574)],
                    [(882, 576), (692, 574), (683, 649), (677, 714), (904, 711)],
                ],
                "ip": "10.5.0.185",
            },
        }

        borders = config[self.camera_name]["borders"]
        ip = config[self.camera_name]["ip"]
        camera_names = list(config.keys())
        indices = {name: idx + 1 for idx, name in enumerate(camera_names)}
        index = indices[self.camera_name]
        return borders, ip, index

    def process_model(self, frame):
        with torch.no_grad():
            results = self.broom_model(frame, stream=True, imgsz=960)
        return results

    def export_frame(self, results, color, pairs):
        points = []
        coords = []
        keypoint_positions = []
        confidence_threshold = self.BROOM_CONFIDENCE_THRESHOLD

        for result in results:
            keypoints_data = result.keypoints
            if keypoints_data is not None and keypoints_data.xy is not None and keypoints_data.conf is not None:
                if keypoints_data.shape[0] > 0:
                    keypoints_array = keypoints_data.xy.cpu().numpy()
                    keypoints_conf = keypoints_data.conf.cpu().numpy()
                    for keypoints_per_object, keypoints_conf_per_object in zip(keypoints_array, keypoints_conf):
                        keypoints_list = []
                        for kp, kp_conf in zip(keypoints_per_object, keypoints_conf_per_object):
                            if kp_conf >= confidence_threshold:
                                x, y = kp[0], kp[1]
                                keypoints_list.append((int(x), int(y)))
                            else:
                                keypoints_list.append(None)
                        keypoint_positions.append(keypoints_list)
                        for point in keypoints_list:
                            if point is not None:
                                points.append(point)
                        for i, j in pairs:
                            if i < len(keypoints_list) and j < len(keypoints_list):
                                if keypoints_list[i] is not None and keypoints_list[j] is not None:
                                    coords.append((keypoints_list[i], keypoints_list[j], color))
                else:
                    continue
        return points, coords, keypoint_positions

    def frame_capture(self):
        rtsp_url = self.rtsp_url
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print("Failed to open stream. Retrying in 5 seconds...")
                cap.release()
                time.sleep(5)
                continue

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame. Reconnecting in 5 seconds...")
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    pass
            cap.release()

    def process_frame(self, frame, current_time, percentage_green, pairs_broom):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results = self.process_model(frame_resized)
        points, coords, keypoint_positions = self.export_frame(results, (0, 255, 0), pairs_broom)

        border_colors = [(0, 255, 0) if state["is_green"] else (0, 255, 255) for state in self.border_states.values()]

        broom_overlapping_any_border = False

        for border_id, border_pt in enumerate(self.borders_pts):
            broom_overlapping = False
            for keypoints_list in keypoint_positions:
                for idx in [2, 3, 4]:
                    if idx < len(keypoints_list):
                        kp = keypoints_list[idx]
                        if kp is not None:
                            result = cv2.pointPolygonTest(border_pt, kp, False)
                            if result >= 0:
                                broom_overlapping = True
                                broom_overlapping_any_border = True
                                break
                if broom_overlapping:
                    break

            if broom_overlapping:
                if self.border_states[border_id]["last_broom_overlap_time"] is None:
                    self.border_states[border_id]["last_broom_overlap_time"] = current_time
                else:
                    delta_time = current_time - self.border_states[border_id]["last_broom_overlap_time"]
                    self.border_states[border_id]["broom_overlap_time"] += delta_time
                    self.border_states[border_id]["last_broom_overlap_time"] = current_time

                if self.border_states[border_id]["broom_overlap_time"] >= self.BROOM_TOUCH_THRESHOLD:
                    self.border_states[border_id]["is_green"] = True
                    border_colors[border_id] = (0, 255, 0)
            else:
                self.border_states[border_id]["last_broom_overlap_time"] = None

        green_borders_exist = any(state["is_green"] for state in self.border_states.values())
        if green_borders_exist:
            if not self.is_counting:
                self.first_green_time = current_time
                self.is_counting = True

            if broom_overlapping_any_border:
                self.broom_absence_timer_start = current_time
            else:
                if self.broom_absence_timer_start is None:
                    self.broom_absence_timer_start = current_time
                elif (current_time - self.broom_absence_timer_start) >= self.BROOM_ABSENCE_THRESHOLD:
                    print(f"Resetting broom borders in percentage {percentage_green:.2f}%")
                    if percentage_green >= self.BROOM_PERCENTAGE_GREEN_THRESHOLD:
                        print(f"Green broom border is bigger than {self.BROOM_PERCENTAGE_GREEN_THRESHOLD}% and data is sent to server")
                        if self.first_green_time is not None:
                            self.elapsed_time = current_time - self.first_green_time
                        overlay = frame_resized.copy()
                        alpha = 0.5
                        for border_pt, color in zip(self.borders_pts, border_colors):
                            cv2.fillPoly(overlay, pts=[border_pt], color=color)
                        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
                        minutes, seconds = divmod(int(self.elapsed_time), 60)
                        time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
                        if self.show_text:
                            cvzone.putTextRect(frame_resized, time_str, (10, 100), scale=1, thickness=2, offset=5)
                            cvzone.putTextRect(
                                frame_resized,
                                f"Percentage of Green Border: {percentage_green:.2f}%",
                                (10, 50),
                                scale=1,
                                thickness=2,
                                offset=5,
                            )
                            cvzone.putTextRect(
                                frame_resized,
                                f"FPS: {int(self.fps)}",
                                (10, 75),
                                scale=1,
                                thickness=2,
                                offset=5,
                            )
                        image_path = "main/images/green_borders_image_182.jpg"
                        cv2.imwrite(image_path, frame_resized)
                        self.send_to_server("10.5.0.2", percentage_green, self.elapsed_time, image_path)

                    for idx in range(len(self.borders)):
                        self.border_states[idx] = {
                            "is_green": False,
                            "broom_overlap_time": 0.0,
                            "last_broom_overlap_time": None,
                        }
                        border_colors[idx] = (0, 255, 255)
                    self.first_green_time = None
                    self.is_counting = False
                    self.broom_absence_timer_start = None
        else:
            self.broom_absence_timer_start = None
            if self.is_counting:
                self.first_green_time = None
                self.is_counting = False

        if percentage_green == 100:
            print("Percentage broom green is 100%, performing immediate reset and data send.")
            if self.first_green_time is not None:
                self.elapsed_time = current_time - self.first_green_time
            overlay = frame_resized.copy()
            alpha = 0.5
            for border_pt, color in zip(self.borders_pts, border_colors):
                cv2.fillPoly(overlay, pts=[border_pt], color=color)
            cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
            minutes, seconds = divmod(int(self.elapsed_time), 60)
            time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
            if self.show_text:
                cvzone.putTextRect(frame_resized, time_str, (10, 100), scale=1, thickness=2, offset=5)
                cvzone.putTextRect(
                    frame_resized,
                    f"Percentage of Green Border: {percentage_green:.2f}%",
                    (10, 50),
                    scale=1,
                    thickness=2,
                    offset=5,
                )
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
            image_path = "main/images/green_borders_image_182.jpg"
            cv2.imwrite(image_path, frame_resized)
            self.send_to_server("10.5.0.2", percentage_green, self.elapsed_time, image_path)

            for idx in range(len(self.borders)):
                self.border_states[idx] = {
                    "is_green": False,
                    "broom_overlap_time": 0.0,
                    "last_broom_overlap_time": None,
                }
                border_colors[idx] = (0, 255, 255)
            self.first_green_time = None
            self.is_counting = False
            self.broom_absence_timer_start = None

        if points and coords:
            for x, y, color in coords:
                cv2.line(frame_resized, x, y, color, 2)
            for point in points:
                cv2.circle(frame_resized, point, 4, (0, 255, 255), -1)
        overlay = frame_resized.copy()
        alpha = 0.5
        for border_pt, color in zip(self.borders_pts, border_colors):
            cv2.fillPoly(overlay, pts=[border_pt], color=color)
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
        if self.is_counting and self.first_green_time is not None:
            self.elapsed_time = current_time - self.first_green_time
            minutes, seconds = divmod(int(self.elapsed_time), 60)
            time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
            if self.show_text:
                cvzone.putTextRect(frame_resized, time_str, (10, 100), scale=1, thickness=2, offset=5)

        return frame_resized

    def send_to_server(self, host, percentage_green, elapsed_time, image_path):
        def server_address(host):
            if host == "localhost":
                user = "root"
                password = "robot123"
                database = "report_ai_cctv"
                port = 3306
            elif host == "10.5.0.2":
                user = "robot"
                password = "robot123"
                database = "report_ai_cctv"
                port = 3307
            return user, password, database, port

        try:
            user, password, database, port = server_address(host)
            connection = pymysql.connect(host=host, user=user, password=password, database=database, port=port)
            cursor = connection.cursor()
            table = "empbro"
            category = "Menyapu Lantai"
            camera_name = self.camera_name
            timestamp_done = datetime.now()
            timestamp_start = timestamp_done - timedelta(seconds=elapsed_time)

            timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S")

            # **Define the parameter time to compare with (e.g., 09:00:00)**
            parameter_time_str = "08:30:00"
            parameter_time = datetime.strptime(parameter_time_str, "%H:%M:%S").time()

            # **Extract the time portion of timestamp_done**
            timestamp_done_time = timestamp_done.time()

            # **Compare and set isdiscipline**
            if timestamp_done_time > parameter_time:
                isdiscipline = "Tidak disiplin"
            else:
                isdiscipline = "Disiplin"

            with open(image_path, "rb") as file:
                binary_image = file.read()

            query = f"""
            INSERT INTO {table} (cam, category, timestamp_start, timestamp_done, percentage, image_done, isdiscipline)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                query,
                (
                    camera_name,
                    category,
                    timestamp_start_str,
                    timestamp_done_str,
                    percentage_green,
                    binary_image,
                    isdiscipline,
                ),
            )
            connection.commit()
            print(f"Broom data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"Error sending broom data to server: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

    def main(self):
        pairs_broom = [(0, 1), (1, 2), (2, 3), (2, 4)]
        process_every_n_frames = 2
        frame_count = 0

        window_name = f"BROOM{self.idx} : {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        if self.video_fps is not None:
            # Local video file, process frames in the main thread
            cap = cv2.VideoCapture(self.rtsp_url)
            frame_delay = int(1000 / self.video_fps)  # Delay in milliseconds

            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of video file or cannot read the frame.")
                    break

                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue

                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                if time_diff > 0:
                    self.fps = 1 / time_diff
                else:
                    self.fps = 0
                self.prev_frame_time = current_time
                total_borders = len(self.borders)
                green_borders = sum(1 for state in self.border_states.values() if state["is_green"])
                percentage_green = (green_borders / total_borders) * 100

                frame_resized = self.process_frame(frame, current_time, percentage_green, pairs_broom)

                if self.show_text:
                    cvzone.putTextRect(
                        frame_resized,
                        f"Percentage of Green Border: {percentage_green:.2f}%",
                        (10, 50),
                        scale=1,
                        thickness=2,
                        offset=5,
                    )
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)

                cv2.imshow(window_name, frame_resized)
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                adjusted_delay = max(int(frame_delay - processing_time), 1)
                key = cv2.waitKey(adjusted_delay) & 0xFF
                if key == ord("n"):
                    break
                elif key == ord("s"):
                    self.show_text = not self.show_text

            cap.release()
            cv2.destroyAllWindows()
        else:
            # RTSP stream, use frame capture thread
            self.frame_thread = threading.Thread(target=self.frame_capture)
            self.frame_thread.daemon = True
            self.frame_thread.start()

            while True:
                if self.stop_event.is_set():
                    break
                try:
                    frame = self.frame_queue.get(timeout=5)
                except queue.Empty:
                    continue

                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue

                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                if time_diff > 0:
                    self.fps = 1 / time_diff
                else:
                    self.fps = 0
                self.prev_frame_time = current_time
                total_borders = len(self.borders)
                green_borders = sum(1 for state in self.border_states.values() if state["is_green"])
                percentage_green = (green_borders / total_borders) * 100

                frame_resized = self.process_frame(frame, current_time, percentage_green, pairs_broom)

                if self.show_text:
                    cvzone.putTextRect(
                        frame_resized,
                        f"Percentage of Green Border: {percentage_green:.2f}%",
                        (10, 50),
                        scale=1,
                        thickness=2,
                        offset=5,
                    )
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)

                cv2.imshow(window_name, frame_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    self.stop_event.set()
                    break
                elif key == ord("s"):
                    self.show_text = not self.show_text

            cv2.destroyAllWindows()
            self.frame_thread.join()


def run_broom(BROOM_ABSENCE_THRESHOLD, BROOM_TOUCH_THRESHOLD, BROOM_PERCENTAGE_GREEN_THRESHOLD, camera_name, window_size=(540, 360), rtsp_url=None):
    detector = BroomDetector(
        BROOM_ABSENCE_THRESHOLD=BROOM_ABSENCE_THRESHOLD,
        BROOM_TOUCH_THRESHOLD=BROOM_TOUCH_THRESHOLD,
        BROOM_PERCENTAGE_GREEN_THRESHOLD=BROOM_PERCENTAGE_GREEN_THRESHOLD,
        camera_name=camera_name,
        window_size=window_size,
        rtsp_url=rtsp_url,
    )

    detector.main()


if __name__ == "__main__":
    run_broom(
        BROOM_ABSENCE_THRESHOLD=30,
        BROOM_TOUCH_THRESHOLD=0,
        BROOM_PERCENTAGE_GREEN_THRESHOLD=50,
        camera_name="METALDET2",
        window_size=(540, 360),
        # rtsp_url="videos/test1.mp4",
    )
