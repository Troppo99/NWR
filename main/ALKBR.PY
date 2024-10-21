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


class BroomDetector:
    def __init__(
        self,
        BROOM_ABSENCE_THRESHOLD=10,
        BROOM_TOUCH_THRESHOLD=0,
        PERCENTAGE_GREEN_THRESHOLD=50,
        camera_name=None,
    ):
        self.CONFIDENCE_THRESHOLD_BROOM = 0.9
        self.BROOM_ABSENCE_THRESHOLD = BROOM_ABSENCE_THRESHOLD
        self.BROOM_TOUCH_THRESHOLD = BROOM_TOUCH_THRESHOLD
        self.PERCENTAGE_GREEN_THRESHOLD = PERCENTAGE_GREEN_THRESHOLD
        self.new_width, self.new_height = 360, 202
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
        self.rtsp_url = f"rtsp://admin:oracle2015@{camera_name}:554/Streaming/Channels/1"
        self.borders = self.camera_config(camera_name)

        for border in self.borders:
            scaled_border = []
            for x, y in border:
                scaled_x = int(x * self.scale_x)
                scaled_y = int(y * self.scale_y)
                scaled_border.append((scaled_x, scaled_y))
            self.scaled_borders.append(scaled_border)

        self.border_states = {
            idx: {
                "sapu_time": None,
                "orang_time": None,
                "is_green": False,
                "person_and_broom_detected": False,
                "broom_overlap_time": 0.0,
                "last_broom_overlap_time": None,
            }
            for idx in range(len(self.borders))
        }
        self.borders_pts = [np.array(border, np.int32) for border in self.scaled_borders]
        self.model_broom = YOLO("broom5l.pt").to("cuda")
        self.model_broom.overrides["verbose"] = False
        print(f"Model Broom device: {next(self.model_broom.model.parameters()).device}")

    def camera_config(self, camera_name):
        config = {
            "10.5.0.182": [
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
            "10.5.0.170": [
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
            "10.5.0.161": [
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
            "10.5.0.110": [
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
            "10.5.0.180": [
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
            "10.5.0.185": [
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
            "10.5.0.146": [
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
            "10.5.0.183": [
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
            "10.5.0.195": [
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
            "10.5.0.201": [
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
        }
        return config[camera_name]

    def process_model_broom(self, frame):
        with torch.no_grad():
            results_broom = self.model_broom(frame, imgsz=960)
        return results_broom

    def export_frame_broom(self, results, color, pairs):
        points = []
        coords = []
        keypoint_positions = []
        confidence_threshold = self.CONFIDENCE_THRESHOLD_BROOM

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

    def process_frame(self, frame, current_time, percentage_green, pairs_broom):
        frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        results_broom = self.process_model_broom(frame_resized)
        points_broom, coords_broom, keypoint_positions = self.export_frame_broom(
            results_broom, (0, 255, 0), pairs_broom
        )

        border_colors = [(0, 255, 0) if state["is_green"] else (0, 255, 255) for state in self.border_states.values()]

        broom_overlapping_any_border = False

        for border_id, border_pt in enumerate(self.borders_pts):
            sapu_overlapping = False
            for keypoints_list in keypoint_positions:
                for idx in [2, 3, 4]:
                    if idx < len(keypoints_list):
                        kp = keypoints_list[idx]
                        if kp is not None:
                            result = cv2.pointPolygonTest(border_pt, kp, False)
                            if result >= 0:
                                sapu_overlapping = True
                                broom_overlapping_any_border = True
                                break
                if sapu_overlapping:
                    break

            if sapu_overlapping:
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
                    print("Resetting borders")
                    if percentage_green >= self.PERCENTAGE_GREEN_THRESHOLD:
                        print(
                            f"Green border is bigger than {self.PERCENTAGE_GREEN_THRESHOLD}% and data is sent to server"
                        )
                        if self.first_green_time is not None:
                            self.elapsed_time = current_time - self.first_green_time
                        overlay = frame_resized.copy()
                        alpha = 0.5
                        for border_pt, color in zip(self.borders_pts, border_colors):
                            cv2.fillPoly(overlay, pts=[border_pt], color=color)
                        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)
                        minutes, seconds = divmod(int(self.elapsed_time), 60)
                        time_str = f"Elapsed Time: {minutes:02d}:{seconds:02d}"
                        cvzone.putTextRect(frame_resized, time_str, (10, 50), scale=1, thickness=2, offset=5)
                        cvzone.putTextRect(
                            frame_resized,
                            f"Persentase Border Hijau: {percentage_green:.2f}%",
                            (10, 75),
                            scale=1,
                            thickness=2,
                            offset=5,
                        )
                        cvzone.putTextRect(
                            frame_resized, f"FPS: {int(self.fps)}", (10, 100), scale=1, thickness=2, offset=5
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

        if points_broom and coords_broom:
            for x, y, color in coords_broom:
                cv2.line(frame_resized, x, y, color, 2)
            for point in points_broom:
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
            cvzone.putTextRect(frame_resized, time_str, (10, 50), scale=1, thickness=2, offset=5)

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
            camera_name = self.camera_name
            timestamp_done = datetime.now()
            timestamp_start = timestamp_done - timedelta(seconds=elapsed_time)

            timestamp_done_str = timestamp_done.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_start_str = timestamp_start.strftime("%Y-%m-%d %H:%M:%S")

            with open(image_path, "rb") as file:
                binary_image = file.read()

            query = f"""
            INSERT INTO {table} (cam, timestamp_start, timestamp_done, elapsed_time, percentage, image_done)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                query,
                (
                    camera_name,
                    timestamp_start_str,
                    timestamp_done_str,
                    elapsed_time,
                    percentage_green,
                    binary_image,
                ),
            )
            connection.commit()
            print(f"Data successfully sent to server.")
        except pymysql.MySQLError as e:
            print(f"Error sending data to server: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals():
                connection.close()

    def run(self):
        rtsp_url = self.rtsp_url
        cap = cv2.VideoCapture(rtsp_url)
        pairs_broom = [(0, 1), (1, 2), (2, 3), (2, 4)]
        frame_count = 0
        process_every_n_frames = 2

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
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
            cvzone.putTextRect(
                frame_resized,
                f"Persentase Border Hijau: {percentage_green:.2f}%",
                (10, 75),
                scale=1,
                thickness=2,
                offset=5,
            )
            cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 100), scale=1, thickness=2, offset=5)
            cv2.imshow("Broom and Person Detection", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = BroomDetector()
    detector.run()
