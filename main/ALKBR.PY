# ALKBR.py

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
        self.new_width, self.new_height = 960, 540
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
    # Default values; can be overridden when importing the module
    detector = BroomDetector()
    detector.run()
