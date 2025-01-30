import time

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 定义手部关键点的连接
        self.HAND_CONNECTIONS = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # 拇指
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # 食指
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # 中指
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # 无名指
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # 小指
        ]

    def find_hands(self, img, draw=True):
        """
        检测图像中的手部并绘制标记
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # 绘制手部关键点
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
                    )
        return img

    def find_positions(self, img, hand_no=0):
        """
        获取指定手的所有关键点位置
        """
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                for id, landmark in enumerate(hand.landmark):
                    height, width, _ = img.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([id, cx, cy])
        return landmark_list


def main():
    # 设置摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头

    # 创建追踪器实例
    tracker = HandTracker()

    # 用于计算FPS
    p_time = 0
    c_time = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # 翻转图像(可选，使显示更直观)
        img = cv2.flip(img, 1)

        # 检测手部并绘制
        img = tracker.find_hands(img)

        # 获取手部关键点位置
        landmark_list = tracker.find_positions(img)
        if landmark_list:
            # 示例：打印食指尖端位置 (关键点8)
            if len(landmark_list) > 8:
                print(f"食指位置: {landmark_list[8]}")

        # 计算和显示FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # 显示图像
        cv2.imshow("Hand Tracking", img)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
