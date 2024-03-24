import mediapipe as mp
import cv2
import numpy as np
import os
from hand_feature import get_str_guester
from hand_feature import get_angle

# 指定文件夹路径
folder_path = "C:/Users/qiuyu/Desktop/Gesture/test"
new_path="C:/Users/qiuyu/Desktop/Gesture/output"

# 获取文件夹中所有图片的路径
image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpg')]

# 定义手 检测对象
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 用于给识别后的图片编号
gesture_count = 1

for image_path in image_paths:
    # 读取图片
    img = cv2.imread(image_path)
    image_height, image_width, _ = np.shape(img)

    # 转换为RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 得到检测结果
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            list_lms = []
            for i in range(21):
                pos_x = hand.landmark[i].x * image_width
                pos_y = hand.landmark[i].y * image_height
                list_lms.append([int(pos_x), int(pos_y)])

            list_lms = np.array(list_lms, dtype=np.int32)
            hull_index =  [0,1,2,3,5,10,14,18,17,10]
            hull = cv2.convexHull(list_lms[hull_index, :])
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)

            n_fig = -1
            ll = [4, 8, 12, 16, 20]
            up_fingers = []

            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                dis_5_9 = list_lms[5, :] - list_lms[9, :]
                dis_5_9 = np.sqrt(np.dot(dis_5_9, dis_5_9))
                if (-dist) / dis_5_9 > 0.6:
                    up_fingers.append(i)

            str_guester = get_str_guester(up_fingers, list_lms)
            cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                        cv2.LINE_AA)

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
            for i in ll:
                pos_x = hand.landmark[i].x*image_width
                pos_y = hand.landmark[i].y*image_height
                # 画点
                cv2.circle(img, (int(pos_x),int(pos_y)), 3, (0,255,255),-1)

    # 生成新的文件名，手势名称加编号
    gesture_name = str_guester + "_" + os.path.basename(image_path)
    new_image_path = os.path.join(new_path, gesture_name + ".jpg")

    # 保存识别后的图片
    cv2.imwrite(new_image_path, img)

    gesture_count += 1

# 释放资源
cv2.destroyAllWindows()
