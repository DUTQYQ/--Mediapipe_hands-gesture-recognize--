import mediapipe as mp
import cv2
import numpy as np


if __name__ == "__main__":
    
    
    # 打开摄像头

    cap = cv2.VideoCapture(0)
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
  
    while True:
     
        # 读取一帧图像
        success, img = cap.read()
        if not success:
            continue
        image_height, image_width, _ = np.shape(img)
        
        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 得到检测结果
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                 
                print("\r%.2f %.2f %.2f %.2f %.2f %.2f "%(hand.landmark[0].z,hand.landmark[4].z,hand.landmark[8].z,hand.landmark[12].z,hand.landmark[16].z,hand.landmark[20].z),end="")
               
                mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)
                 
                # for i in range(21):
                    # pos_x = hand.landmark[i].x*image_width
                    # pos_y = hand.landmark[i].y*image_height
                    # # 画点
                    # cv2.circle(img, (int(pos_x),int(pos_y)), 3, (0,255,255),-1)
                    
       
        cv2.imshow("hands",img)

        key =  cv2.waitKey(1) & 0xFF   

        # 按键 "q" 退出
        if key ==  ord('q'):
            break
    cap.release() 
       
    
    
    
    
    
    
    
    

    