# 基于Mediapipe实现的实时手势识别和图片分类 项目结题报告

Teammennber：丘宇乾，周泽霖

为完成**手势识别**任务，团队采用基于**Mediapipe**，**opencv**库 实现对手势的实时检测和图像分类

mediapipe可以实时扫描图片并提供**21个关键点**的3d坐标

![21](https://github.com/DUTQYQ/-Mediapipe-/blob/main/20210612195704176.png)
***


```python
#构造凸包点
import numpy as np

list_lms = np.array(list_lms,dtype=np.int32)

hull_index = [0,1,2,3,5,10,14,18,17,10]

hull = cv2.convexHull(list_lms[hull_index,:])
```

### 选取 0,1,2,3,5,10,14,18,17,10 点作为“**凸包点**” （手掌范围）

---
## **实时检测**
用` cap = cv2.VideoCapture(0)` 读取逐帧图片，进行识别


```python
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
               
                mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS) 画点
``` 

#### **对图像进行预处理**

---
## **图片分类**

对**文件夹test**中的图片进行识别，识别后放入**output文件夹**

#### test文件夹中待检测的图片

![test](https://github.com/DUTQYQ/-Mediapipe-/blob/main/test.png)

```python
# 获取文件夹中所有图片的路径
image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpg')]
```

经同实时检测一样**画点**，**画出凸包点**处理后，把处理后的文件放在output文件夹

---

![test](https://github.com/DUTQYQ/-Mediapipe-/blob/main/output.png)

---

## 进行检测

### **14个手势**分别为

**1, 3, 4, 5, 6 ,8 , 9, yeah , iloveyou , pink , gun, thumbUp ,fist, heartSingle**，

___
***
## 关键步骤

```python
            for i in ll:
                pt = (int(list_lms[i][0]),int(list_lms[i][1]))
                dist= cv2.pointPolygonTest(hull,pt,True)
                dis_5_9 = list_lms[5, :] - list_lms[9, :]
                dis_5_9 = np.sqrt(np.dot(dis_5_9, dis_5_9))
                if (-dist)/dis_5_9 >0.6:
                    up_fingers.append(i)
```
#### ——————————**判断哪些手指立起来**——————————
**记录向量长度`dis_5_9`**
## `dis_5_9`为基准点，后续的长度判断都基于此展开 
## 指尖离手掌的距离比上基准距离大于0.6时，判断手指伸出
     

### 例如检测手势heartSingle

```python
         elif len(up_fingers)==2 and up_fingers[0]==4 and up_fingers[1]==8:
        dis_4_8 = list_lms[4, :] - list_lms[8, :]
        dis_4_8 = np.sqrt(np.dot(dis_4_8, dis_4_8))

        dis_5_0 = list_lms[5, :] - list_lms[0, :]
        dis_5_0 = np.sqrt(np.dot(dis_5_0, dis_5_0))

        if(dis_5_0/dis_4_8>1.3):
            str_guester = "heartSingle"
        else:
            str_guester="gun"
```

此时可知 长度比值大于1.3时，为heartsingle。 长度比值小于1.3时，为gun

### 用比值而不单纯用距离数值来并表示的原因为 如果**手距离移动（离远）**的同时，点的大小会发生变化，而比例不变，导致判断错误。或者可能指尖**稍微出去**判断为伸出（其实没伸出）

---
---
---

## **效果分析**


## 实时：

![test](https://github.com/DUTQYQ/-Mediapipe-/blob/main/%E5%AE%9E%E6%97%B6%E6%95%88%E6%9E%9C.png)

## 图片文件处理

![test](https://github.com/DUTQYQ/-Mediapipe-/blob/main/output.png)
### 发现前50个图片识别结果为''(空)

#### 原因可能有几个：

#### **1.图片问题：像素太低，手势角度刁钻，导致mediapipe不能读出手的21点**

#### **2.算法问题：当凸包点连成一条线的时候，`lim`数据缺失。导致识别失败**

# -------------------------------------------------------------------------------------

# **总结**

### 使用mediapipe在**实时中可以较为完美的完成手势的识别（单手）**
### 但是对于图片文件，**由于图片的像素问题，而且手势只能单帧检测，对一张图片检测导致图片可得数据较视频来比更少，导致识别的正确率大打折扣**




    

2024.3.24
