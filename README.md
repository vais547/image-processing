# image-processing
http://localhost:8810/notebooks/vaishnavii/program2.ipynb

develop a program to display grayscale image using read and write operations.
import cv2
img=cv2.imread('flower2.jpg',0)
cv2.imshow('flower2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
output:
![image](https://user-images.githubusercontent.com/98145574/173815026-c72d45aa-07c3-4dab-81d2-6bd3ad7bf938.png)

2.matplotlib
import cv2
import matplotlib.pyplot as plt
image=cv2.imread('flower2.jpg')
plt.imshow(image)
plt.show()
![download](https://user-images.githubusercontent.com/98145574/173815741-b99f5bc0-0176-4da3-a1fa-d081a22caea8.png)

3.rotation
from PIL import Image
original_Image=Image.open('flower2.jpg')
rotate_image1=original_Image.rotate(180)
rotate_image1.show()

![image](https://user-images.githubusercontent.com/98145574/173816447-cb37138f-82d8-495b-ba48-40ebd93134da.png)


4.
from PIL import ImageColor
img1=ImageColor.getrgb("yellow")
print(img1)
img2=ImageColor.getrgb("red")
print(img2)

output:
(255, 255, 0)
(255, 0, 0)

5.image using color spaces
from PIL import Image
img=Image.new('RGB',(200,400),(255,0,0))
img.show()

![image](https://user-images.githubusercontent.com/98145574/173816816-bc8e51e4-1c96-4a87-aa6f-408ade55af47.png)

6.image using various colours
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('flower2.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

output:
![download](https://user-images.githubusercontent.com/98145574/173817498-2cb91906-9749-4ad9-907f-9d420ff0b8f4.png)

![download](https://user-images.githubusercontent.com/98145574/173817570-ab4b3782-e345-45f0-9441-2951ef91ff0c.png)
![download](https://user-images.githubusercontent.com/98145574/173817632-60c12506-8db8-48fa-acc8-47ba0853323e.png)

7.
image=Image.open('flower2.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("mode:",image.mode)
print("size:",image.size)
print("width:",image.width)
print("height:",image.height)
image.close();

output:
Filename: flower2.jpg
Format: JPEG
mode: RGB
size: (239, 211)
width: 239
height: 211
