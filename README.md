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
