# image-processing
http://localhost:8810/notebooks/vaishnavii/program2.ipynb

1.develop a program to display grayscale image using read and write operations.<br>
import cv2<br>
img=cv2.imread('flower2.jpg',0)<br>
cv2.imshow('flower2',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output:<br>
![image](https://user-images.githubusercontent.com/98145574/173815026-c72d45aa-07c3-4dab-81d2-6bd3ad7bf938.png)<br>

2.develop a program to display the image using matplotlib<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image=cv2.imread('flower2.jpg')<br>
plt.imshow(image)<br>
plt.show()<br>
![download](https://user-images.githubusercontent.com/98145574/173815741-b99f5bc0-0176-4da3-a1fa-d081a22caea8.png)<br>

3.#develop a program to dispaly linear tranformation.<br>
from PIL import Image<br>
original_Image=Image.open('flower2.jpg')<br>
rotate_image1=original_Image.rotate(180)<br>
rotate_image1.show()<br>

![image](https://user-images.githubusercontent.com/98145574/173816447-cb37138f-82d8-495b-ba48-40ebd93134da.png)<br>


4.#develop a program to convert colour string into RGB colour values.<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>

output:<br>
(255, 255, 0)<br>
(255, 0, 0)<br>

5.#write a pprogram to create image using colour spaces. <br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,0,0))<br>
img.show()<br>

![image](https://user-images.githubusercontent.com/98145574/173816816-bc8e51e4-1c96-4a87-aa6f-408ade55af47.png)<br>

6.#develop a program to visualize images using various colours.<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('flower2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>

output:<br>
![download](https://user-images.githubusercontent.com/98145574/173817498-2cb91906-9749-4ad9-907f-9d420ff0b8f4.png)<br>

![download](https://user-images.githubusercontent.com/98145574/173817570-ab4b3782-e345-45f0-9441-2951ef91ff0c.png)<br>
![download](https://user-images.githubusercontent.com/98145574/173817632-60c12506-8db8-48fa-acc8-47ba0853323e.png)<br>

7.<br>
image=Image.open('flower2.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("mode:",image.mode)<br>
print("size:",image.size)<br>
print("width:",image.width)<br>
print("height:",image.height)<br>
image.close();<br>

output:<br>
Filename: flower2.jpg<br>
Format: JPEG<br>
mode: RGB<br>
size: (239, 211)<br>
width: 239<br>
height: 211<br>

8.resize<br>
import cv2<br>
img=cv2.imread('leaf3.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
imgresize=cv2.resize(img,(400,160))<br>
cv2.imshow('resized image',imgresize)<br>
print('Resized image legth width',imgresize.shape)<br>
cv2.waitKey(0)<br>

output:<br>
original image length width (177, 284, 3)<br>
Resized image legth width (160, 400, 3)<br>

![image](https://user-images.githubusercontent.com/98145574/174038962-29d31a06-a2f1-4820-b0f7-4ec79cbc62b8.png)<br>
![image](https://user-images.githubusercontent.com/98145574/174039141-54e97615-25c5-412b-a226-4bca6dd9a729.png)<br>

9.import cv2<br>
#read the image file.<br>
img=cv2.imread('leaf1.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>

##gray scale
img=cv2.imread("leaf1.jpg",0)<br>
cv2.imshow("gray",img)<br>
cv2.waitKey(0)<br>

#binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

output:<br>
![image](https://user-images.githubusercontent.com/98145574/174042660-ca59280b-f556-4665-8ea9-a97f5f4c571b.png)<br>
![image](https://user-images.githubusercontent.com/98145574/174042719-882b7e9d-93b7-4620-9e18-3e5f93580810.png)<br>
![image](https://user-images.githubusercontent.com/98145574/174042778-9661ede2-3c31-4807-9762-c44aa8107e57.png)<br>





