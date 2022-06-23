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


from skimage import io
import matplotlib.pyplot as plt
url='https://cdn.theatlantic.com/thumbor/viW9N1IQLbCrJ0HMtPRvXPXShkU=/0x131:2555x1568/976x549/media/img/mt/2017/06/shutterstock_319985324/original.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()

![download](https://user-images.githubusercontent.com/98145574/175021544-cece9979-329f-4b72-bed4-772985eb149b.png)


import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('fish.jpg')
plt.imshow(img)
plt.show()


![download](https://user-images.githubusercontent.com/98145574/175018057-96e6bfaa-3c58-44cb-9994-999c5bebc362.png)

hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
light_orange=(1, 190, 200)
dark_orange=(18, 255, 255)
mask=cv2.inRange(hsv_img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()

![download](https://user-images.githubusercontent.com/98145574/175018146-f2f12bb8-5ee3-42c7-90ac-1dc5d76e2bff.png)
![download](https://user-images.githubusercontent.com/98145574/175018184-c7b9e33a-fad6-4e76-b201-af4b6f45d421.png)


light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()

![download](https://user-images.githubusercontent.com/98145574/175018278-c425d835-3b28-4bf5-83ac-49870341ab72.png)
![download](https://user-images.githubusercontent.com/98145574/175018369-33dd1a0b-9f54-455d-b32a-5fa06fc931d5.png)


final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()

![download](https://user-images.githubusercontent.com/98145574/175018525-539b2079-6856-4b31-8205-bef0dbf0f176.png)


blur=cv2.GaussianBlur(final_result, (7,7), 0)
plt.imshow(blur)
plt.show()

![download](https://user-images.githubusercontent.com/98145574/175018694-f2084ce4-3e1f-4f7f-858f-7691005efcf5.png)


3)Develop the program to change the image to different color spaces.
import cv2 
img=cv2.imread("butterfly2.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLs image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/98145574/175263583-3385ca4c-b120-4e3a-9340-67073ba818da.png)



import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt

#Read file
img1=cv2.imread('butter1.jpg')
img2=cv2.imread('butter2.jpg')

#numpy add
fimg1 = img1 + img2
plt.imshow(fimg1)
plt.show()

#saving
cv2.imwrite('output.jpg',fimg1)
fimg2 = img1 - img2
plt.imshow(fimg2)
plt.show()
#saving
cv2.imwrite('output.jpg',fimg2)
fimg3 = img1 * img2
plt.imshow(fimg3)
plt.show()
#saving
cv2.imwrite('output.jpg',fimg3)
fimg4 = img1 / img2
plt.imshow(fimg4)
plt.show()
#saving
cv2.imwrite('output.jpg',fimg4)
![download](https://user-images.githubusercontent.com/98145574/175256482-a5175c37-c4b1-430d-8391-310aa90fa583.png)
![download](https://user-images.githubusercontent.com/98145574/175256520-c86958a5-9ca6-497e-abdf-2336552375a0.png)
![download](https://user-images.githubusercontent.com/98145574/175256547-9ff23830-b8be-4137-8d4e-e42dd9d3a2e6.png)

C:\Users\Central Computer Lab\AppData\Local\Temp\ipykernel_204\624335271.py:26: RuntimeWarning: divide by zero encountered in true_divide
  fimg4 = img1 / img2
C:\Users\Central Computer Lab\AppData\Local\Temp\ipykernel_204\624335271.py:26: RuntimeWarning: invalid value encountered in true_divide
  fimg4 = img1 / img2
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

![download](https://user-images.githubusercontent.com/98145574/175256682-8ae60fc8-56f5-401f-9812-05fad778a337.png)

program to create an image using 2D array.
import cv2 as c
import numpy as np
from PIL import Image
array=np.zeros([100,200,3],dtype=np.uint8)
array[:,:100]=[255,130,0]
array[:,100:]=[0,0,255]
img=Image.fromarray(array)
img.save('butterfly2.jpg')
img.show()
c.waitKey(0)

![image](https://user-images.githubusercontent.com/98145574/175262066-270b73b8-a719-4ac8-8ecc-a524e6d6f315.png)
