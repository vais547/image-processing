# image-processing

**1.develop a program to display grayscale image using read and write operations.**<br>
import cv2<br>
img=cv2.imread('flower2.jpg',0)<br>
cv2.imshow('flower2',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**output:**<br>
![image](https://user-images.githubusercontent.com/98145574/173815026-c72d45aa-07c3-4dab-81d2-6bd3ad7bf938.png)<br>

**2.develop a program to display the image using matplotlib**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image=cv2.imread('flower2.jpg')<br>
plt.imshow(image)<br>
plt.show()<br>
**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/173815741-b99f5bc0-0176-4da3-a1fa-d081a22caea8.png)<br>

**#3.develop a program to dispaly linear tranformation.**<br>
from PIL import Image<br>
original_Image=Image.open('flower2.jpg')<br>
rotate_image1=original_Image.rotate(180)<br>
rotate_image1.show()<br>
**output:**<br>
![image](https://user-images.githubusercontent.com/98145574/173816447-cb37138f-82d8-495b-ba48-40ebd93134da.png)<br>


**4.#develop a program to convert colour string into RGB colour values.**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>

**output:**<br>
(255, 255, 0)<br>
(255, 0, 0)<br>

**5.#write a pprogram to create image using colour spaces.**<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,0,0))<br>
img.show()<br>
**output:**<br>
![image](https://user-images.githubusercontent.com/98145574/173816816-bc8e51e4-1c96-4a87-aa6f-408ade55af47.png)<br>

**6.#develop a program to visualize images using various colours.**<br>
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

**output:**<br>

![download](https://user-images.githubusercontent.com/98145574/173817498-2cb91906-9749-4ad9-907f-9d420ff0b8f4.png)<br>
![download](https://user-images.githubusercontent.com/98145574/173817570-ab4b3782-e345-45f0-9441-2951ef91ff0c.png)<br>
![download](https://user-images.githubusercontent.com/98145574/173817632-60c12506-8db8-48fa-acc8-47ba0853323e.png)<br>

**7.write a program to display the image attributes.**<br>
image=Image.open('flower2.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("mode:",image.mode)<br>
print("size:",image.size)<br>
print("width:",image.width)<br>
print("height:",image.height)<br>
image.close();<br>

**output:**<br>
Filename: flower2.jpg<br>
Format: JPEG<br>
mode: RGB<br>
size: (239, 211)<br>
width: 239<br>
height: 211<br>

**8. write a program to resize the original image.**<br>
import cv2<br>
img=cv2.imread('leaf3.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
imgresize=cv2.resize(img,(400,160))<br>
cv2.imshow('resized image',imgresize)<br>
print('Resized image legth width',imgresize.shape)<br>
cv2.waitKey(0)<br>

**output:**<br>
original image length width (177, 284, 3)<br>
Resized image legth width (160, 400, 3)<br>

![image](https://user-images.githubusercontent.com/98145574/174038962-29d31a06-a2f1-4820-b0f7-4ec79cbc62b8.png)<br>
![image](https://user-images.githubusercontent.com/98145574/174039141-54e97615-25c5-412b-a226-4bca6dd9a729.png)<br>

**9.write a program to convert the original image into gray scale and then to binary.**<br>
import cv2<br>
#read the image file.<br>
img=cv2.imread('leaf1.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>

**#gray scale**
img=cv2.imread("leaf1.jpg",0)<br>
cv2.imshow("gray",img)<br>
cv2.waitKey(0)<br>

**#binary image**<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**output:**<br>
![image](https://user-images.githubusercontent.com/98145574/174042660-ca59280b-f556-4665-8ea9-a97f5f4c571b.png)<br>
![image](https://user-images.githubusercontent.com/98145574/174042719-882b7e9d-93b7-4620-9e18-3e5f93580810.png)<br>
![image](https://user-images.githubusercontent.com/98145574/174042778-9661ede2-3c31-4807-9762-c44aa8107e57.png)<br>

**10.develop a program to readimage using url.**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://cdn.theatlantic.com/thumbor/viW9N1IQLbCrJ0HMtPRvXPXShkU=/0x131:2555x1568/976x549/media/img/mt/2017/06/shutterstock_319985324/original.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
output:<br>
![download](https://user-images.githubusercontent.com/98145574/175021544-cece9979-329f-4b72-bed4-772985eb149b.png)<br>

**11.write a program to mask and blur the image.**<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('fish.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/175018057-96e6bfaa-3c58-44cb-9994-999c5bebc362.png)<br>

hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1, 190, 200)<br>
dark_orange=(18, 255, 255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/175018146-f2f12bb8-5ee3-42c7-90ac-1dc5d76e2bff.png)<br>


light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/175018278-c425d835-3b28-4bf5-83ac-49870341ab72.png)<br>



final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>

![download](https://user-images.githubusercontent.com/98145574/175018525-539b2079-6856-4b31-8205-bef0dbf0f176.png)<br>


blur=cv2.GaussianBlur(final_result, (7,7), 0)<br>
plt.imshow(blur)<br>
plt.show()<br>

![download](https://user-images.githubusercontent.com/98145574/175018694-f2084ce4-3e1f-4f7f-858f-7691005efcf5.png)<br>


**12)Develop the program to change the image to different color spaces.**<br>
import cv2 <br>
img=cv2.imread("butterfly2.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLs image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**output**:<br>
![image](https://user-images.githubusercontent.com/98145574/175263583-3385ca4c-b120-4e3a-9340-67073ba818da.png)<br>


**13)write a program to perform arithmetic operations on images.**<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

**#Read file**<br>
img1=cv2.imread('butter1.jpg')<br>
img2=cv2.imread('butter2.jpg')<br>

**#numpy add**<br>
fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

**#saving**<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2 = img1 - img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

**#saving**<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

**#saving**<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>

**#saving**<br>
cv2.imwrite('output.jpg',fimg4)<br>

**output:**
![download](https://user-images.githubusercontent.com/98145574/175256482-a5175c37-c4b1-430d-8391-310aa90fa583.png)<br>
![download](https://user-images.githubusercontent.com/98145574/175256520-c86958a5-9ca6-497e-abdf-2336552375a0.png)<br>
![download](https://user-images.githubusercontent.com/98145574/175256547-9ff23830-b8be-4137-8d4e-e42dd9d3a2e6.png)<br>
![download](https://user-images.githubusercontent.com/98145574/175256682-8ae60fc8-56f5-401f-9812-05fad778a337.png)<br>

**14)program to create an image using 2D array.**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('butterfly2.jpg')<br>
img.show()<br>
c.waitKey(0)<br>

**output:**
![image](https://user-images.githubusercontent.com/98145574/175262066-270b73b8-a719-4ac8-8ecc-a524e6d6f315.png)<br>

**15)bitwise operation**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('butterfly1.jpg',1)<br>
image2=cv2.imread('butterfly1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/176424847-7b92b00b-b756-4316-a0c6-c0a55a2dd070.png)<br>

**16)blur_image**<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('flower4.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>

**#Gaussian blur**<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Guassian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>

**#Median blur**<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>

**#bilateral blur**<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**output:**<br>
![image](https://user-images.githubusercontent.com/98145574/176425100-09a0520d-f4d8-4fe8-b95a-77033434dc70.png)<br>
![image](https://user-images.githubusercontent.com/98145574/176425555-d23eceb1-5c15-4160-ac02-036517775d68.png)<br>
![image](https://user-images.githubusercontent.com/98145574/176425703-e7fafc97-e4e8-4266-a86e-3d7097e0b272.png)<br>
![image](https://user-images.githubusercontent.com/98145574/176425739-62c2dc8b-7130-44b0-85df-0e3f4b7d378b.png)<br>

**17)image_enhancement**<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('flower4.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>

image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>

image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>

image_contrasted=enh_col.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>   

**output:**<br>
![image](https://user-images.githubusercontent.com/98145574/176425910-64e41170-8550-457d-9f4d-957371312c2c.png)<br>
![image](https://user-images.githubusercontent.com/98145574/176425973-020cc07f-0188-4b06-b083-2f574dfb31e0.png)<br>
![image](https://user-images.githubusercontent.com/98145574/176426036-f09b7b18-619d-4470-8d1e-bd02714fca8b.png)<br>
![image](https://user-images.githubusercontent.com/98145574/176426074-3e6eb360-8322-4125-b607-1db65b3efe56.png)<br>
![image](https://user-images.githubusercontent.com/98145574/176426120-adc303ae-ff33-4168-9b24-8d0eee18081b.png)<br>

**18)morphological_operation.**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('butterfly4.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/176426378-7544e78c-0401-4313-bcd5-5223c38ee5dd.png)<br>

**19)write a program to**<br>
**i)Read the image,**<br>
**ii)write (save) the grayscale image and**<br>
**iii)dispaly the original iamge and grayscale iamge**<br>
**(Note:To save iamge to local storage using Python,we use cv2.imwrite() function on opencv library)**<br>

import cv2<br>
OriginalImg=cv2.imread('butterfly1.jpg')<br>
GrayImg=cv2.imread('butterfly1.jpg',0)<br>
isSaved=cv2.imwrite('F:\i.jpg' ,GrayImg)<br>
cv2.imshow('Dispaly Original Image',OriginalImg)<br>
cv2.imshow('Dispaly Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The image is succussfully saved.')<br>
    
**output:**<br>
The image is succussfully saved.<br>

![image](https://user-images.githubusercontent.com/98145574/178700980-46a134eb-5b39-4d65-8cc3-03b6d449730c.png)<br>

**20)slicing with backgroung**<br>

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

**output:**<br>

![download](https://user-images.githubusercontent.com/98145574/178705417-4683e919-a00d-4813-a2f5-49069c633a0e.png)<br>

**21)slicing without bakground**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing withot background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/178705575-919654fe-ff7a-4fc8-a00b-fb1e6bcb24a5.png)<br>

**22)analyse the image data using histogram.**<br>
import numpy as np<br>
import skimage.color<br>
import skimage.io as io<br>
import matplotlib.pyplot as plt<br>

**# Load the image**<br>
image = skimage.io.imread(fname="cats.jpg",as_gray=True)<br>
image1 = skimage.io.imread(fname="cats.jpg")<br>

fig, ax = plt.subplots()<br>
plt.imshow(image, cmap="gray")<br>
plt.show()<br>

fig, ax = plt.subplots()<br>
plt.imshow(image1, cmap="gray")<br>
plt.show()<br>

**#create the histogram**<br>
histogram, bin_edges=np.histogram(image,bins=256,range=(0,1))<br>

**#configue and draw the histogram figure**<br>
plt.figure()<br>
plt.title("grayscale histogram ")<br>
plt.xlabel("grayscale value")<br>
plt.ylabel("pixel count")<br>
plt.xlim([0.0,1.0])<br>
plt.plot(bin_edges[0:-1],histogram)<br>
plt.show()<br>

**output:**
![download](https://user-images.githubusercontent.com/98145574/178966950-1b2b66bd-df71-489e-bafa-b1f9e60aa9a2.png)<br>
![download](https://user-images.githubusercontent.com/98145574/178966972-a5c000b5-9688-4e7b-9517-ee3c970e4ef7.png)<br>
![download](https://user-images.githubusercontent.com/98145574/178966994-832952d8-f8de-4a4c-8dfe-aa095acdaf72.png)<br>


**#program to perfor the basic image data analysis using intensity transformation.**<br>
1)image negative<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('cat4.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/179966611-b95b272f-105a-4931-9dc1-98bd10bc5ca5.png)<br>

negative=255- pic #neg=(L-1)-img<br>
plt.figure(figsize= (6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/179966657-1ac5b789-cd86-4c32-ac76-d35e44249f4b.png)<br>

**2)log transformation**<br>

%matplotlib inline<br>

import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

pic=imageio.imread('cat4.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>

max=np.max(gray)<br>

def log_transform():<br>
   return(255/np.log(1+max))* np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/179966805-f1c09579-28d8-467e-bf98-2d93adfa0a47.png)<br>

**3)gamma correction**<br>

import imageio<br>
import matplotlib.pyplot as plt<br>

**# gamma encoading**<br>
pic=imageio.imread('cat4.jpg')<br>
gamma=2.2#Gamma<1~dark;gamma>1~bright<br>

gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
output:<br>
![download](https://user-images.githubusercontent.com/98145574/179966932-923b1f91-2917-48ca-93d0-84b08774b8a6.png)<br>

**2)program to perform the basic image manipulation.**<br>
**a)sharpness**<br>
**#image sharpen**<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>

**#load the image**<br>
my_image=Image.open('cat4.jpg')<br>

**#use sharpen function**<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>

**#save the image**<br>
sharp.save('F:/cat.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>

output:<br>
![download](https://user-images.githubusercontent.com/98145574/179967260-f80ffb90-de52-4917-952d-c79cbe39e5b5.png)<br>

**b)flipping**<br>
**#image flip**<br>
import matplotlib.pyplot as plt<br>

**#load the image**<br>
img=Image.open('cat4.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

**#use the flip function**<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>

**#save the image**<br>
flip.save('f:/pussy.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/179967424-d166e323-f2be-4617-b2ca-36908e6764a3.png)<br>
![download](https://user-images.githubusercontent.com/98145574/179967434-7b626f72-a57c-4e87-9ac6-011fb191501c.png)<br>

**c)croping**<br>
**#Importing Image class from PIL module**<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>

**#opens a image in rgb mode**<br>
im=Image.open('cat4.jpg')<br>

**#size of the image in pixels(size of original image)**<br>
**#(This is not mandatory)**<br>
width,height=im.size<br>

**#cropped image of above dimension**<br>
**#(it will not change original image)**<br>
im1=im.crop((50,30,230,160))<br>

**#shows the image in image viewer**<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

**output:**<br>
![download](https://user-images.githubusercontent.com/98145574/179967533-29846537-bcc9-40a4-95f5-ab0ea7135dab.png)<br>


**#edge detection using prewitt operartor**
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('shapes.jpeg',)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian=cv2.GaussianBlur(gray,(3,3),0)

#prewitt
kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely=np.array([[-1,0,-1],[-1,0,1],[-1,0,1]])
img_prewittx=cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty=cv2.filter2D(img_gaussian, -1, kernely)

cv2.imshow("original image",img)
cv2.imshow("prewitt x", img_prewittx)
cv2.imshow("prewitt y", img_prewitty)
cv2.imshow("prewitt", img_prewittx+img_prewitty)
cv2.waitKey()
cv2.destroyAllWindows()

output:
![image](https://user-images.githubusercontent.com/98145574/187904159-fe2e89b1-663b-4412-be7a-62c867d6833d.png)


**#roberts edge detection-roberts cross operator**
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
roberts_cross_v=np.array([[1,0],
                        [0,-1]])
roberts_cross_h=np.array([[0,1],
                        [-1,0]])
img=cv2.imread('shapes.jpeg',0).astype('float64')
img/=255.0
vertical=ndimage.convolve(img, roberts_cross_v )
horizontal=ndimage.convolve(img, roberts_cross_h )

edged_img=np.sqrt(np.square(horizontal)+np.square(vertical))
edged_img*=255
cv2.imwrite("output.jpg",edged_img)
cv2.imshow("OutputImage", edged_img)
cv2.waitKey()
cv2.destroyAllWindows()
output:
![image](https://user-images.githubusercontent.com/98145574/187904341-83a78e0f-4716-4455-893a-bb0a719b7973.png)
