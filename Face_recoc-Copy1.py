#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


get_ipython().system('pip install CMake')
get_ipython().system('pip install dlib')


# In[3]:


get_ipython().system('pip install face_recognition')


# In[ ]:


import oss2

endpoint = 'http://oss-me-central-1.aliyuncs.com' # Suppose that your bucket is in the Hangzhou region.

auth = oss2.Auth('LTAI5tRttayv3RA1JM9GY3E7', 'PfUlXeodsPgILlMAHqGM6ivapMVKaX')
bucket = oss2.Bucket(auth, endpoint, 'images-tensor')

# The object key in the bucket is story.txt
key = 'images-tensor/person'

# Traverse all objects in the bucket
for object_info in oss2.ObjectIterator(bucket):
    print(object_info.key)


# In[4]:


import cv2
import face_recognition


# In[5]:


video_capture = cv2.VideoCapture(0)


# In[6]:


face_locations = []


# In[ ]:





# In[ ]:


# Grab a single frame of video
ret, frame = video_capture.read()
# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb_frame = frame[:, :, ::-1]


# In[ ]:


face_locations = face_recognition.face_locations(rgb_frame)


# In[ ]:


for top, right, bottom, left in face_locations:
# Draw a box around the face
   cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
   cv2.imshow('Video', frame)
   if cv2.waitKey(1) & 0xFF == ord("q"):
       break


# In[ ]:


video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




