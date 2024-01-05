# Recognizing Students and Detecting Student Engagement with Image Processing
## Problem Motivation
With COVID-19, formal education was interrupted in all countries and the importance of
distance learning has increased. It is possible to teach any lesson with various communication
tools but it is difficult to know how far this lesson reaches the students. In this project, we aim to
monitor the students in a classroom or in front of the computer with a camera, recognizing their
faces, their head poses, and scoring their distraction to detect student engagement based on
their head poses and Eye Aspect Ratios. The output of this project will be attendance and
emotions records for each day. This data can be used for further data analysis.
## Non-primitive Functions
- Histogram Equalization
- Gamma Correction
- Rotating Mask
- Median Filter
- Sobel Operator
- Laplace Operator
- Canny Edge
- Morphological Opening
- Morphological Closing
- LBP (Local Binary Pattern)
- Histogram of Gradients (HoG)
- Haar Cascade
## Block Diagram
![image](https://github.com/ahmedoshelmy/Face-Emotion-Vision/assets/77215230/c7588a67-9f94-417d-84fb-2366dfa89439)

## Additional Comments
Lighting conditions pose a challenge to our system. To solve this, we will use pre-processing
techniques that will normalize contrast, standardize brightness conditions and make the images
more consistent. These include histogram equalization and gamma correction.  

Another challenge facing us is that of face orientations. For face detection, we will use the
histogram of gradients algorithm, a gradient-orientation based algorithm that is effective in
handling different face orientations. The algorithm divides the image into small regions in order
to generate local descriptors for edges and contours. In addition, the histogram is often
normalized to reduce the effects of lighting and contrast. For face recognition, we will use local
binary pattern (LBP), a statistical texture measure for feature extraction that is robust to face
orientations. LBP measures the relationships between the intensities of neighboring pixels in a
circular pattern around the central pixel. This circular neighborhood structure helps in capturing
texture information that remains relatively stable even if the image is slightly rotated.  

We may use edge detection techniques to increase the accuracy of our face recognition model.
We will apply the machine learning model on the images before and after face recognition using
morphological and texture analysis to compare accuracies.  

We may use data augmentation if the data was limited or if validation wasn’t accurate enough.
This will improve the system’s accuracy when dealing with different face alignments.  

## Sample Images
### Engaged
#### Confused
![image](https://github.com/ahmedoshelmy/Face-Emotion-Vision/assets/77215230/c7219c0f-1c56-4439-af44-f194e21f4c24)

#### Engaged
![image](https://github.com/ahmedoshelmy/Face-Emotion-Vision/assets/77215230/68c2003b-7335-4228-b67c-f0b671c885b4)

#### Frustrated
![image](https://github.com/ahmedoshelmy/Face-Emotion-Vision/assets/77215230/27aafaa5-fdd3-4bb1-8a6c-8ce9afa07992)

### Not Engaged
#### Bored
![image](https://github.com/ahmedoshelmy/Face-Emotion-Vision/assets/77215230/249fd247-38c7-48ef-831a-92e82ab5b177)

#### Drowsy
![image](https://github.com/ahmedoshelmy/Face-Emotion-Vision/assets/77215230/730a4641-d9b7-401c-8842-e3a6849ca43b)

#### Looking Away
![image](https://github.com/ahmedoshelmy/Face-Emotion-Vision/assets/77215230/7dd2025e-3ccf-4cd4-9bc7-c2ab42b5c048)

## References
- Uçar, M.U.; Özdemir, E. Recognizing Students and Detecting Student Engagement with
Real-Time Image Processing. Electronics 2022, 11, 1500.
https://doi.org/10.3390/electronics11091500
- Xuran et al. A co-training approach to automatic face recognition. EUSIPCO 2011.
https://www.eurasip.org/Proceedings/Eusipco/Eusipco2011/papers/1569427521.pdf
- Dey, J. (2021, September). Student-engagement, Version 1. Retrieved November 4, 2023 from https://www.kaggle.com/datasets/joyee19/studentengagement
