import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from skimage.transform import resize


# READ THE IMAGES
input_dir = "D:\Artifical Intelligence\Image classification\images"

Items=['Bottle detected','no bottle']

data =[]
label=[]
for Item in Items:
    for file in os.listdir(os.path.join(input_dir,Item)):
        img_path=os.path.join(input_dir,Item,file)
        img=cv2.imread(img_path)
        img1=cv2.resize(img,(200,200))
        data.append(img1.flatten())
        label.append(Item)

# PREPROCESSING THE DATA
data=np.asarray(data)
label=np.asarray(label)

# TRAIN TEST SPLIT
x_train,x_test, y_train,y_test=train_test_split(data,label,test_size=0.1,shuffle=True,stratify=label)

label_encoder=LabelEncoder()
y_train_encoded=label_encoder.fit_transform(y_train)
y_test_encoded=label_encoder.transform(y_test)

# TRAINING THE MODEL
svm_classifier=SVC(kernel='linear',C=1.0)
svm_classifier.fit(x_train,y_train_encoded)

y_predicted_encoded=svm_classifier.predict(x_test)

y_predicted=label_encoder.inverse_transform(y_predicted_encoded)

accuracy=accuracy_score(y_test,y_predicted)
print("Accuracy:",accuracy)

cap = cv2.VideoCapture(4747)  # 0 corresponds to the default webcam
address = "https://192.168.137.112:8080/video"
cap.open(address)


while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # Display the frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Save the captured frame as an image file
        cv2.imwrite('captured_image.jpg', frame)
        print("Image captured!")
        break




cap.release()
cv2.destroyAllWindows()

d=cv2.imread("D:\Artifical Intelligence\Image classification\captured_image.jpg")
d1=cv2.resize(d,(200,200))
c=d1.flatten()
d2=c.reshape(1,-1)

e=svm_classifier.predict(d2)
f=label_encoder.inverse_transform(e)

print(f)





