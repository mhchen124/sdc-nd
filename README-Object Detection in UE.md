###########################################
# Object Detection in UE - Project Write-up

This is a brief write-up for the project of "Object Detection in Urban Environment".
The file included for this project are:
- 1_train_fasterrcnn-updated.ipynb
- 1_train_mobilenet-updated.ipynb
- 1_train_resnet-updated.ipynb
- faster_rcnn_resnet50.config
- mobilenet.config
- resnet50.config
- output.avi

Some observations while doing the project:

* Faster RCNN model seems to train the fastest, training time is 1337 sec (while efficientnet needs nearly double that amount?)
* Had OOM error while running MobileNet and ResNet50 amid the training phase (tried to use different EC2 instance type but it seems we are restricted to only use ml.t3.2xlarge), so unable to get more detailed performance metrics on these two.
* Faster RCNN seems to perform better here, 

with its training loss as:
========================

I0321 19:31:25.993554 139798555100992 model_lib_v2.py:708] {'Loss/BoxClassifierLoss/classification_loss': 0.13113965,
 'Loss/BoxClassifierLoss/localization_loss': 0.2547032,
 'Loss/RPNLoss/localization_loss': 0.23501393,
 'Loss/RPNLoss/objectness_loss': 0.06719543,
 'Loss/regularization_loss': 0.0,
 'Loss/total_loss': 0.6880522,
 'learning_rate': 0.06420001}

and Eval Loss as:
=================

I0321 19:33:37.039008 139942282868544 model_lib_v2.py:1015] Eval metrics at step 2000
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP: 0.014993
I0321 19:33:37.051634 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP: 0.014993
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP@.50IOU: 0.033763
I0321 19:33:37.053084 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP@.50IOU: 0.033763
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP@.75IOU: 0.012015
I0321 19:33:37.054572 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP@.75IOU: 0.012015
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (small): 0.006119
I0321 19:33:37.055983 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (small): 0.006119
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (medium): 0.052663
I0321 19:33:37.057398 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (medium): 0.052663
INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (large): 0.105345
I0321 19:33:37.058852 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (large): 0.105345
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@1: 0.006214
I0321 19:33:37.060263 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@1: 0.006214
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@10: 0.019045
I0321 19:33:37.061743 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@10: 0.019045
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100: 0.026994
I0321 19:33:37.063163 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100: 0.026994
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (small): 0.014208
I0321 19:33:37.064594 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (small): 0.014208
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (medium): 0.064215
I0321 19:33:37.066019 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (medium): 0.064215
INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (large): 0.250285
I0321 19:33:37.067497 139942282868544 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (large): 0.250285
INFO:tensorflow:#011+ Loss/RPNLoss/localization_loss: 1.018562
I0321 19:33:37.068623 139942282868544 model_lib_v2.py:1018] #011+ Loss/RPNLoss/localization_loss: 1.018562
INFO:tensorflow:#011+ Loss/RPNLoss/objectness_loss: 0.239952
I0321 19:33:37.069841 139942282868544 model_lib_v2.py:1018] #011+ Loss/RPNLoss/objectness_loss: 0.239952
INFO:tensorflow:#011+ Loss/BoxClassifierLoss/localization_loss: 0.115643
I0321 19:33:37.070960 139942282868544 model_lib_v2.py:1018] #011+ Loss/BoxClassifierLoss/localization_loss: 0.115643
INFO:tensorflow:#011+ Loss/BoxClassifierLoss/classification_loss: 0.125752
I0321 19:33:37.072101 139942282868544 model_lib_v2.py:1018] #011+ Loss/BoxClassifierLoss/classification_loss: 0.125752
INFO:tensorflow:#011+ Loss/regularization_loss: 0.000000
I0321 19:33:37.073292 139942282868544 model_lib_v2.py:1018] #011+ Loss/regularization_loss: 0.000000
INFO:tensorflow:#011+ Loss/total_loss: 1.499909
I0321 19:33:37.074490 139942282868544 model_lib_v2.py:1018] #011+ Loss/total_loss: 1.499909

* Noticed that some of the loss metrics in training is larger than that in validation, which is not normal. My guess is that maybe the training set is not large enough?
* Further improvement can be done through argumentation, as our training dataset seems not that big (?) We can introduce V/H flipping, scaling up/down, rotating, changing colormap, histogram, etc to generate lots of argumented images.

How Faster RCNN Works
=====================

Faster RCNN architecture can be depicted as follow (courtesy of Neeraj Krishna):
<img width="710" alt="Faster Rcnn Architecture" src="https://user-images.githubusercontent.com/1509571/227423012-cc418644-174c-4369-9869-e9a7a0aa75a5.png">

The Faster RCNN is a 2-stage architecture, where the first stage is to propose a number of candidate object region (with different sizes and aspect ratios), and the second stage has two sibling fully connected layers - one for classifying whether the region box contains the object or not, and the other regress on the bonding box coordinates to get the best-fit bonding box for the object. It excel in the performance speed (suitable for real-time object detection applications), and the major gain of the speed comes from the idea of sharing the CNN engine which is used both for region proposal and region classification - so that wights does not need to incur heavy repetitive computations. 

