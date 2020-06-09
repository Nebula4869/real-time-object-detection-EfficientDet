# real-time_object_detection_EfficientDet
Real-time object detection using COCO-pretrained EfficientDet under Pytorch and TensorFlow
### Environment

- python==3.6.5
- torch==1.5.0
- torchvision==0.6.0
- tensorflow==1.15.0
- opencv-python


### Getting Started

1. Download checkpoint and convert to .pb model files from this [repo](https://github.com/google/automl/tree/master/efficientdet).
2. Download Pytorch .pth model files from this [repo](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 
3. Run "demo_pytorch.py" to detect with Pytorch model.
4. Run "demo_tf.py" to detect with TensorFlow frozen model.

