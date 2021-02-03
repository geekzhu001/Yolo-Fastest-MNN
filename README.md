# Yolo-Fastest-MNN
  Run on :raspberry pi 4B 2G 
  
  Input size :320*320
  
  Average inference time : 0.035s 
  
## How to use
1.
* https://github.com/alibaba/MNN 

Compile mnn library yourself  and replace lib/libMNN.so
2.
* https://github.com/dog-qiuqiu/MobileNet-Yolo#darknet2caffe-tutorial   

Convert model to mnn type
3.

sudo apt install libopencv-dev

4.
cd Yolo-Fastest-MNN

mkdir build && cd build && cmake ..

make

./yolo

##  Reference
* model from :https://github.com/dog-qiuqiu/Yolo-Fastest
* nms:https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
