#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

using namespace std;
using namespace MNN;

#define hard_nms 1
#define blending_nms 2

const float mean_vals[3] = {0.f, 0.f, 0.f};
const float norm_vals[3] = {1/256.0f, 1/256.0f, 1/256.0f};
const float thres[2] = {0.6,0.5} ;
const int BIAS_W[6] = {26, 67, 72, 189, 137, 265};  
const int BIAS_H[6] = {48, 84, 175, 126, 236, 259};
const int INPUT_SIZE = 320;
const int CLASS_NUM = 20;

struct BBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
};

float sigmod(float x){
	return 1.0 / (1.0 + exp(-x));
}

int topK(float * labels,int size){
	int index = 0;
	float maxid = labels[0];
	
	for(int i = 1;i < size;i++){

		if(labels[i]>maxid){
			index = i;
			maxid = labels[i];
		}
	}
	return index;
}



void postprocess(std::vector<MNN::Tensor*> output,std::vector<BBox> &boxes,int iw,int ih){

	//fpn 图像金字塔
	for(int fpn = 0;fpn < 2;fpn++){

		int channel = output[fpn]->channel();
		int width = output[fpn]->width();
		int fmscale = width*width;
		int fmsize = 5+CLASS_NUM;

		//dect_box_handle {"10":0.6, "20":0.5}
		//32倍下采样 fm_size = input_size/32	
		for(int c = 0;c < fmscale;c++){
			int x = c % width;
			int y = c / width;
			// 3种anchor box
			for(int s = 0;s < 3;s++){
				float scores = sigmod( output[fpn]->host<float>()[ c * channel + s * fmsize + 4 ] );
				if(scores > thres[fpn]){
		
				//cout<<scores<<" "<<c<< " x: "<<x<<"   y: "<<y<<" anchor: "<<s<<endl;
				BBox rect;
				float xx = ((sigmod(output[fpn]->host<float>()[c * channel + s * fmsize ]) + x) / width) * iw;
				float yy = ((sigmod(output[fpn]->host<float>()[c * channel + s * fmsize +1]) + y) / width) * ih;
				float w = ((BIAS_W[fpn * 3 + s] * exp(output[fpn]->host<float>()[c * channel + s * fmsize + 2]))/INPUT_SIZE) * iw;
				float h = ((BIAS_H[fpn * 3 + s] * exp(output[fpn]->host<float>()[c * channel + s * fmsize + 3]))/INPUT_SIZE) * ih;
				rect.x1 = int(xx - w * 0.5);
				rect.x2 = int(xx + w * 0.5);
				rect.y1 = int(yy - h * 0.5);
				rect.y2 = int(yy + h * 0.5);
				rect.score = scores;
				rect.label = topK(output[fpn]->host<float>()+c * channel + s * 25 +5,CLASS_NUM);
				boxes.push_back(rect);
				//cout<<rect.x1<< " "<<rect.y1<<" "<<rect.x2<<" "<<rect.y2<<endl;
				}
			}
		}
	}
}

void nms(std::vector<BBox> &input,std::vector<BBox> &output,float iou_threshold,int type){
	if(input.size()!=0){
		std::sort(input.begin(), input.end(), [](const BBox &a, const BBox &b) { return a.score > b.score; });
    
    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<BBox> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);

                break;
            }
            case blending_nms: {


                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                BBox rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }



	}
}

int main(){

	//cv::Mat row_img = cv::imread("../1.jpg");
	//int iw = row_img.cols;
	//int ih = row_img.rows;
	cv::Mat row_image;
	cv::Mat image;
	//cv::resize(row_img, image,cv::Size(INPUT_SIZE,INPUT_SIZE));

	const char *mnn_path = "../model/yolofastest.mnn";
	std::shared_ptr<MNN::Interpreter> interpreter(MNN::Interpreter::createFromFile(mnn_path));

	//int num_thread = 4 ;
	MNN::ScheduleConfig config;  
	//config.numThread = num_thread;

	MNN::BackendConfig backendConfig;
	
	backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 1;
	
	config.backendConfig = &backendConfig;

	auto session = interpreter->createSession(config);   

	auto input_tensor = interpreter->getSessionInput(session, nullptr);
	
	interpreter->resizeTensor(input_tensor, {1,3,INPUT_SIZE,INPUT_SIZE}); //输入tensor格式NCHW

	interpreter->resizeSession(session);

    cv::VideoCapture cap;
	cap.open(0,200);	
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	int iw = 640;
	int ih = 480;
	while (1)
	{
		/* code */
	
	cap >> row_image;
	cv::resize(row_image,image,cv::Size(320,320));
   	std::shared_ptr<MNN::CV::ImageProcess> pretreat(
       MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,norm_vals, 3));
    	pretreat->convert(image.data, 320, 320,0, input_tensor);

       
	auto start =chrono::steady_clock::now();

	interpreter->runSession(session);

	auto end =chrono::steady_clock::now();
	chrono::duration<double> elapsed = end - start;
	std::cout<<"inference time: " <<elapsed.count()<<endl;
	//cout<<"fps: " <<1/elapsed.count()<<endl;
	std::string scores = "layer125-conv";
	std::string scores2 = "layer115-conv";
       
	MNN::Tensor *tensor_scores = interpreter->getSessionOutput(session, scores.c_str());
	MNN::Tensor *tensor_scores2 = interpreter->getSessionOutput(session, scores2.c_str());
        
  	auto tensor_scores_host = new Tensor(tensor_scores, MNN::Tensor::TENSORFLOW);//NHWC
   	tensor_scores->copyToHostTensor(tensor_scores_host);
   	
   	auto tensor_scores_host2 = new Tensor(tensor_scores2, MNN::Tensor::TENSORFLOW);
   	tensor_scores2->copyToHostTensor(tensor_scores_host2);
  
	std::vector<MNN::Tensor *> output_tensor;
	output_tensor.push_back(tensor_scores_host);
	output_tensor.push_back(tensor_scores_host2);

	std::vector<BBox> row_boxes,dest_boxes;

	postprocess(output_tensor,row_boxes,iw,ih);
	nms(row_boxes,dest_boxes,0.35,blending_nms);

 	auto end2 =chrono::steady_clock::now();
	chrono::duration<double> elapsed2 = end2 - end;
	cout<<"postprocess time: " <<elapsed2.count()<<endl;

	for(auto box : dest_boxes){
		cv::Point pt1(box.x1, box.y1);
		cv::Point pt2(box.x2, box.y2);
		cv::rectangle(row_image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("Yolo-Fastest", row_image);
	cv::waitKey(1);

	}

}
