#ifndef NODE_H
#define NODE_H

#include <fstream>
#include<stdio.h>
#include<vector>
#include<list>
#include <opencv.hpp>

#include <time.h>
#include <math.h>
#include <cmath>
#include<numeric>

using namespace std;
using namespace cv;

class Node{
private:
	//split parameter of node
	int x1;
	int x2;
	int y1;
	int y2;
	int d;
	float theta;
	float voting;

	//parameter of model
	int maxDepth;
	int minLeafSample;
	int minInfoGain;

	//status of node
	bool LeafFlag;
	int sample_num;
	int positive_num;
	int current_depth;
	float infoGain;
	float Entro;

	Node *leftchild;
	Node *rightchild;
	
public:
	Node(int curr_depth=0, int w_w=0, int maxD=0, int minL=0, float minInfo=0, int s_n=0, int ps_n=0);
	~Node();

	void setLeaf();
	inline bool isLeaf(){return LeafFlag;};

	float calculate_entropy(int sample_num, int positive_num);
	inline float get_infoGain(){return infoGain;};

	void split_Node(vector<Mat> &sample, vector<int> &label, int *ID);
	void save(ofstream &fout);
	void load(ifstream &fin);

	float predict(Mat &test_img);
};

inline int get_Sum(Mat &img, int x, int y, int d){
	return img.at<int>(x+d-1, y+d-1) + img.at<int>(x,y) - img.at<int>(x,y+d-1) - img.at<int>(x+d-1,y);
}
#endif//NODE_H