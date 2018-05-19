#include "Node.h"

Node::Node(int curr_depth, int w_w, int maxD, int minL, float minInfo, int s_n, int ps_n){
	//parameter of model
	maxDepth = maxD;
	minLeafSample = minL;
	minInfoGain = minInfo;

	x1 = 0;
	x2 = 0;
	y1 = 0;
	y2 = 0;
	theta = 0;
	d = w_w;
	voting = 2.0;

	//status of node
	LeafFlag = false;
	sample_num = s_n;
	positive_num = ps_n;
	current_depth = curr_depth;
	infoGain = 0;
	Entro = calculate_entropy(sample_num, positive_num);

	leftchild = NULL;
	rightchild = NULL;
	//cout << "Entro = " << Entro << endl;
}

Node::~Node(){
	if(!LeafFlag){
		delete leftchild;
		leftchild = NULL;
		delete rightchild;
		rightchild = NULL;
	}
}

void Node::setLeaf(){
	//set the flag
	LeafFlag = true;
	
	//calculate the vote
	voting = 1.0*positive_num/sample_num;

	/*if(p_2 > p_1)
		voting = 0;
	else
		voting = 1;*/
	//cout << "This node is setted to be leaf, depth = " << current_depth << endl;
	//cin.get();
}

float Node::calculate_entropy(int sample_n, int positive_n){
	float entropy = 0;

	if(sample_n != 0){
		float pp = 1.0 * positive_n / sample_n;						//positive%
		float np = 1.0 * (sample_n - positive_n) / sample_n;		//negtive%

		if(pp!=0 && np !=0)
			entropy = -1.0*pp*log(1.0*pp)/log(2.0) - 1.0*np*log(1.0*np)/log(2.0);
	}
	return entropy;
}

void Node::split_Node(vector<Mat> &sample, vector<int> &label, int *ID){
	/*if(Entro==0 || current_depth>maxDepth || sample_num<minLeafSample){
		setLeaf();
		return;
	}*/

	if(Entro==0 || sample_num<minLeafSample){
		delete ID;
		ID = NULL;
		setLeaf();
		return;
	}

	if(current_depth>maxDepth){
		cout << "insufficient depth" << endl;
		delete ID;
		ID = NULL;
		setLeaf();
		return;
	}

	srand (time(NULL));

	int r = sample[0].rows;
	int c = sample[0].cols;

	int left_num = 0;
	int left_positive = 0;
	int right_num = 0;
	int right_positive = 0;

	for(int i=0; i<100; i++){
		//randomly choose the parameter
		int d_tmp = rand() % (min(sample[0].cols, sample[0].rows)) + 1;
		int x1_tmp = rand() % (c-d_tmp+1);
		int y1_tmp = rand() % (r-d_tmp+1);
		int x2_tmp = rand() % (c-d_tmp+1);
		int y2_tmp = rand() % (r-d_tmp+1);

		//randomly choose one positive and one negative sample to calculate the theta
		int ss_index1 = rand() % sample_num;
		//float theta_tmp = mean(imgList[ss_index1](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0] - mean(imgList[ss_index1](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0];
		//int theta_tmp = get_Sum(imgList[ss_index1], x1_tmp, y1_tmp, d_tmp) - get_Sum(imgList[ss_index1], x2_tmp, y2_tmp, d_tmp);
		int theta_tmp = get_Sum(sample[ID[ss_index1]], x1_tmp, y1_tmp, d_tmp);
		while(true){
			int ss_index2 = rand() % sample_num;
			if((label[ID[ss_index1]] + label[ID[ss_index2]]) == 1){
				//theta_tmp += (mean(imgList[ss_index2](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0] - mean(imgList[ss_index2](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0]);
				//theta_tmp += (get_Sum(imgList[ss_index2], x1_tmp, y1_tmp, d_tmp) - get_Sum(imgList[ss_index2], x2_tmp, y2_tmp, d_tmp));

				theta_tmp += get_Sum(sample[ID[ss_index2]], x1_tmp, y1_tmp, d_tmp);
				theta_tmp /= 2;
				break;
			}
		}

		//split the node under this set of parameter
		int left_num_tmp = 0;
		int left_positive_tmp = 0;
		int right_num_tmp = 0;
		int right_positive_tmp = 0;
		

		for(int p=0; p<sample_num; p++){
			//cout << "img size " << imgList[p].cols <<" " << imgList[p].rows << endl;

			//float mean1 = mean(imgList[p](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0];
			int sum1 = get_Sum(sample[ID[p]], x1_tmp,y1_tmp,d_tmp);
			//cout << 11 << endl;
			//float mean2 = mean(imgList[p](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0];
			//int sum2 = get_Sum(imgList[p], x2_tmp,y2_tmp,d_tmp);
			int sum2 = 0;

			//if(mean1-mean2>theta_tmp){
			if(sum1-sum2>theta_tmp){
				//leftImg_tmp.push_back(imgList[p]);
				//leftLabel_tmp.push_back(imgLabel[p]);
				left_num_tmp++;
				left_positive_tmp += label[ID[p]];
			}
			else{
				//rightImg_tmp.push_back(imgList[p]);
				//rightLabel_tmp.push_back(imgLabel[p]);
				right_num_tmp++;
				right_positive_tmp += label[ID[p]];
			}
		}

		//calculate the current information gain
		float infoGain_new = Entro - (left_num_tmp*calculate_entropy(left_num_tmp, left_positive_tmp) 
			+ right_num_tmp*calculate_entropy(right_num_tmp, right_positive_tmp))/sample_num;
		//cout << "new gain = " << infoGain_new << endl;
		//cin.get();
		
		if(infoGain_new > infoGain){	
			//cout << "tmp : " << x1_tmp << " " << y1_tmp << " "  << x2_tmp << " "  << y2_tmp << " "  << " " << d << endl;
			infoGain = infoGain_new;
			x1 = x1_tmp;
			x2 = x2_tmp;
			y1 = y1_tmp;
			y2 = y2_tmp;
			d = d_tmp;
			theta = theta_tmp;

			left_num = left_num_tmp;
			left_positive = left_positive_tmp;
			right_num = right_num_tmp;
			right_positive = right_positive_tmp;

			//cout << "new gain = " << infoGain << endl;
		}
	}

	if(infoGain<minInfoGain){
		setLeaf();
		return;
	}

	int *left_ID = new int[left_num];
	int *right_ID = new int[right_num];
	int left_current = 0;
	int right_current = 0;

	for(int p=0; p<sample_num; p++){
		//cout << "location: " << x1 << " " << y1 << " " << x2 << " " << y2 << " " << d << endl;
		//float mean1 = mean(imgList[p](Rect(x1,y1,d,d)))[0];
		//float mean2 = mean(imgList[p](Rect(x2,y2,d,d)))[0];

		int sum1 = get_Sum(sample[ID[p]], x1,y1,d);
		//int sum2 = get_Sum(sample[ID[p]], x2,y2,d);
		int sum2 = 0;

		if(sum1-sum2>theta){
			left_ID[left_current++] = ID[p];
		}
		else{
			right_ID[right_current++] = ID[p];
		}
	}

	delete ID;
	ID = NULL;

	//create the left and right child node
	leftchild = new Node(current_depth+1, d, maxDepth, minLeafSample, minInfoGain, left_num, left_positive);
	rightchild = new Node(current_depth+1, d, maxDepth, minLeafSample, minInfoGain, right_num, right_positive);

	//split the child node recursively
	leftchild->split_Node(sample, label, left_ID);
	//delete left_ID;
	//left_ID = NULL;
	rightchild->split_Node(sample, label, right_ID);
	//delete right_ID;
	//right_ID = NULL;
}

void Node::save(ofstream &fout){
	fout << x1 << " " << x2 << " " << y1 << " " << y2 << " " << d << " " << theta << " " << voting << " " << LeafFlag << endl;
	if(!LeafFlag){
		leftchild->save(fout);
		rightchild->save(fout);
	}
	return;
}

void Node::load(ifstream &fin){
	fin >> x1 >> x2 >> y1 >> y2 >> d >> theta >> voting >> LeafFlag;
	//cout << "x1:" << x1 << " x2:" << x2 << " y1:" << y1 << " y2:" << y2 << " d:" << d << " theta:" << theta << " voting:" << voting << " LeafFlag:" << LeafFlag << endl;
	//cin.get();
	
	if(!LeafFlag){
		leftchild = new Node();
		leftchild->load(fin);
		rightchild = new Node();
		rightchild->load(fin);
	}
}

float Node::predict(Mat &test_img){
	while(!LeafFlag){
		//float mean1 = mean(test_img(Rect(x1,y1,d,d)))[0];
		//float mean2 = mean(test_img(Rect(x2,y2,d,d)))[0];
		//if(mean1-mean2>theta)
		//cout << "x1 = " << x1 << " y1 = " << y1 << " x2 = " << x2 << " y2 = " << y2 << " d = " << d << " col = " << test_img.cols << " row = " << test_img.rows << endl;
		//cin.get();
		int sum1 = get_Sum(test_img, x1,y1,d);
		//int sum2 = get_Sum(test_img, x2,y2,d);
		int sum2 = 0;

		if(sum1-sum2>theta)
			return leftchild->predict(test_img);
		else
			return rightchild->predict(test_img);
	}

	//if(voting != 0 && voting != 1){
	if(voting < 0 || voting > 1){
		cout << "vote error" << endl;
		cin.get();
	}
	else{
		//cout << "one prediction ended" << endl;
		return voting;
	}
}