#include "RandomForest.h"

RandomForest::RandomForest(int w_w, int t_n, int s_n, int maxD, int minL, float minInfo){
	window_width = w_w;

	tree_num = t_n;
	maxDepth = maxD;
	minLeafSample = minL;
	minInfoGain = minInfo;
	root_list = new Node*[tree_num];

	sample_num = s_n;

}

RandomForest::~RandomForest(){
	for(int i=0; i<tree_num; i++){
		if(root_list[i] != NULL){
			delete root_list[i];
			root_list[i] = NULL;
		}
	}

	delete[] root_list;
	root_list = NULL;

}

void RandomForest::train(vector<Mat> &sample, vector<int> &label){
	if(sample_num > sample.size()){
		cout << "Sample size out of range, " << sample.size() << " sample will be used" << endl;
		sample_num = sample.size();
	}

	srand(unsigned(time(NULL)));

	int *ID = new int[sample.size()];

	for(int i=0; i<tree_num; i++){
		cout << "Start to train the " << i << "th tree" << endl;

		int count = 0;
		int pos_count = 0;
		
		float sp = 1.0*sample_num/sample.size();

		for(int j=0; j<sample.size(); j++){
			if((rand()%10001)/10000.0<=sp){
				ID[count++] = j;
				pos_count += label[j];
			}
		}

		int *ID_root = new int[count];
		memcpy(ID_root, ID, count*sizeof(ID_root));

		/*cout << "count = " << count << endl;
		cout << "pos_count = " << pos_count << endl;
		cout << "ID ROOT = " << ID_root << endl;
		cin.get();*/

		root_list[i] = new Node(0, window_width, maxDepth, minLeafSample, minInfoGain, count, pos_count);
		root_list[i]->split_Node(sample, label, ID_root);
	}
}

void RandomForest::save(ofstream &fout){
	cout << "*****************Start to save the model*****************" << endl;
	fout << tree_num << " " << sample_num << " " << maxDepth << " " << minLeafSample << " " << minInfoGain << endl;
	for(int i=0; i<tree_num; i++)
		root_list[i]->save(fout);
	cout << "*****************Saving completed*****************" << endl << endl;
}


void RandomForest::load(ifstream &fin){
	cout << "*****************Start to load the model*****************" << endl;
	fin >> tree_num >> sample_num >> maxDepth >> minLeafSample >> minInfoGain;
	//cout << "tree_num:" << tree_num << " sample_num:" << sample_num << " maxDepth:" << maxDepth << " minLeafSample:" << minLeafSample << " minInfoGain:" << minInfoGain << endl;
	//cin.get();
	for(int i=0; i<tree_num; i++){
		root_list[i] = new Node();
		root_list[i]->load(fin);
	}
	cout << "*****************Loading completed*****************" << endl << endl;
}

float RandomForest::predict(Mat test_img){
	float vote = 0.0;
	for(int j=0; j<tree_num; j++)
		vote += root_list[j]->predict(test_img);
	
	return 1.0*vote/tree_num;
}

vector<float> RandomForest::predict(vector<Mat> &test_img){
	//cout << "Start to predict" << endl;
	//cout << "test size = " << test_img.size() << endl;
	vector<float> predict_result;
	for(int i=0; i<test_img.size(); i++){
		predict_result.push_back(predict(test_img[i]));
	}
	//cout << "predict size = " << predict_result.size() << endl;

	return predict_result;
}