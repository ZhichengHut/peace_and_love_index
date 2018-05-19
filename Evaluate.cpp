#include "Evaluate.h"


void get_predict_result(RandomForest *RF, string test_fold, Mat &mask, float threshold){
    char curDir[100];

    for(int c=1; c<=12; c++){
		sprintf(curDir, "%s%02i", test_fold.c_str(), c);
		cout << curDir << endl;

		DIR* pDIR;
		struct dirent *entry;
		struct stat s;

		stat(curDir,&s);

		// if path is a directory
		if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
			if(pDIR=opendir(curDir)){
				//for all entries in directory
				while(entry = readdir(pDIR)){
					stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
					if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
						if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "tif"){
							//find the corresponding fold according to the image name
							string sub_curDIR = string(curDir) + "/" + string(entry->d_name).substr(0,2);
							
							vector<Mat> imgTest;
							vector<int> X,Y;

							DIR* subDIR;
							struct dirent *sub_entry;
							struct stat sub_s;

							stat(sub_curDIR.c_str(), &sub_s);

							/////////extract the test image and their position
							if((sub_s.st_mode & S_IFMT) == S_IFDIR ){
								if(subDIR=opendir(sub_curDIR.c_str())){
									while(sub_entry = readdir(subDIR)){
										stat((sub_curDIR + string("/") + string(sub_entry->d_name)).c_str(),&sub_s);
										if (((sub_s.st_mode & S_IFMT ) != S_IFDIR ) && ((sub_s.st_mode & S_IFMT) == S_IFREG )){
											if(string(sub_entry->d_name).substr(string(sub_entry->d_name).find_last_of('.') + 1) == "png"){
												Mat img_tmp = imread(sub_curDIR + string("/") + string(sub_entry->d_name), 0);
												//if(TLBO_test(img_tmp, mask, threshold)){
												if(true){
													integral(img_tmp, img_tmp);	
													int x = atoi(string(sub_entry->d_name).substr(0,4).c_str());
													int y = atoi(string(sub_entry->d_name).substr(5,4).c_str());
													imgTest.push_back(img_tmp);
													X.push_back(x);
													Y.push_back(y);
												}
											}
										}
									}
								}
							}
							vector<float> result = RF->predict(imgTest);

							imgTest.clear();
							vector<Mat>().swap(imgTest);

							for(float p_t=0.1; p_t<=1; p_t+=0.05){
								//string csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict.csv";
								stringstream stream_pt;
								stream_pt << p_t;
								
								string csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict_" + stream_pt.str() + ".csv";
								ofstream fout(csv_name);
								for(int i=0; i<result.size(); i++){
									if(result[i] >= p_t)
										fout << Y[i] << "," << X[i] << endl;
								}
								fout.close();
							}

							result.clear();
							vector<float>().swap(result);
							X.clear();
							vector<int>().swap(X);
							Y.clear();
							vector<int>().swap(Y);

							//fout.close();
						}
					}
				}
			}
		}
	}
}

void get_predict_result(RandomForest *RF, string test_fold, int width, int sample_interval, float prob_threshold, Mat &mask, float thresh){
    char curDir[100];

    for(int c=1; c<=12; c++){
		sprintf(curDir, "%s%02i", test_fold.c_str(), c);
		cout << curDir << endl;

		DIR* pDIR;
		struct dirent *entry;
		struct stat s;

		stat(curDir,&s);

		// if path is a directory
		if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
			if(pDIR=opendir(curDir)){
				//for all entries in directory
				while(entry = readdir(pDIR)){
					stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
					if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
						if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "tif"){
							//find the corresponding fold according to the image name
							string cur_img = string(curDir) + "/" + string(entry->d_name);
							cout << "current img: " << cur_img << endl;
							
							//Mat imgTest = imread(cur_img,0);
							Mat imgTest = imread(cur_img,1);
							Mat out[3];
							split(imgTest, out);
							
							Mat r = out[2];
							Mat g = out[1];
							Mat b = out[0];
						
							r.convertTo(r,CV_32FC1);
							g.convertTo(g,CV_32FC1);
							b.convertTo(b,CV_32FC1);
						
							Mat b_r = 100*b/(1+r+g)*256/(1+r+g+b);
							b_r.convertTo(b_r,CV_8UC1);

							vector<float> result;
							Mat test_tmp;

							for(int x=0; x<imgTest.cols-2*width; x+=sample_interval){
								for(int y=0; y<imgTest.rows-2*width; y+=sample_interval){
									integral(b_r(Rect(x,y,2*width,2*width)), test_tmp);	
									result.push_back(RF->predict(test_tmp));
								}
							}

							int m = sqrt(result.size()*1.0);
							//cout << "test size: " << result.size() << endl;
							//cout << "m: " << m << endl;

							Mat heat_map_tmp = Mat::zeros(m,m,CV_32FC1);
							for(int j=0; j<m; j++){
								for(int i=0; i<m; i++){
									heat_map_tmp.at<float>(i,j) = result[j*m+i];
								}
							}
							
							for(int window_width=11; window_width<=31; window_width+=2){
								Mat s_map = NMS(heat_map_tmp, window_width);
								for(float p_t=0; p_t<1.01; p_t+=0.05){
									Mat s_map_tmp;
									threshold(s_map, s_map_tmp, p_t, 255, THRESH_BINARY);
									s_map_tmp.convertTo(s_map_tmp, CV_8UC1);
									/*cout << "type = " << s_map_tmp.type() << endl;
									cin.get();
									cout << "threshold = " << p_t << endl;
									cout << s_map_tmp(Rect(50,50,50,50)) << endl;
									cin.get();*/
									Mat nonzero_location;
									findNonZero(s_map_tmp, nonzero_location);
									nonzero_location = nonzero_location*sample_interval+width;

									cout << "mitosis number: " << nonzero_location.rows << endl;
								
									stringstream stream_pt, stream_R;  
									stream_pt << p_t;
									stream_R << window_width;

									string csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict_" + stream_pt.str() + "_" + stream_R.str() + ".csv";
									ofstream fout(csv_name);

									for(int i=0; i<nonzero_location.rows; i++)
										fout << nonzero_location.at<Point>(i,0).y << "," << nonzero_location.at<Point>(i,0).x << endl;
									
									fout.close();
								}
							}
						}
					}
				}
			}
		}
	}
}

void get_predict_result(RandomForest *RF, string test_fold, int width){
	char curDir[100];

    for(int c=1; c<=12; c++){
		sprintf(curDir, "%s%02i", test_fold.c_str(), c);
		cout << curDir << endl;

		DIR* pDIR;
		struct dirent *entry;
		struct stat s;

		stat(curDir,&s);

		// if path is a directory
		if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
			if(pDIR=opendir(curDir)){
				//for all entries in directory
				while(entry = readdir(pDIR)){
					stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
					if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
						if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "tif"){
							//find the corresponding fold according to the image name
							string cur_img = string(curDir) + "/" + string(entry->d_name);
							cout << "current img: " << cur_img << endl;
							
							Mat imgTest = imread(cur_img,0);

							string csv_name= string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict.csv";
							vector<int> prediction;
							ifstream fin(csv_name);
							if(fin){
								cout << csv_name << endl;
								prediction = readCSV(csv_name);
							}
							fin.close();

							string csv_name_second = string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict_second.csv";
							ofstream fout(csv_name_second);

							for(int i=0; i<prediction.size(); i+=2){
								int y = prediction[i];
								int x = prediction[i+1];

								if(x < width + 1)
									x = width + 1;
								else if(x > imgTest.cols - width)
									x = imgTest.cols - width;
								
								if(y < width + 1)
									y = width + 1;
								else if(y > imgTest.rows - width)
									y = imgTest.rows - width;

								Mat imgTest_integral;
								integral(imgTest(Rect(x-width,y-width,2*width,2*width)),imgTest_integral);

								if(RF->predict(imgTest_integral)>0.5)
									fout << y << "," << x << endl;
							}
						}
					}
				}
			}
		}
	}
}


float get_F1_score(string test_fold){
	int indexx = 0;

	for(float p_t=0; p_t<1.01; p_t+=0.05){
		for(int R = 11; R<=31; R+=2){
			cout << "th = " << p_t << ", R = " << R << endl;
	//for(float p_t=0.1; p_t<=1; p_t+=0.05){

			int TP = 0;
			int FP = 0;
			int FN = 0;

			float F1_score = 0.0;

			char curDir[100];

			for(int c=1; c<=12; c++){
				sprintf(curDir, "%s%02i", test_fold.c_str(), c);

				DIR* pDIR;
				struct dirent *entry;
				struct stat s;

				stat(curDir,&s);

				// if path is a directory
				if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
					if(pDIR=opendir(curDir)){
						//for all entries in directory
						while(entry = readdir(pDIR)){
							stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
							if (((s.st_mode & S_IFMT ) != S_IFDIR ) && ((s.st_mode & S_IFMT) == S_IFREG )){
								if(string(entry->d_name).substr(string(entry->d_name).find_last_of('.') + 1) == "tif"){
									vector<int> ground_truth;
									vector<int> prediction;

									string csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,2) + ".csv";
									ifstream fin1(csv_name);
									if(fin1){
										//cout << csv_name << endl;
										ground_truth = readCSV(csv_name);
									}
									fin1.close();

									stringstream stream_pt, stream_R;  
									stream_pt << p_t;
									stream_R << R;

									/*stringstream stream_pt;  
									stream_pt << p_t;*/

									csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict_" + stream_pt.str() + "_" + stream_R.str() + ".csv";
									//csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict_" + stream_pt.str() + ".csv";
									//csv_name = string(curDir) + "/" + string(entry->d_name).substr(0,2) + "_predict.csv";
									ifstream fin2(csv_name);
									if(fin2){
										//cout << csv_name << endl;
										prediction = readCSV(csv_name);
									}
									fin2.close();

									int num1 = ground_truth.size() / 2;
									int num2 = prediction.size() / 2;
									FN += num1;
									FP += num2;

									/*for(int i=0; i<ground_truth.size(); i+=2){
										for(int j=0; j<prediction.size(); j+=2){
											float distance = sqrt(pow(ground_truth[i]-prediction[j],2.0)+pow(ground_truth[i+1]-prediction[j+1],2.0));
											if(distance <= 30){
												TP++;
												FP--;
												FN--;
											}
										}
									}*/

									/*string img_name = string(curDir) + "/" + string(entry->d_name);
									cout << "img_name = " << img_name << endl;
									//cin.get();
									Mat ground_img = imread(img_name, 1);
									string detect_fold = "E:/detected/";
									char curDetect[100];*/
							
									for(vector<int>::iterator i=ground_truth.begin(); i!=ground_truth.end();){
										bool detected_flag = false;
										for(vector<int>::iterator j=prediction.begin(); j!=prediction.end();){
											 float distance = sqrt(pow(*i-*j,2.0)+pow(*(i+1)-*(j+1),2.0));
											 if(distance <= 30){
												 /*int x = *(j+1);
												 int y = *j;

												//whether the position is out of bound
												if(x < 35 + 1)
													x = 35 + 1;
												else if(x > ground_img.cols - 35)
													x = ground_img.cols - 35;

												if(y < 35 + 1)
													y = 35 + 1;
												else if(y > ground_img.rows - 35)
													y = ground_img.rows - 35;

												 sprintf(curDetect, "%s%04i_TP.png", detect_fold.c_str(), indexx++);
												 imwrite(curDetect, ground_img(Rect(x-35,y-35,70,70)));*/

												TP++;
												FP--;
												FN--;
												i = ground_truth.erase(i);
												i = ground_truth.erase(i);
												j = prediction.erase(j);
												j = prediction.erase(j);
												detected_flag = true;
												break;
											 }
											 else{
												 j+=2;
											 }
										}

										if(!detected_flag){
											/*int x = *(i+1);
											int y = *i;
											
											//whether the position is out of bound
											if(x < 35 + 1)
												x = 35 + 1;
											else if(x > ground_img.cols - 35)
												x = ground_img.cols - 35;
											
											if(y < 35 + 1)
												y = 35 + 1;
											else if(y > ground_img.rows - 35)
												y = ground_img.rows - 35;
											sprintf(curDetect, "%s%04i_FN.png", detect_fold.c_str(), indexx++);
											imwrite(curDetect, ground_img(Rect(x-35,y-35,70,70)));*/

											i+=2;
										}
									}
									/*for(vector<int>::iterator j=prediction.begin(); j!=prediction.end();){
										int x = *(j+1);
										int y = *j;
										
										//whether the position is out of bound
										if(x < 35 + 1)
											x = 35 + 1;
										else if(x > ground_img.cols - 35)
											x = ground_img.cols - 35;
										
										if(y < 35 + 1)
											y = 35 + 1;
										else if(y > ground_img.rows - 35)
											y = ground_img.rows - 35;
										
										sprintf(curDetect, "%s%04i_FP.png", detect_fold.c_str(), indexx++);
										imwrite(curDetect, ground_img(Rect(x-35,y-35,70,70)));

										j = prediction.erase(j);
										j = prediction.erase(j);
									}*/
								}
							}
						}
					}
				}
			}

			float Pr = 1.0*TP/(TP+FP);
			float Re = 1.0*TP/(TP+FN);
			F1_score = 2*Pr*Re/(Pr+Re);
			cout << "Pr = " << Pr << ", Re = " << Re << ", F1 score = " << F1_score << endl;
			cout << "TP = " << TP << ", FP = " << FP << ", FN = " << FN << endl;

			ofstream fin("e:\\45 Thesis\\result\\result.csv",ios::app);
			if(!fin){
				cout << "open file error" <<endl; 
				cin.get();
				return 0;
			}
			
			//fin << "TP," << TP << ",FP," << FP << ",FN," << FN << ",Pr," << Pr << ",Re," << Re << ",F1 score," << F1_score << ",prob," << p_t << ",R," << R;
			fin << "TP," << TP << ",FP," << FP << ",FN," << FN << ",Pr," << Pr << ",Re," << Re << ",F1 score," << F1_score << ",prob," << p_t << ",R," << R << endl;
			//fin << "TP," << TP << ",FP," << FP << ",FN," << FN << ",Pr," << Pr << ",Re," << Re << ",F1 score," << F1_score << ",prob," << p_t << endl;
			fin.close();
			//}
	}
			}

	return 0.0;
}

Mat NMS(Mat &img_tmp, int window_width){
	Mat img = img_tmp.clone();

	int row = img.rows;
	int col = img.cols;

	double max, min;
	Point min_loc, max_loc;

	Mat blank_box = Mat::zeros(row, col, CV_32FC1);
	//img.convertTo(img, blank_box.type());
	Mat s_map = Mat::zeros(row, col, CV_32FC1);

	while(true){
		minMaxLoc(img, &min, &max, &min_loc, &max_loc);
		if(max > 0){
			int x_left = (max_loc.x-window_width < 0) ? max_loc.x : window_width;
			int x_right = (max_loc.x+window_width > col) ? col-max_loc.x : window_width;
			int y_up = (max_loc.y-window_width < 0) ? max_loc.y : window_width;
			int y_down = (max_loc.y+window_width > row) ? row-max_loc.y : window_width;

			s_map.at<float>(max_loc.y, max_loc.x) = max;
			blank_box(Rect(max_loc.x-x_left, max_loc.y-y_up, x_left+x_right, y_up+y_down)).copyTo(img(Rect(max_loc.x-x_left, max_loc.y-y_up, x_left+x_right, y_up+y_down)));
		}
		else
			break;
	}

	return s_map;
}


bool TLBO_test(Mat &img, Mat &mask, float threshold){
	int histSize = 256;
	float range[] = {0, 256} ;
	const float* histRange = {range};
	
	bool uniform = true; 
	bool accumulate = false;
	
	Mat hist;
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	Mat value = mask*hist;

	if(value.at<float>(0,0) > threshold)
		return false;
	else
		return true;
}