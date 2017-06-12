#include <iostream>
#include "FeatureFileSet.h"
#include "DTWCluster.h"
#include "CWaveFormatFile.h"
#include <string>
#include <fstream>
#include "ClusterParam.h"
#include "DisCluster.h"
#define  MAXFRAME 5000

bool AutoRate = true;
int maxCCnt = 3;//最大聚类帧长
long long int cltCount = 0;
long long int FrameCount = 0;

string DirMatchMainFile(string filename){
	auto iter = filename.rbegin();
	while (*iter != '#')
		iter++;
	iter++;
	string out;
	while (*iter != '\\'){
		out += *iter;
		iter++;
	}
	reverse(out.begin(), out.end());
	return out;
}
vector<string> readWavList(string filename) {
	vector<string> r;

	ifstream infile(filename);
	string t;
	while (infile >> t) {
		r.push_back(t);
	}
	infile.close();
	return r;
}
void getClusterInfoFileDTW(const float& rate, const int& fDim, ClusterParam& cParam,const int & clusterFlag){
	//DTW提取的实现。

	vector<CSpeechFile> inputs = cParam.getClusterFiles();

	vector<float> energyFeature;

	for (int ii = 0; ii != inputs.size(); ii++)
	{
		printf("Files：%.2f%%\t%s\n",(float)(ii) / inputs.size() * 100, inputs[ii].getFeatureFileName().c_str());
		//读写文件
		int p = inputs[ii].getFeatureFileName().find_last_of('/'), q = inputs[ii].getFeatureFileName().find_last_of('.');
		string d45Filename = inputs[ii].getFeatureFileName();
		string cltDirName = cParam.getSaveDir();//d45Filename.substr(0, p + 1)+ to_string(rate).substr(0,4);
		if (GetFileAttributes(cltDirName.c_str()) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(cltDirName.c_str(), NULL);
		}
		string outFilename = string(cltDirName + d45Filename.substr(p, q - p) + ".clt");//保存输出的文件名，可以自己决定更换位置

		long cnt = 0;
		long maxIter = 55;
		FeatureFileSet input(inputs[ii].getFeatureFileName(), inputs[ii].getMaskFileName(), inputs[ii].getAnswerFileName(), fDim);

		int sentence = input.getSpeechNum();

		ClusterIndex* Headvec = new ClusterIndex[sentence];
		int* InfoVec = new int[sentence* (MAXFRAME)];
		long offset = sentence * sizeof(ClusterIndex);

		for (int j = 0; j != sentence; j++)
		{
			printf("%.2f%%\r",(float)(j + 1)/sentence * 100);
			int segNum = input.getSegNum(j);
			int fNum = 0;
			double* features = nullptr;
			if (cParam.getHalfLen())//5ms帧移，取特征序列
			{
				fNum = input.getFrameNumInSpeech_half_framelen(j);
				features = new double[fNum * fDim];
				input.getSpeechAt_half_framelen(j, features, fNum);
			}
			else{//10ms帧移，取特征序列
				fNum = input.getFrameNumInSpeech(j);
				features = new double[fNum * fDim];
				input.getSpeechAt(j, features);
			}

			DTWCluster clust(fDim);
			clust.init(int(fNum/rate), maxIter, fDim);
			clust.doCluster(features, fNum, clusterFlag, NULL, segNum);
			auto infovec = clust.getClusterInfo();
			for (auto ii : infovec)
			{
				InfoVec[cnt] = ii;
				cnt++;
			}
			Headvec[j].ClusterNum = infovec.size();
			Headvec[j].offset = offset;
			offset += Headvec[j].ClusterNum * sizeof(float);
			delete []features;
		}
		FILE* fp = fopen(outFilename.c_str(),"wb+");
		fwrite(Headvec, sizeof(ClusterIndex) ,sentence, fp);
		fwrite(InfoVec, sizeof(long), cnt, fp);
		fclose(fp);

		delete []Headvec;
		delete []InfoVec;
	}

}


void getClusterInfoFileNICV(const float& threshold, const int& fDim, ClusterParam& cParam,const int & clusterFlag){

	vector<CSpeechFile> inputs = cParam.getClusterFiles();

	vector<float> energyFeature;

	for (int ii = 0; ii != inputs.size(); ii++)
	{
		printf("Files：%.2f%%\t%s\n",(float)(ii) / inputs.size() * 100, inputs[ii].getFeatureFileName().c_str());

		int p = inputs[ii].getFeatureFileName().find_last_of('/'), q = inputs[ii].getFeatureFileName().find_last_of('.');
		string d45Filename = inputs[ii].getFeatureFileName();
		string cltDirName = cParam.getSaveDir();//d45Filename.substr(0, p + 1)+ to_string(threshold * 100).substr(0,4);

		if (GetFileAttributes(cltDirName.c_str()) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(cltDirName.c_str(), NULL);
		}
		string outFilename = string(cltDirName + d45Filename.substr(p, q - p) + ".clt");//保存输出的文件名，可以自己决定更换位置
		long cnt = 0;
		FeatureFileSet input(inputs[ii].getFeatureFileName(), inputs[ii].getMaskFileName(), inputs[ii].getAnswerFileName(), fDim);

		int sentence = input.getSpeechNum();
		ClusterIndex* Headvec = new ClusterIndex[sentence];
		int* InfoVec = new int[sentence* (MAXFRAME)];
		long offset = sentence * sizeof(ClusterIndex);
		double sumRate = 0.0f;
		int fileFrameNum = 0;
		for (int j = 0; j != sentence; j++)
		{
			printf("%.2f%%\r",(float)(j + 1)/sentence * 100);
			int segNum = input.getSegNum(j);

			int fNum = 0;
			double* features = nullptr;
			if (cParam.getHalfLen())
			{
				fNum = input.getFrameNumInSpeech_half_framelen(j);
				features = new double[fNum * fDim];
				input.getSpeechAt_half_framelen(j, features, fNum);
			}
			else{
				fNum = input.getFrameNumInSpeech(j);
				features = new double[fNum * fDim];
				input.getSpeechAt(j, features);
			}
			DisCluster clust(fDim);
			clust.init(fDim, threshold, maxCCnt);
			double res = clust.doCluster(features, fNum);//返回的是压缩率
			vector<int> infovec = clust.getClusterInfo();
			sumRate += res;
			for (auto ii : infovec)
			{
				InfoVec[cnt] = ii;
				cnt++;
			}
			Headvec[j].ClusterNum = infovec.size();
			Headvec[j].offset = offset;
			offset += Headvec[j].ClusterNum * sizeof(float);

			cltCount += infovec.size();
			FrameCount += fNum;
			delete []features;
		}
		cout<<"Average Rate:"<<sumRate/sentence<<endl;//平均压缩率

		FILE* fp = fopen(outFilename.c_str(),"wb+");
		fwrite(Headvec, sizeof(ClusterIndex) ,sentence, fp);
		fwrite(InfoVec, sizeof(long), cnt, fp);
		fclose(fp);//保存文件

		delete []Headvec;
		delete []InfoVec;
	}

}

int main(int argc, char** argv){
	string configFile(argv[1]);
	ClusterParam cparam(configFile.c_str());
	if(cparam.getNICVflag()){
		getClusterInfoFileNICV(cparam.getThres()/100.0, cparam.getFdim(), cparam, DisCluster::MEAN_CENTER);
	}else{
		getClusterInfoFileDTW(cparam.getRate(), cparam.getFdim(), cparam, DTWCluster::MEAN_CENTER);
	}

	return 0;
}