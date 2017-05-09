#include <iostream>
#include "FeatureFileSet.h"
#include "DTWCluster.h"
#include "CWaveFormatFile.h"
#include <string>
#include <fstream>
#include "EFeature.h"
#include "RecParam.h"
#include "DisCluster.h"
#define  MAXFRAME 5000
#define  HALF_FRAMELEN 1

float threshold = 0.0f;
bool AutoRate = true;
int simplePointCenter = 0;
int maxCCnt = 3;
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
void getClusterInfoFile(const string& configName, const float& rate, const int& iterTime, const int & clusterFlag){

	RecParam rparam(configName.c_str());

	vector<RSpeechFile> inputs = rparam.getRecFiles();

	vector<float> energyFeature;

	for (int ii = 0; ii != inputs.size(); ii++)
	{
		printf("Files£º%.2f%%\t%s\n",(float)(ii) / inputs.size() * 100, inputs[ii].getFeatureFileName().c_str());

		int p = inputs[ii].getFeatureFileName().find_last_of('/'), q = inputs[ii].getFeatureFileName().find_last_of('.');
		string d45Filename = inputs[ii].getFeatureFileName();
		string cltDirName = d45Filename.substr(0, p + 1)+ to_string(AutoRate ? threshold : rate).substr(0,4);
		if (GetFileAttributes(cltDirName.c_str()) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(cltDirName.c_str(), NULL);
		}
		string outFilename = string(cltDirName + d45Filename.substr(p, q - p) + ".clt");
		//string outFilename = string(d45Filename.substr(0, q) + ".clt") ;

		int fDim = rparam.getFdim();
		long cnt = 0;
		long maxIter = 55;
		FeatureFileSet input(inputs[ii].getFeatureFileName(), inputs[ii].getMaskFileName(), inputs[ii].getAnswerFileName(), fDim);

		int sentence = input.getSpeechNum();
		ClusterIndex* Headvec = new ClusterIndex[sentence];
		int* InfoVec = new int[sentence* (MAXFRAME)];
		long offset = sentence * sizeof(ClusterIndex);
		double fileAdjoinDistance = 0.0f, sumRate = 0.0f;
		int fileFrameNum = 0;
		for (int j = 0; j != sentence; j++)
		{
			printf("%.2f%%\r",(float)(j + 1)/sentence * 100);
			int segNum = input.getSegNum(j);

#if HALF_FRAMELEN

			int fNum = input.getFrameNumInSpeech_half_framelen(j);
			double* features = new double[fNum * fDim];
			input.getSpeechAt_half_framelen(j, features, fNum);
			auto speechNum = fNum;
#else
			auto speechNum = input.getFrameNumInSpeech(j);
			auto firstIdx = input.getFirstFrameNumInSpeech(j);
			int fNum = input.getFrameNumInSpeech(j);
			double* features = new double[fNum * fDim];
			input.getSpeechAt(j, features);
			int* jumpTable = new int[segNum];
			input.getJumpTable(j, jumpTable);
#endif
			int minFrameLen = (speechNum);


			for (int k = 0; k != maxIter; k++)
			{
				DTWCluster clust(fDim);

				double myrate = clust.initDistance(features, fDim, fNum, rate - k * 0.05);
				//clust.init((double)minFrameLen/(rate + k * 0.005), iterTime, fDim);
				double res = clust.doCluster(features, minFrameLen, clusterFlag, /*jumpTable*/ NULL, segNum);
				fileAdjoinDistance += res * minFrameLen;
				fileFrameNum += minFrameLen;
				//cout << myrate << endl;
				if(myrate > threshold && k < maxIter - 1 && AutoRate && rate - (k + 1) * 0.05 >=0){
					continue;
				}
				sumRate += myrate;
				/*
				if(res < threshold && k < maxIter - 1 && AutoRate && rate + k *0.005 < 1.1 ){
				continue;
				}*/
				//cout<< "Rate:"<< myrate <<"res:"<<res<<endl;
				auto infovec = clust.getClusterInfo(simplePointCenter);
				for (auto ii : infovec)
				{
					InfoVec[cnt] = ii;
					cnt++;
				}
				Headvec[j].ClusterNum = infovec.size();
				Headvec[j].offset = offset;
				offset += Headvec[j].ClusterNum * sizeof(float);
				break;
			}
			//debug
			//delete []jumpTable;
			delete []features;
		}
		//cout<<"Average AdjoinDistance:"<<fileAdjoinDistance/fileFrameNum<<endl;
		cout<<"Average sumRate:"<<sumRate/sentence<<endl;

		FILE* fp = fopen(outFilename.c_str(),"wb+");
		fwrite(Headvec, sizeof(ClusterIndex) ,sentence, fp);
		fwrite(InfoVec, sizeof(long), cnt, fp);
		fclose(fp);

		delete []Headvec;
		delete []InfoVec;
	}


}




void getClusterInfoFileDis(const string& configName, const float& rate, const int& iterTime, const int & clusterFlag){

	RecParam rparam(configName.c_str());

	vector<RSpeechFile> inputs = rparam.getRecFiles();

	vector<float> energyFeature;

	for (int ii = 0; ii != inputs.size(); ii++)
	{
		printf("Files£º%.2f%%\t%s\n",(float)(ii) / inputs.size() * 100, inputs[ii].getFeatureFileName().c_str());

		int p = inputs[ii].getFeatureFileName().find_last_of('/'), q = inputs[ii].getFeatureFileName().find_last_of('.');
		string d45Filename = inputs[ii].getFeatureFileName();
		//string cltDirName = d45Filename.substr(0, p + 1)+ to_string(AutoRate ? threshold : rate).substr(0,4);
		string cltDirName = d45Filename.substr(0, p + 1)+ to_string(rate * 100).substr(0,4);

		if (GetFileAttributes(cltDirName.c_str()) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(cltDirName.c_str(), NULL);
		}
		string outFilename = string(cltDirName + d45Filename.substr(p, q - p) + ".clt");
		//string outFilename = string(d45Filename.substr(0, q) + ".clt") ;

		int fDim = rparam.getFdim();
		long cnt = 0;
		FeatureFileSet input(inputs[ii].getFeatureFileName(), inputs[ii].getMaskFileName(), inputs[ii].getAnswerFileName(), fDim);

		int sentence = input.getSpeechNum();
		ClusterIndex* Headvec = new ClusterIndex[sentence];
		int* InfoVec = new int[sentence* (MAXFRAME)];
		long offset = sentence * sizeof(ClusterIndex);
		double fileAdjoinDistance = 0.0f, sumRate = 0.0f;
		int fileFrameNum = 0;
		for (int j = 0; j != sentence; j++)
		{
			printf("%.2f%%\r",(float)(j + 1)/sentence * 100);

#if HALF_FRAMELEN

			int fNum = input.getFrameNumInSpeech_half_framelen(j);
			auto speechNum1 = input.getFrameNumInSpeech(j);

			double* features = new double[fNum * fDim];
			input.getSpeechAt_half_framelen(j, features, fNum);
			auto speechNum = fNum;
#else
			auto speechNum = input.getFrameNumInSpeech(j);
			auto firstIdx = input.getFirstFrameNumInSpeech(j);
			int fNum = input.getFrameNumInSpeech(j);
			double* features = new double[fNum * fDim];
			input.getSpeechAt(j, features);
			int* jumpTable = new int[segNum];
			input.getJumpTable(j, jumpTable);
#endif
			int minFrameLen = (speechNum);

			DisCluster clust(fDim);

			clust.init(fDim, rate, maxCCnt);
			//clust.init((double)minFrameLen/(rate + k * 0.005), iterTime, fDim);
			double res = clust.doCluster(features, minFrameLen);//, clusterFlag, /*jumpTable*/ NULL, segNum);
			vector<int> infovec = clust.getClusterInfo();
			//cout<< "rate:" << 1/res << endl;
			sumRate += 1/res;
			for (auto ii : infovec)
			{
				InfoVec[cnt] = ii;
				cnt++;
			}
			Headvec[j].ClusterNum = infovec.size();
			Headvec[j].offset = offset;
			offset += Headvec[j].ClusterNum * sizeof(float);

			cltCount += infovec.size();
			FrameCount += minFrameLen;
			//debug
			delete []features;
		}
		//cout<<"Average AdjoinDistance:"<<fileAdjoinDistance/fileFrameNum<<endl;
		cout<<"Average sumRate:"<<sumRate/sentence<<endl;

		FILE* fp = fopen(outFilename.c_str(),"wb+");
		fwrite(Headvec, sizeof(ClusterIndex) ,sentence, fp);
		fwrite(InfoVec, sizeof(long), cnt, fp);
		fclose(fp);

		delete []Headvec;
		delete []InfoVec;
	}

}
// Dtw.exe rate iterTime threshold MeanCenter simplePointCenter
int main(int argc, char** argv){
	string configFile(argv[1]);
	float rate = atof(argv[2]);
	rate /= 100;
	maxCCnt = atof(argv[3]);

	int iterTime = 1;
	if (argc > 4)
	{
		//iterTime = atof(argv[3]);
		threshold = atof(argv[4]);
		AutoRate = threshold != 0.0f;
	}

	int clusterflag = DTWCluster::MEAN_CENTER;
	cout<<"argc:"<<argc<<endl;
	if (argc >= 6)
	{
		clusterflag = atof(argv[5]);
		if(clusterflag == 0)cout << "Mean_Center" << endl;
		if(clusterflag == 5)cout << "Medoid_Center" << endl;
		cout << "clusterFlage" << clusterflag <<endl;
	}
	if (argc == 7)
	{
		simplePointCenter = atof(argv[6]);
		if(simplePointCenter != 0)cout << "simplePointCenter " << endl;
	}


	getClusterInfoFileDis(configFile, rate, iterTime, clusterflag);
	FILE* f = fopen((/*to_string(rate*100)*/string("log") + ".txt").c_str(),"a+");
	fprintf(f,"%f %lf\n",rate * 100,((double)FrameCount/cltCount));
	fclose(f);
	return 0;
}