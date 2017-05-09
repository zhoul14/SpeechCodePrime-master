#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include <string>
using namespace std;
class Cluster{
public:
	static const int MEAN = 1;
	static const int RANDOM = 2;
	static const int MULTI = 3;
	static const int MIDDLE = 4;
	static const int MEDOIDS = 5;
public:
	Cluster(string filename, FeatureFileSet& fs, string dirName =""){
		mEfBuf = nullptr;
		if (dirName == "")
		{
			return ;
		}
		string pf = filename.substr(filename.find_last_of('/') + 1,filename.find_last_of('.') - filename.find_last_of('/') -1);
		string fDir = filename.substr(0, filename.find_last_of('/')+1);
		//string s = filename.substr(0, filename.find_last_of('.') + 1);
		//FILE* df = fopen(string("clt/"+ s + ".clt").c_str(),"rb");
		dirName += dirName != ""? "/" : "";
		FILE* df = fopen(string(fDir + dirName + pf + ".clt").c_str(),"rb");
		int n = fs.fileByteNum(df);
		mEfBuf = new char[n];
		fread(mEfBuf, 1, n, df);
		fclose(df);
	};
	~Cluster(){
		if (mEfBuf!= nullptr)
		{
			delete []mEfBuf;
		}
	}
	int* getClusterInfo(const int& j, const int&fNum){
		ClusterIndex* clustidx = (ClusterIndex*)(mEfBuf + sizeof(ClusterIndex) * j);
		int* segBuf = (int*)(clustidx->offset + mEfBuf);
		int clusterNum = clustidx->ClusterNum;
		if(segBuf[clusterNum -1] != fNum){
			printf("last cluster idx:%d != fNum:%d\n",segBuf[clusterNum -1],fNum);
		}
		return segBuf;
	}
	int getClusterNum(const int& j){
		ClusterIndex* clustidx = (ClusterIndex*)(mEfBuf + sizeof(ClusterIndex) * j);
		int clusterNum = clustidx->ClusterNum;
		return clusterNum;
	}
	int getClusterInfo(const int& j, const int&fNum, int* outBuf){
		ClusterIndex* clustidx = (ClusterIndex*)(mEfBuf + sizeof(ClusterIndex) * j);
		int* segBuf = (int*)(clustidx->offset + mEfBuf);
		int clusterNum = clustidx->ClusterNum;
		if(segBuf[clusterNum -1] != fNum){
			printf("last cluster idx:%d != fNum:%d\n",segBuf[clusterNum -1],fNum);
		}
		memcpy(outBuf,segBuf, sizeof(int) * clusterNum);
		return clusterNum;
	}


	double calcVar(double* features, double* center, int clusterNum, int fDim){
		double energy = 0.0f;
		for (int k = 0; k != clusterNum; k++)
		{
			for (int i = 0; i != 14; i++)
			{
				energy += features[i + k * fDim] * features[i + k * fDim];
			}
			//energy += features[42 + k * fDim] * features[42 + k * fDim];
		}
		double dis = 0.0f;
		for (int k = 0; k != clusterNum; k++)
		{
		
			for (int i = 0; i != 14; i++)
			{
				dis += (features[i + k * fDim] - center[i])* (features[i + k * fDim] - center[i]);
			}
			//dis += (features[42 + k * fDim] - center[42])* (features[42 + k * fDim] - center[42]);
		}
		auto res = dis/energy;
		return res;
	}
	int clusterStat(double* features, const int& j, const int &fDim, const int&fNum,vector<int>& statVec){
		ClusterIndex* clustidx = (ClusterIndex*)(mEfBuf + sizeof(ClusterIndex) * j);
		int* segBuf = (int*)(clustidx->offset + mEfBuf);
		int clusterNum = clustidx->ClusterNum;

		int counter = 0;
		double* temp = new double[fNum * fDim];
		double* head = temp;
		memset(temp,0,sizeof(double) * fNum * fDim);
		int outCnt = 0;
		int clustCnt = 0;
		int lastCnt = 0;
		vector<double>var;
		for (int t = 0; t !=fNum; t++)
		{
			if(t>=segBuf[outCnt]){
				lastCnt = segBuf[outCnt];

				if(clustCnt!=1){
					int idx = calcVar(features + (t - clustCnt) * fDim, head, clustCnt, fDim) * 1000 + 1;
					idx = min(149, idx);
					statVec[idx]+=1;
				}

				head += fDim;
				outCnt++;

			}
			auto fHead  = features + t * fDim;
			clustCnt = (segBuf[outCnt] > fNum? fNum : segBuf[outCnt]) - lastCnt;
			for (int d = 0; d != fDim; d++)
			{
				head[d] += fHead[d]/clustCnt;
			}
		}
		memcpy(features, temp, clusterNum * fDim * sizeof(double));
		delete[] temp;
		return fNum;
	}
	int clusterFrameSame(double* features, const int& j, const int &fDim, const int&fNum, int flag = MEAN){
		ClusterIndex* clustidx = (ClusterIndex*)(mEfBuf + sizeof(ClusterIndex) * j);
		int* segBuf = (int*)(clustidx->offset + mEfBuf);
		int clusterNum = clustidx->ClusterNum;
		if(segBuf[clusterNum -1] != fNum && flag != MEDOIDS){
			printf("last cluster idx:%d != fNum:%d\n",segBuf[clusterNum -1],fNum);
		}
		int counter = 0;
		double* temp = new double[fNum * fDim];
		double* head = temp;
		memset(temp,0,sizeof(double) * fNum * fDim);
		int outCnt = 0;
		int clustCnt = 0;
		int lastCnt = 0;
		switch(flag){
		case MEAN:
			{
				for (int t = 0; t !=fNum; t++)
				{
					if(t>=segBuf[outCnt]){
						lastCnt = segBuf[outCnt];
						head += fDim;
						outCnt++;
						for (int j = 1; j < clustCnt; j++)
						{
							memcpy(head,head - fDim, sizeof(double) * fDim);
							head+= fDim;
						}
					}
					auto fHead  = features + t * fDim;
					clustCnt = (segBuf[outCnt] > fNum? fNum : segBuf[outCnt]) - lastCnt;
					for (int d = 0; d != fDim; d++)
					{
						head[d] += fHead[d]/clustCnt;
					}
				}
			}
			break;
		}
		memcpy(features, temp, clusterNum * fDim * sizeof(double));
		delete[] temp;
		return fNum;
	}
	int clusterFrame(double* features, const int& j, const int &fDim, const int&fNum, int flag = MEAN){
		ClusterIndex* clustidx = (ClusterIndex*)(mEfBuf + sizeof(ClusterIndex) * j);
		int* segBuf = (int*)(clustidx->offset + mEfBuf);
		int clusterNum = clustidx->ClusterNum;
		if(segBuf[clusterNum -1] != fNum && flag != MEDOIDS){
			printf("last cluster idx:%d != fNum:%d\n",segBuf[clusterNum -1],fNum);
		}
		int counter = 0;
		double* temp = new double[clusterNum * fDim];
		double* head = temp;
		memset(temp,0,sizeof(double) * clusterNum * fDim);
		int outCnt = 0;
		int clustCnt = 0;
		int lastCnt = 0;
		switch(flag){
		case MEAN:
			{
				for (int t = 0; t !=fNum; t++)
				{
					if(t>=segBuf[outCnt]){
						lastCnt = segBuf[outCnt];
						head += fDim;
						outCnt++;
					}
					auto fHead  = features + t * fDim;
					clustCnt = (segBuf[outCnt] > fNum? fNum : segBuf[outCnt]) - lastCnt;

					for (int d = 0; d != fDim; d++)
					{
						head[d] += fHead[d]/clustCnt;
					}
				}
			}
			break;
		case RANDOM:
			{
				memcpy(head, features, sizeof(double) * fDim);
				head += fDim;

				for (int t = 0; t !=clusterNum - 1; t++)
				{
					auto fHead  = features + segBuf[t] * fDim;
					memcpy(head, fHead, sizeof(double) * fDim);
					head += fDim;
				}
			}
			break;
		case MEDOIDS:
			{
				for (int t = 0; t !=clusterNum; t++)
				{
					auto fHead  = features + segBuf[t] * fDim;
					memcpy(head, fHead, sizeof(double) * fDim);
					head += fDim;
				}
			}
			break;
		case MULTI:
			{
				for (int t = 0; t !=fNum; t++)
				{
					if(t>=segBuf[outCnt]){
						lastCnt = segBuf[outCnt];
						head += fDim;
						outCnt++;
					}
					auto fHead  = features + t * fDim;
					clustCnt = (segBuf[outCnt] > fNum? fNum : segBuf[outCnt]) - lastCnt;

					for (int d = 0; d != fDim; d++)
					{
						head[d] += fHead[d]/clustCnt;
					}
				}
			}
			break;
		}
		memcpy(features, temp, clusterNum * fDim * sizeof(double));
		delete[] temp;
		return clusterNum;
	}

	int clusterMultiFrame(double* features, const int& j, const int& fDim, const int& fNum, int* maskData, const int& segNum, double* outBuffer, const int& flag){
		ClusterIndex* clustidx = (ClusterIndex*)(mEfBuf + sizeof(ClusterIndex) * j);
		int* segBuf = (int*)(clustidx->offset + mEfBuf);
		int clusterNum = clustidx->ClusterNum;


		double* featuresBuffer = new double[fNum * fDim];
		int fBufIdx = 0;


		int totalFrameNum = 0;
		int cnt = 0;
		int localCnt = 0;
		int lastEndFrame = 0;
		for (int i = 0; i < segNum; i++) {
			int begFrame = maskData[i * 2];
			int endFrame = maskData[i * 2 + 1];

			if(begFrame <= MULTIFRAMES_COLLECT )
				begFrame = MULTIFRAMES_COLLECT ;
			if (endFrame >= fNum - MULTIFRAMES_COLLECT )
				endFrame = fNum - 1 - MULTIFRAMES_COLLECT; 

			memcpy(featuresBuffer + fBufIdx, features + lastEndFrame * fDim, sizeof(double) * fDim * (begFrame - lastEndFrame));//复制头
			fBufIdx += fDim * (begFrame - lastEndFrame);//记录这次的Head 到 上一次的tail 的中间。buff，指针往前移动
			lastEndFrame = endFrame + 1;//

			totalFrameNum = endFrame - begFrame + 1;
			double* temp = new double[totalFrameNum * fDim];
			memset( temp, 0 , sizeof(double) * totalFrameNum * fDim);
			double* head = temp;

			int clustCnt = 0;
			int lastCnt = localCnt == 0 ? 0 : segBuf[localCnt - 1];
			int historySegCnt = localCnt;

			if(flag == 1){
				for (int j = begFrame; j <= endFrame; j++) {
					if(cnt>=segBuf[localCnt]){
						lastCnt = segBuf[localCnt];
						head += fDim;
						localCnt++;
					}
					auto fHead  = features + j * fDim;
					clustCnt = (segBuf[localCnt] > endFrame ? endFrame - lastCnt : segBuf[localCnt]) - lastCnt;
					for (int d = 0; d != fDim; d++)
					{
						head[d] += fHead[d]/clustCnt;
					}
					cnt++;
				}
			}
			else{//单点

				for (int j = begFrame; j <= endFrame; j++) 
				{
					if(cnt == segBuf[localCnt]){
						auto fHead  = features + (j) * fDim;
						memcpy(head, fHead, sizeof(double) * fDim);
						head += fDim;
						localCnt++;
					}
					cnt++;
				}
				localCnt -= 1;
			}

			int deltaSegCnt = (++localCnt - historySegCnt);//记录已经采集了多少cluster
			memcpy(featuresBuffer + fBufIdx, temp, deltaSegCnt * fDim * sizeof(double));//复制聚类后的数据
			maskData[i * 2] = fBufIdx / fDim;
			fBufIdx += deltaSegCnt * fDim;
			maskData[i * 2 + 1] = fBufIdx / fDim - 1;
			delete []temp;
		}
		int tail = maskData[segNum * 2 - 1] + 1;
		memcpy(featuresBuffer + fBufIdx, features + tail * fDim, sizeof(double) * fDim * (fNum - tail));//复制尾巴

#ifdef _DEBUG
		for (int i = 0; i != fNum; i++)
		{
			for (int k = 0; k != fDim; k++)
			{
				if (featuresBuffer[i *fDim + k] != features[i*fDim + k])
				{
					int debug = 1;
				}
			}
		}
#endif

		//重新分帧
		cnt = 0;
		for (int i = 0; i < segNum; i++) {
			int begFrame = maskData[i * 2];
			int endFrame = maskData[i * 2 + 1];
			for (int j = begFrame; j <= endFrame; j++) {
				for (int m = -MULTIFRAMES_COLLECT; m <= MULTIFRAMES_COLLECT; m++)
				{
					memcpy(outBuffer + cnt * fDim, featuresBuffer + (j + m) * fDim, sizeof(double) * fDim);
					cnt++;
				}
			}
		}


		delete []featuresBuffer;
		return clusterNum;
	}
private:
	char* mEfBuf;
};
