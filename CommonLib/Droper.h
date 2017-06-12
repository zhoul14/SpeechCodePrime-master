#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include <string>
using namespace std;
class Droper{
public:
	Droper(string filename, FeatureFileSet& fs, int dropVal){
		string s = filename.substr(filename.find_last_of('/') + 1,filename.find_last_of('.') - filename.find_last_of('/') -1);
		FILE* df = fopen(string("ef/"+ s + ".ef").c_str(),"rb");
		int n = fs.fileByteNum(df);
		mClusterFileBuf = new char[n];
		fread(mClusterFileBuf, 1, n, df);
		fclose(df);
		mDropVal = dropVal;
	};
	~Droper(){
		delete []mClusterFileBuf;
	}

	int getNewFnum(const int& j, const int& primeFnum){
		ClusterIndex* clustidx = (ClusterIndex*)(mClusterFileBuf + sizeof(ClusterIndex) * j);
		float* energyBuf = (float*)(clustidx->offset + mClusterFileBuf);
		int frameLen = clustidx->ClusterNum;
		if(frameLen != primeFnum){
			printf("frameLen:%d != fNum:%d\n",frameLen,primeFnum);
		}
		int newfNum= 0;
		int counter = 0;
		for (int t = 0; t != primeFnum; t++)
		{
			if(energyBuf[t] > mDropVal){
				counter++;
				if (counter%2==0)continue;
			}
			newfNum ++;
		}

		mFNum = newfNum;
		return newfNum;
	}

	int dropFrame(double* features, const int& j, const int &fDim, const int&fNum){
		ClusterIndex* clustidx = (ClusterIndex*)(mClusterFileBuf + sizeof(ClusterIndex) * j);
		float* energyBuf = (float*)(clustidx->offset + mClusterFileBuf);
		int counter = 0;
		double* temp = new double[mFNum * fDim];
		int outCnt = 0;
		for (int t = 0; t !=fNum; t++)
		{

			if(energyBuf[t] > mDropVal){
				counter++;
				if(counter%2 == 0)continue;
			}
			memcpy(temp + (outCnt++) * fDim, features + t * fDim , fDim * sizeof(double));
		}
		memcpy(features, temp, mFNum * fDim * sizeof(double));
		delete[] temp;
		return mFNum;
	}
private:
	int mFNum;
	char* mClusterFileBuf;
	int mDropVal;
};
