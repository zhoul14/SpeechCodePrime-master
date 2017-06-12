#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>
using namespace std;
class DisCluster
{
public:
	DisCluster(int x){
		m_nFDim = x;
	}
	~DisCluster(){

	}
	//NICV聚类
	double doCluster(double* fts, const int& frameLen, int flag = MEAN_CENTER, const int* jumpTable = NULL){

		m_dVar = 0;
		m_dEnergy = 0;
		int clustClt = 1, p = 0;
		segPoints.clear();

		double* tmpCenter = new double[m_nFDim];
		memcpy(tmpCenter, fts, m_nFDim * sizeof(double));
		m_dEnergy = calcEnergy(fts);

		for (int i = 1; i != frameLen; i++)
		{
			double* pFrame = fts + i * m_nFDim;
			for (int j = 0; j != m_nFDim; j++)
			{
				tmpCenter[j] = ((tmpCenter[j] * clustClt) + pFrame[j])/(clustClt + 1);
			}//先更新，如果不是后面memcpy覆盖掉。

			double res = calcVar(fts + p * m_nFDim, tmpCenter, i - p + 1);
			if(res < m_dK  &&clustClt < m_maxClusterClt){
				m_dEnergy += m_dtmpEnergy;
				clustClt++;
			}
			else{
				segPoints.push_back(i);
				clustClt = 1;
				memcpy(tmpCenter, fts + i * m_nFDim, m_nFDim * sizeof(double));
				m_dEnergy = m_dtmpEnergy;
				p = i;
			}
		}
		if(*segPoints.rbegin() != frameLen)segPoints.push_back(frameLen);

		delete []tmpCenter;

		return ((double)frameLen/segPoints.size());
	}
	//初始化
	void init(const int&fdim, const double& k, const int& maxClusterClt){
		m_nFDim = fdim;
		m_dK = k;
		m_maxClusterClt = maxClusterClt;
	}
	//计算feature里的总能量
	double calcEnergy(double* features){
		double energy = 0.0f;
		for (int i = 0; i != 14; i++)
		{
			energy += features[i ] * features[i];
		}
		return energy;
	}
	//计算方差
	double calcVar(double* features, double* center, int clusterNum){
		double energy = calcEnergy(features + (clusterNum - 1) * m_nFDim);
		double dis = 0.0f;
		for (int k = 0; k != clusterNum; k++)
		{

			for (int i = 0; i != 14; i++)
			{
				dis += (features[i + k * m_nFDim] - center[i])* (features[i + k * m_nFDim] - center[i]);
			}
		}
		m_dtmpEnergy = energy;

		auto res = dis/(energy + m_dEnergy);
		return res;
	}
	//计算NICV，目前没用到
	double calcNICV(double* features, double* center){
		double energy = 0.0f;
		for (int i = 0; i != 14; i++)
		{
			energy += features[i ] * features[i];
		}
		double dis = 0.0f;
		for (int i = 0; i != 14; i++)
		{
			dis += (features[i] - center[i])* (features[i] - center[i]);
		}
		m_dtmpEnergy = energy;
		m_dtmpVar = dis;

		auto res = (dis + m_dVar)/(energy + m_dEnergy);		
		return res;
	}
	vector<int> getClusterInfo(){
		return segPoints;
	}

	static const int MEAN_CENTER = 0;

private:
	int m_nFDim;
	int m_maxClusterClt;

	double m_dK;
	double m_dEnergy;
	double m_dVar;
	double m_dtmpEnergy;
	double m_dtmpVar;
	vector<int>segPoints;


};

