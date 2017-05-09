#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
class DTWCluster
{
public:
	DTWCluster(int x){
		m_nFDim = x;
	}
	~DTWCluster();
	double doCluster(double* fts, const int& frameLen, int flag = RANDOM_CENTER, const int* jumpTable = NULL, const int& segNum = 0);
	void init(const int & clusterNum, const int& iterNum, const int&fdim);
	float initDistance(double* fts, const int&fdim, const int& frameLen, const double& thes);
	void getUnrepRandomList(int* outVect, const int &n, const int &outLen);
	
	vector<int> getClusterInfo(const int& flag);

	double calcAdjionDistance();
	static const int MEAN_CENTER = 0;
	static const int FIRST_CENTER = 1;
	static const int MIDDLE_CENTER = 2;
	static const int RIGHT_CENTER = 3;
	static const int RANDOM_CENTER = 4;
	static const int MEDOIDS_CENTER = 5;
private:
	int m_nFDim;
	int m_nIterNum;
	int m_nFrameLen;
	int m_nClusterNum;



	double* m_pFeatures;
	double* m_pCenters;

	vector<double> m_vecDistances;
	vector<int> m_vecSimplePointCenters;
	vector<int> m_vecSegInfo;
	vector<vector<int>>m_HistoryMap;

	void calcCenters(double* fts, bool bInit, int flag);
	double calcDistance(double* ft, const int& idx, bool d15 = true);
	double calcDistance(double* ft, double* ft2, bool d15= true);
	double calcBCdistance();
	double getWCdistance(double* features);
	int getClusterIdx(int& idx);


};

