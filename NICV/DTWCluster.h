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

	//对输入的帧序列做聚类
	double doCluster(double* fts, const int& frameLen, int flag = RANDOM_CENTER, const int* jumpTable = NULL, const int& segNum = 0);

	//初始化聚类数和迭代次数以及维度
	void init(const int & clusterNum, const int& iterNum, const int&fdim);


	//通过前后帧间的距离初始化，产生初始代表帧序列。如果小于一定门限则判定为可以聚类
	float initDistance(double* fts, const int&fdim, const int& frameLen, const double& thes);

	//产生最大是n，的outLen长度的 不重复随机整数列。
	//如最大是5，outlen是3，产生的可能是1，2，5
	void getUnrepRandomList(int* outVect, const int &n, const int &outLen);
	
	//获取聚类信息
	vector<int> getClusterInfo(const int& flag= MEAN_CENTER);

	//获取邻接距离
	double calcAdjionDistance();
	static const int MEAN_CENTER = 0;
	static const int FIRST_CENTER = 1;
	static const int MIDDLE_CENTER = 2;
	static const int RIGHT_CENTER = 3;
	static const int RANDOM_CENTER = 4;
	static const int MEDOIDS_CENTER = 5;
private:
	int m_nFDim;
	// 输入特征维数
	int m_nIterNum;
	// 迭代次数
	int m_nFrameLen;
	// 帧长
	int m_nClusterNum;
	// 代表帧长

	double* m_pFeatures;
	// 输入特征序列指针
	double* m_pCenters;
	// 聚类中心序列指针
	vector<double> m_vecDistances;
	// DTW的距离序列
	vector<int> m_vecSimplePointCenters;
	// 单点中心（没啥用）的位置，在做MEdoids算法的时候用的，kmeans不用 
	vector<int> m_vecSegInfo;
	// 聚类分段信息
	vector<vector<int>>m_HistoryMap;
	// 历史路径矩阵

	void calcCenters(double* fts, bool bInit, int flag);
	double calcDistance(double* ft, const int& idx, bool d15 = true);
	double calcDistance(double* ft, double* ft2, bool d15= true);
	double calcBCdistance();
	double getWCdistance(double* features);
	int getClusterIdx(int& idx);


};

