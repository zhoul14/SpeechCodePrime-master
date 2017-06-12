#include "DTWCluster.h"

int bianrySearch(int a[], int nLength, int val)  
{  
	int start = 0;  
	int end = nLength - 1;  
	int index = -1;  
	while(start<=end)  
	{  
		index = (start+ end)/2;  
		if(a[index] == val)  
		{  
			return index;  
		}  
		else if(a[index] < val)  
		{  
			start = index + 1;  
		}  
		else  
		{  
			end = index -1;  
		}  
	}  

	return -1;  
} 

void DTWCluster::calcCenters(double* ft, bool bInit, int flag = MEAN_CENTER){
	vector<int>& intervals =*m_HistoryMap.rbegin();
	int lastIdx = 0;
	int idx = 0;
	m_vecSimplePointCenters.clear();
	if (bInit){
		//随机初始化，可以不采用（和initDistance同时存在时用随机初始化，无效，不用请注释掉if里面的内容）
		int* tmp = new int[m_nClusterNum];
		getUnrepRandomList(tmp, m_nFrameLen, m_nClusterNum);//产生随机序列，并补充进intervals
		intervals.clear();
		for (int i = 0; i != m_nClusterNum; i++)
		{
			intervals.push_back(tmp[i]);
		}
		*intervals.rbegin()= m_nFrameLen;
		delete[] tmp;
	}

	while(idx < m_nClusterNum){
		int i = lastIdx;
		int clusterId = idx;
		int endId = intervals[idx];

		double* pCenter =  m_pCenters + clusterId * m_nFDim;

		int t = i;
		switch (flag)
		{
		case MEAN_CENTER:
			memset(pCenter, 0, m_nFDim * sizeof(double));
			for (int j = i; j != endId; j++){
				double* tmpFrame = ft + j * m_nFDim;
				for (int k = 0; k != m_nFDim; k++){
					pCenter[k] += tmpFrame[k]/(endId - i);
				}
			}
			break;
		case FIRST_CENTER:	
			{
				memset(pCenter, 0, m_nFDim * sizeof(double));
				double* tmpFrame = ft + i * m_nFDim;
				for (int k = 0; k != m_nFDim; k++){
					pCenter[k] += tmpFrame[k];
				}
				break;

			}
		case MIDDLE_CENTER:
			{
				memset(pCenter, 0, m_nFDim * sizeof(double));
				t = (i + endId)/2;
				double* tmpFrame = ft + t * m_nFDim;
				for (int k = 0; k != m_nFDim; k++){
					pCenter[k] += tmpFrame[k];
				}
				break;
			}
		case RANDOM_CENTER:
			{
				if(endId - i >1)t = i + rand()%(endId - i);
				double* tmpFrame = ft + t * m_nFDim;
				memcpy(pCenter, tmpFrame, sizeof(double) * m_nFDim);
				break;
			}
		case MEDOIDS_CENTER:
			{
				if(bInit || endId - i == 1){//首次随机选取
					if(endId - i >1)t = i + rand()%(endId - i);
					double* tmpFrame = ft + t * m_nFDim;
					memcpy(pCenter, tmpFrame, sizeof(double) * m_nFDim);
					m_vecSimplePointCenters.push_back(t);
				}
				else{//后面找最近点
					double minDis = INT_MAX;
					int closestFrameIdx = -1;
					for (int jj = i; jj != endId; jj++){
						double dis = 0.0f;
						for (int j = i; j != endId; j++){
							if(jj == j)continue;
							double* tmpFrame = ft + j * m_nFDim;
							dis += calcDistance(ft + jj * m_nFDim, (double*)tmpFrame);
						}
						if(minDis > dis){
							closestFrameIdx = jj;
							minDis = dis;
						}
					}
					memcpy(pCenter, ft + closestFrameIdx * m_nFDim, sizeof(double) * m_nFDim);
					m_vecSimplePointCenters.push_back(closestFrameIdx);
				}
				break;
			}
		default:
			break;
		}
		lastIdx = endId;
		idx++;
	}
}
double DTWCluster::calcDistance(double* ft, const int& idx, bool d15){
	double res = 0.0;
	double* d = m_pCenters + idx * m_nFDim;

	if(d15){
		for (int i = 0; i != 14; i++)
		{
			res += (d[i] - ft[i]) * 	(d[i] - ft[i]);
		}
		res += (d[42] - ft[42]) * (d[42] - ft[42]);
	}
	else{
		for (int i = 0; i != m_nFDim; i++)
		{
			res += (d[i] - ft[i]) * 	(d[i] - ft[i]);
		}
	}
	return (res);
}
double DTWCluster::calcDistance(double* ft, double* ft2, bool d15){
	double res = 0.0;
	if(d15){
		auto d = ft2;
		for (int i = 0; i != 14; i++)
		{
			res += (d[i] - ft[i]) * 	(d[i] - ft[i]);
		}
		res += (d[42] - ft[42]) * (d[42] - ft[42]);
	}
	else{
		for (int i = 0; i != m_nFDim; i++)
		{
			res += (ft2[i] - ft[i]) * (ft2[i] - ft[i]);
		}
	}
	return (res);
}

double DTWCluster::calcAdjionDistance(){
	double res = 0.0f;
	for (int i = 0; i != m_nFrameLen; i++)
	{
		int SegIdx = getClusterIdx(i);
		if(SegIdx != 0){
			double adDis = calcDistance(m_pFeatures + i * m_nFDim, SegIdx - 1);
			res += adDis != 0 ? log(adDis) : 0;
		}
		if(SegIdx != m_nClusterNum - 1){
			double adDis = (calcDistance(m_pFeatures + i * m_nFDim, SegIdx + 1));
			res += adDis != 0 ? log(adDis) : 0;
		}
	}
	return res/m_nFrameLen;
}

double DTWCluster::calcBCdistance(){
	double res = 0.0;
	double minDistance = INT_MAX, maxDistance = 0.0f;
	for (int j = 0; j != m_nClusterNum; j++)
	{
		double* d = m_pCenters + j * m_nFDim;
		for (int k = j + 1; k < m_nClusterNum; k++ )
		{
			double* d2 = m_pCenters + k * m_nFDim;
			double distance = 0.0f;
			for (int i = 0; i != m_nFDim; i++)
			{
				distance  += (d[i] - d2[i]) * 	(d[i] - d2[i]);
			}

			minDistance = min(distance, minDistance);
			maxDistance = max(distance, maxDistance);
			res += sqrt(distance);
		}
	}
	//cout<< "min Btn Distance:["<< minDistance <<"] max Btn Distance:["<<maxDistance<<"]"<<endl;
	return (res / (m_nClusterNum * (m_nClusterNum - 1)) * 2);
}
double DTWCluster::getWCdistance(double* features){
	vector<int>& x = (*m_HistoryMap.rbegin());
	int lastIdx = 0;
	double res = 0.0f, minDistance = INT_MAX, maxDistance = 0.0f;
	for (int i = 0 ; i != x.size(); i++)
	{
		if (x[i] - lastIdx != 1)
		{
			double distance = 0.0f;

			for (int j = lastIdx; j != x[i]; j++)
			{
				double* d = features + m_nFDim * j;
				distance += calcDistance(d, i)/ (x[i] - lastIdx);
			}
			maxDistance = max(distance, maxDistance);
			minDistance = min(distance, minDistance);
			res += distance;
		}
		lastIdx =  x[i];
	}
	//cout<< "min within Distance:["<< minDistance <<"] max WithIn Distance:["<<maxDistance<<"]"<<endl;
	return res/ m_nClusterNum;
}

int DTWCluster::getClusterIdx(int& idx){
	vector<int>&vec=(*m_HistoryMap.rbegin());

	if (m_vecSegInfo.empty())
	{
		int k = 0;
		for (int j = 0; j != m_nFrameLen; j++)
		{
			while(vec.at(k) < j){
				k++;
			}
			m_vecSegInfo.push_back(k);
		}
	}
	return m_vecSegInfo.at(idx);
}

vector<int> DTWCluster::getClusterInfo(const int& flag) {
	if(flag == MEAN_CENTER)return *m_HistoryMap.rbegin();
	return m_vecSimplePointCenters;
}


float DTWCluster::initDistance(double* fts, const int&fdim, const int& frameLen, const double& thres){
	double* tmpCenterList = new double[frameLen * fdim];
	double* tmpCenter = new double[fdim];
	bool cltFlag = false;//是否聚类
	int cltNum = 0;//聚类帧总数
	int tmpCltNum = 1;//聚类
	memcpy(tmpCenter, fts,  sizeof(double) * fdim);
	vector<int> intervals;//聚类分段信息
	for (int i = 0; i < frameLen; i++)
	{

		if(i != frameLen - 1){
			float d = calcDistance(tmpCenter, fts + i * fdim + fdim)/15;
			if (d < thres)//和后帧对比距离，如果小于门限则判定为同一帧。求和
			{
				cltFlag = true;
				tmpCltNum ++;
				for (int j = 0; j <fdim; j++)tmpCenter[j]+=(fts + i * fdim + fdim)[j];
				continue;
			}
		}
		if (cltFlag)
		{//如果判定为同一聚类帧，取均值，cltFlag置零
			for (int j = 0; j <fdim; j++)tmpCenter[j]/=tmpCltNum;
			tmpCltNum = 1;
			cltFlag = false;
		}
		memcpy(tmpCenterList + cltNum * fdim, tmpCenter, sizeof(double) * fdim);
		if(i != frameLen - 1)memcpy(tmpCenter, fts + i * fdim + fdim, sizeof(double)* fdim);
		++cltNum;
		intervals.push_back(i+1);
	}

	init(cltNum, 30, fdim);

	*intervals.rbegin()= frameLen;
	(*m_HistoryMap.rbegin()) = intervals;//聚类分段信息
	memcpy(m_pCenters, tmpCenterList, fdim * cltNum * sizeof(double));
	delete []tmpCenter;
	delete []tmpCenterList;
	return (float)frameLen/cltNum;
}


void DTWCluster::init(const int & clusterNum, const int& iterNum, const int&fdim)
{
	m_nIterNum = iterNum;
	m_nClusterNum = clusterNum;

	m_pFeatures = nullptr;
	m_pCenters = new double[fdim * clusterNum];
	memset(m_pCenters, 0, fdim * clusterNum * sizeof(double));

	m_vecDistances.resize(clusterNum,INT_MAX);
	m_HistoryMap.resize(clusterNum);
	for (int i = 0; i != m_nClusterNum; i++)
	{
		for (int j = 0; j != i + 1; j++)
		{
			m_HistoryMap[i].push_back(-1);
		}
	}

}

double DTWCluster::doCluster(double* fts,const int& frameLen, int flag, const int* jumpTable , const int& segNum){

	m_nFrameLen = frameLen;
	m_pFeatures = fts;
	float historyDistance = INT_MAX, lastDistance = INT_MAX, tempDistance = 0;
	//历史最佳总距离，上次迭代总距离，当前距离
	int ite, lastIte, sameTime = 0;
	bool randomIte = false;
	vector<int> bestFA;
	for ( ite = 0; ite != m_nIterNum; ite++)
	{
		for (double& i : m_vecDistances)i = INT_MAX;
		m_vecDistances[0] = calcDistance(fts, 0);
		randomIte = false;
		calcCenters(fts, ite == 0, flag);
		for (vector<int>& i : m_HistoryMap)
		{
			for (int& j:i)
			{
				j = -1;
			}
		}
		m_HistoryMap[0][0] = 1;
		for (int i = 1; i != frameLen; i++)
		{
			for (int C = m_nClusterNum - 1; C >0; C--)
			{
				if (m_vecDistances[C - 1] == INT_MAX)
				{
					continue;
				}
				double D = calcDistance(fts + i * m_nFDim, C);
				if (m_vecDistances[C] < m_vecDistances[C - 1] &&
					((jumpTable != nullptr && find(jumpTable, jumpTable + segNum, i) == jumpTable + segNum) || jumpTable == nullptr)
					)//横走更近，且不属于必须跳过帧时，可以横走。否则斜走
				{
					m_vecDistances[C] = m_vecDistances[C] + D;
				}
				else{
					m_vecDistances[C] = m_vecDistances[C - 1] + D;
					for (int j = 0; j != C; j++)//更新路径
					{
						m_HistoryMap[C][j] = m_HistoryMap[C-1][j];
					}
				}

				m_HistoryMap[C][C] = i + 1;
			}
			m_vecDistances[0] = m_vecDistances[0] + calcDistance(fts + i * m_nFDim, 0);
			m_HistoryMap[0][0] = i + 1;
		}//DTW结束

		float sumD = m_vecDistances[m_nClusterNum - 1];
		if(abs(sumD - lastDistance)<10e-8){
			//本次迭代和上次迭代，总距离相差小于10e-8则停止迭代
			sameTime++;
			if (sameTime > 1){
				break;
			}
		}
		tempDistance =  ite != 0 ? tempDistance  : sumD;
		lastDistance = sumD;
		if(historyDistance>sumD && randomIte == false && ite != 0){
			//保存历史最佳路径
			historyDistance =  min(sumD,historyDistance);
			bestFA.clear();
			bestFA.assign((*m_HistoryMap.rbegin()).begin(),(*m_HistoryMap.rbegin()).end());
			lastIte = ite;
		}

	}
	(*m_HistoryMap.rbegin()).assign(bestFA.begin(),bestFA.end());
	return 0;
}
void DTWCluster::getUnrepRandomList(int* outVect, const int &n, const int &outLen){
	int i;
	int* a = new int[n];
	for(i = 0; i < n; ++i) a[i]=i;
	for(i = n - 1; i >= 1; --i) swap(a[i], a[rand()%i]);
	for (i = 0; i < outLen; ++i){outVect[i] = a[i];}
	sort(outVect, outVect + outLen);
	delete[]a;
}
DTWCluster::~DTWCluster()
{
	delete []m_pCenters;
}


