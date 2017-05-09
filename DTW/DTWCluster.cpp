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
	vector<int> intervals;
	if (bInit){
		return;
		int* tmp = new int[m_nClusterNum];
		getUnrepRandomList(tmp, m_nFrameLen, m_nClusterNum);
		for (int i = 0; i != m_nClusterNum; i++)
		{
			intervals.push_back(tmp[i]);
		}
		*intervals.rbegin()= m_nFrameLen;
		delete[] tmp;
		/*
		int interval = m_nFrameLen/ m_nClusterNum;
		if(interval >= 2){
		for (int i = 0; i + 2 * interval<= m_nFrameLen; i+= interval){
		int endId = interval + i;
		intervals.push_back(endId);
		}
		intervals.push_back(m_nFrameLen);
		}
		else{
		int y = m_nFrameLen - m_nClusterNum; //x + 2y = mnFramelen; x + y = mnclusternum;
		int x = m_nClusterNum - y;
		int i = 0;
		int beginId = 0;
		while(i < m_nClusterNum)
		{
		if (x!=0)
		{
		x--;
		intervals.push_back(++beginId);		
		i++;
		}
		if (y!=0)
		{
		y--;
		beginId +=2;
		intervals.push_back(beginId);					
		i++;
		}
		}
		}*///平均初始点
	}
	else{
		intervals = (*m_HistoryMap.rbegin());
	}
	int lastIdx = 0;
	int idx = 0;
	m_vecSimplePointCenters.clear();

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
	bool cltFlag = false;
	int cltNum = 0, i = 0;
	int tmpCltNum = 0;
	memcpy(tmpCenter, fts,  sizeof(double) * fdim);
	for (i = 0; i < frameLen; i++)
	{
		if(i != frameLen - 1){
			float d = calcDistance(fts + i * fdim, fts + i * fdim + fdim)/15;
			if (d < thres)
			{
				cltFlag = true;
				tmpCltNum ++;
				for (int j = 0; j <fdim; j++)tmpCenter[j]+=(fts + i * fdim + fdim)[j];
				continue;
			}
		}
		if (cltFlag)
		{
			for (int j = 0; j <fdim; j++)tmpCenter[j]/=tmpCltNum;
			tmpCltNum = 0;
			cltFlag = false;
		}
		memcpy(tmpCenterList + cltNum * fdim, tmpCenter, sizeof(double) * fdim);
		if(i != frameLen - 1)memcpy(tmpCenter, fts + i * fdim + fdim, sizeof(double)* fdim);
		++cltNum;
	}

	//printf("rate:%f\n", (float)frameLen/cltNum);
	init(cltNum, 30, fdim);
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
	float historyDistance = INT_MAX, LASTdISTANCE = INT_MAX, FirstDistance = 0, lastJf = 0.0f;
	double Wdistance;
	int ite, lastIte, sameTime = 0;
	bool randomIte = false;
	vector<int> bestFA;
	//for (int jj = 0; jj != 50; jj++)//随机初始化
	for ( ite = 0; ite != m_nIterNum; ite++)
	{
		for (double& i : m_vecDistances)i = INT_MAX;
		m_vecDistances[0] = calcDistance(fts, 0);
		randomIte = false;
		calcCenters(fts, ite == 0, flag);
		/*switch (flag)
		{
		case RANDOM_CENTER:
		{
		if (sameTime>2)
		{
		sameTime = 0;
		calcCenters(fts, ite == 0, RANDOM_CENTER);
		randomIte = true;
		}
		else		
		calcCenters(fts, ite == 0, MEAN_CENTER);
		break;
		}
		case MEAN_CENTER:
		calcCenters(fts, ite == 0, MEAN_CENTER);
		break;
		case  MIDDLE_CENTER:
		calcCenters(fts, ite == 0, MIDDLE_CENTER);
		break;
		case  FIRST_CENTER:
		calcCenters(fts, ite == 0, FIRST_CENTER);
		break;
		default:
		break;
		}*/

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
		}
		float sumD = m_vecDistances[m_nClusterNum - 1];
		//Wdistance = getWCdistance(fts);
		
		//double Bdistance = calcBCdistance();
		//double Jf = log(Bdistance) - log(Wdistance + 1) ;
		//lastJf = Jf;
		//cout<<Wdistance<<endl;
		//cout << "[JF]:"<< Jf << endl;
		if(abs(sumD - LASTdISTANCE)<10e-8){
			sameTime++;
			if (sameTime > 1){
				//cout << "[JF]:"<< Jf << endl;
				break;
			}
		}
		FirstDistance =  ite != 0 ? FirstDistance  : sumD;
		LASTdISTANCE = sumD;
		if(historyDistance>sumD && randomIte == false && ite != 0){
			historyDistance =  min(sumD,historyDistance);
			bestFA.clear();
			bestFA.assign((*m_HistoryMap.rbegin()).begin(),(*m_HistoryMap.rbegin()).end());
			lastIte = ite;
		}

	}
	(*m_HistoryMap.rbegin()).assign(bestFA.begin(),bestFA.end());
	return /*Wdistance*/0;//calcAdjionDistance();
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


