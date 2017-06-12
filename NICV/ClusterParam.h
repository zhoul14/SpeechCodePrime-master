#ifndef	ZHOULU_CLT_PERAMETER_H
#define	ZHOULU_CLT_PERAMETER_H

#include <string>
#include <vector>

class CSpeechFile {
	friend class ClusterParam;

private:
	std::string featureFile;

	std::string maskFile;

	std::string answerFile;

public:
	std::string getFeatureFileName() {
		return featureFile;
	}

	std::string getMaskFileName() {
		return maskFile;
	}

	std::string getAnswerFileName() {
		return answerFile;
	}

};

class ClusterParam {
private:
	int fileNum;//文件个数

	int fDim;//维度

	double rate;//DTW 压缩率rate

	double threshold;//NICV 门限

	std::string saveDir;//保存的路径

	bool useDTW;//是否使用DTW

	bool useNICV;//是否使用NICV

	bool useHalfLen;//是否使用半帧长

	std::vector<CSpeechFile> fileSets;

	void checkExist(const char* str);

	void checkExist(const std::string& str);

	void trim(std::string& s);

public:
	ClusterParam(const char* filename);

	int getFileNum();//

	std::vector<CSpeechFile> getClusterFiles();

	std::string getSaveDir();

	int getFdim();

	bool getDTWflag();

	bool getHalfLen();

	bool getNICVflag();

	double getRate();

	double getThres();
};

#endif