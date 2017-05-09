#ifndef	ZHOULU_CLT_PERAMETER_H
#define	ZHOULU_CLT_PERAMETER_H

#include <string>
#include <vector>

class RSpeechFile {
	friend class ClusterParam;

private:
	std::string featureFile;

	std::string maskFile;

	std::string answerFile;

	std::string recResultFile;

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

	std::string getRecResultFileName() {
		return recResultFile;
	}
};

class ClusterParam {
private:
	int recNum;

	int fDim;

	std::vector<RSpeechFile> fileSets;

	void checkExist(const char* str);

	void checkExist(const std::string& str);

	void trim(std::string& s);

public:
	ClusterParam(const char* filename);

	int getRecNum();

	std::vector<RSpeechFile> getRecFiles();

	int getFdim();


};

#endif