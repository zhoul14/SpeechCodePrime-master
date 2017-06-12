#ifndef	WJQ_REC_PERAMETER_H
#define	WJQ_REC_PERAMETER_H

#include <string>
#include <vector>

class RSpeechFile {
	friend class RecParam;

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

class RecParam {
private:
	std::string codebookFileName;

	std::string wordDictFileName;

	std::string cltDirName;

	std::string NNmodelName;

	int recNum;

	int multiJump;

	int useCuda;

	int usePy;

	int useDrop;

	int useHalfLen;

	int fDim;

	bool triPhone;

	std::vector<RSpeechFile> fileSets;

	double durWeight;

	int useSegmentModel;

	bool m_bHeadNOise;

	int useCluster;

	void checkExist(const char* str);

	void checkExist(const std::string& str);

	void trim(std::string& s);

public:
	RecParam(char* filename);

	std::string getCodebookFileName();

	std::string getWordDictFileName();

	std::string getCltDirName();

	int getRecNum();

	int getMultiJump();

	bool getUseCudaFlag();

	bool getSegmentModelFlag();

	double getDurWeight();

	bool getUseHalfLenFlag();

	bool getUseDropFlag();

	bool getUsePYFlag();

	std::vector<RSpeechFile> getRecFiles();

	bool getTriPhone();

	int getFdim();

	bool getBHeadNoise();

	std::string getNNmodelName();

	int getClusterFlag();
};

#endif