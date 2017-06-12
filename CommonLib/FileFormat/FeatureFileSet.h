#ifndef	WJQ_FEATURE_READER_H
#define	WJQ_FEATURE_READER_H

#include <stdio.h>
#include <string>
#include <vector>
struct FeatureIndex {
	long	offset;
	long	byteSize;
};

struct MaskIndex {
	long	offset;
	long	endpointNum;
};
struct ClusterIndex {
	long	offset;
	long	ClusterNum;
};
struct SegPointIndex{
	long	offset;
	long	SegPtNum;
};

struct AnswerIndex{
	long	offset;
	long	wordNum;
};

class FeatureFileSet {
private:

	char* featureFileBuf;

	char* maskFileBuf;

	char* answerFileBuf;

	int speechNum;

	int featureDim;

	std::vector<SegPointIndex>SegPtList;
	std::vector<std::vector<int>>SegmentpoitData;

public:
	FeatureFileSet(const std::string& featureFileName, const std::string& maskFileName, const std::string& answerFileName, int featureDim);

	int fileByteNum(FILE* f);

	int getSpeechNum();

	void printFeatureFileToTxt(const char * filename);

	//获取帧特征序列
	void getSpeechAt(int speechIdx, double* outputBuffer);

	//获取帧数
	int getFrameNumInSpeech(int speechIdx);

	//获取答案字数
	int getWordNumInSpeech(int speechIdx);

	//获取多帧特征序列
	void getMultiSpeechAt(int speechIdx, double* outputBuffer);

	//获取答案表
	void getWordListInSpeech(int speechIdx, int* outputWordList);

	//保存分割点
	void SaveSegmentPointToBuf(int speechIdx, std::vector<int> &res);

	void PrintSegmentPointBuf(std::string fileName);

	//获得有无声音段的信息
	void getMaskData(int* buffer, const int& speechIdx);

	//获取不加mask的帧数
	int getPrimeFrameLen(const int& speechIdx);

	void getPrimeSpeechAt(int speechIdx, double* outputBuffer);

	int getMaskLen(const int& speechIdx);

	~FeatureFileSet();

	//获取5ms帧移的特征矢量
	void getSpeechAt_half_framelen(int speechIdx, double* outputBuffer, int e);
	
	//获取5ms帧移的帧数
	int getFrameNumInSpeech_half_framelen(int speechIdx);
};


#endif