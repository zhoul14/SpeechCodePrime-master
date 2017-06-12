#include "ClusterParam.h"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
void ClusterParam::checkExist(const char* str) {
	if (GetFileAttributes(str) == INVALID_FILE_ATTRIBUTES) {
		printf("path %s doesn't exists\n", str);
		exit(-1);
	}
	return;
}

void ClusterParam::checkExist(const std::string& str) {
	checkExist(str.c_str());
}

void ClusterParam::trim(std::string& s) {
	const char* endstr = " \r\n\t";
	s.erase(s.find_last_not_of(endstr)+1);
	s.erase(0, s.find_first_not_of(endstr));
}

ClusterParam::ClusterParam(const char* filename) {
	checkExist(filename);
	std::ifstream infile(filename);
	if (!infile) {
		printf("can't open rec config file [%s]\n", filename);
		exit(-1);
	}

	bool startParam = false;
	std::string line;
	while (getline(infile, line)) {
		trim(line);
		if (line == "")
			continue;

		if (line == "PARAM:") {
			startParam = true;
			break;
		}
	}
	if (!startParam) {
		printf("PARAM label not found in config file\n");
		exit(-1);
	}

	bool startData = false;
	std::unordered_map<std::string, std::string> config;
	
	while (getline(infile, line)) {
		trim(line);
		if (line == "") {
			continue;
		}
		if (line == "DATA:") {
			startData = true;
			break;
		}
		std::string pname, pvalue;
		std::stringstream ss(line);
		ss >> pname >> pvalue;
		printf("pname = %s, pvalue = %s\n", pname.c_str(), pvalue.c_str());
		trim(pname);
		trim(pvalue);
		std::transform(pname.begin(), pname.end(), pname.begin(), ::toupper);
		config[pname] = pvalue;
	}
	if (!startData) {
		printf("DATA label not found in config file\n");
		exit(-1);
	}
	while (getline(infile, line)) {
		trim(line);
		if (line == "")
			continue;

		std::stringstream ss(line);
		CSpeechFile f;
		std::string tmp = "";
		ss >> f.featureFile >> f.answerFile >> f.maskFile;
		trim(f.featureFile);
		trim(f.answerFile);
		trim(f.maskFile);
		fileSets.push_back(f);
	}
	infile.close();

	fileNum = std::stoi(config["NUM"]);
	if (fileNum <= 0) {
		printf("RECNUM = %d, must > 0\n", fileNum);
		exit(-1);
	}
	if (fileNum < fileSets.size())
		fileSets.resize(fileNum);
	fDim = config["FDIM"] == "" ? 45 :std::stoi(config["FDIM"]);
	useDTW = config["DTW"]=="" ? 0 :std::stoi(config["DTW"]);
	useNICV = config["NICV"]=="" ? 1 :std::stoi(config["NICV"]);
	rate = config["RATE"] == ""? 1 : std::stof(config["RATE"]);
	threshold = config["THRESHOLD"] == ""? 0.6 : std::stof(config["THRESHOLD"]);
	saveDir = config["SAVEDIR"];
	useHalfLen = config["HALFLEN"] == ""? 0 : std::stof(config["HALFLEN"]);
}

int ClusterParam::getFileNum() {
	return fileNum;
}

std::string ClusterParam::getSaveDir(){
	return saveDir;
}

int ClusterParam::getFdim(){
	return fDim;
}

std::vector<CSpeechFile> ClusterParam::getClusterFiles() {
	return fileSets;
}

bool ClusterParam::getDTWflag(){
	return useDTW;
}

bool ClusterParam::getNICVflag(){
	return useNICV;
}

double ClusterParam::getRate(){
	return rate;
}

double ClusterParam::getThres(){
	return threshold;
}

bool ClusterParam::getHalfLen(){
	return useHalfLen;
}