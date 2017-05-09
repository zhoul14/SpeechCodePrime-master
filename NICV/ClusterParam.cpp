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
	std::string ansFile, ftFile, maskFile, resFile;
	while (getline(infile, line)) {
		trim(line);
		if (line == "")
			continue;

		std::stringstream ss(line);
		RSpeechFile f;
		ss >> f.featureFile >> f.answerFile >> f.maskFile >> f.recResultFile;
		trim(f.answerFile);
		trim(f.featureFile);
		trim(f.maskFile);
		trim(f.recResultFile);
		fileSets.push_back(f);
	}
	infile.close();

	recNum = std::stoi(config["RECNUM"]);
	if (recNum <= 0) {
		printf("RECNUM = %d, must > 0\n", recNum);
		exit(-1);
	}
	if (recNum < fileSets.size())
		fileSets.resize(recNum);
	fDim = config["FDIM"] == "" ? 45 :std::stoi(config["FDIM"]);

}

int ClusterParam::getRecNum() {
	return recNum;
}

int ClusterParam::getFdim(){
	return fDim;
}

std::vector<RSpeechFile> ClusterParam::getRecFiles() {
	return fileSets;
}
