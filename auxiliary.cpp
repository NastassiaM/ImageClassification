#include "auxiliary.h" 
#include <windows.h> 
#include <opencv2/core/core.hpp> 
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


int GetFilesInFolder(const string& dirPath, std::vector<string> &filesList)
{
	HANDLE handle;
	WIN32_FIND_DATAA fileData;
	int count = 0;

	if ((handle = FindFirstFileA((dirPath +	"/*.jpg").c_str(), &fileData)) == INVALID_HANDLE_VALUE)
	{
		return -1;
	}
	do
	{
		const string file_name = fileData.cFileName;
		const string full_file_name = dirPath + "/" + file_name;
		filesList.push_back(full_file_name);
		count++;
	} while (FindNextFileA(handle, &fileData));
	FindClose(handle);

	return count;
}

void InitRandomBoolVector(vector<bool>& mask, double prob)
{
	RNG rng = theRNG();
	for (size_t i = 0; i < mask.size(); i++)
	{
		mask[i] = (rng.uniform(0.0, 1.0) < prob) ? true	: false;
	}
}


void InitBoolVector(std::vector<bool>& mask, double prob, int count1, int count2)
{
	int tresh1 = count1*prob;
	int tresh2 = count2*prob + count1;

	for (int i = 0; i < mask.size(); i++)
	{
		if ((i < tresh1) || (i >= count1 && i < tresh2))
		{
			mask[i] = true;
		}
		else
		{
			mask[i] = false;
		}
	}
}


void ParseInputFile(const string& fileName,
	string& detectorType,
	string& descriptorType)
{
	ifstream in(fileName);
	in >> detectorType;
	in >> descriptorType;
	in.close();
}


void WriteOutputFile(double **mistakes,
					 int n,
					 int m,
					 std::string& detectorType,
					 std::string& descriptorType)
{
	stringstream outName;
	outName << detectorType << "_" << descriptorType << ".txt";

	ofstream out(outName.str());
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			out << mistakes[i][j] << " ";
		}
		out << "\n";
	}
	out.close();
}