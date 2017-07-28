#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


const char* loadTextFromFile(const char* pathToFile) {
	std::ifstream textFile;
	std::string textFileContent;

	textFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		textFile.open(pathToFile);

		std::stringstream textFileStream;
		textFileStream << textFile.rdbuf();

		textFile.close();

		textFileContent = textFileStream.str();
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR: FILE LOADER: Cannot open " << pathToFile << std::endl;
	}

	std::cout << textFileContent << std::endl;

	return textFileContent.c_str();
};