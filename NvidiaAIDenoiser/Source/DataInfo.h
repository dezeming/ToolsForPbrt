/**
Copyright (C) <2023>  <Dezeming>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef __DATAINFO_H__
#define __DATAINFO_H__

#include <iostream>
#include <string>
#include <vector>

inline bool isNameInVector(const std::vector<std::string> &typeNames, std::string name) {

	for (int i = 0; i < typeNames.size(); i++) {
		if (name == typeNames[i]) {
			return true;
		}
	}
	return false;
}

inline int getNameIndexInVector(const std::vector<std::string>& typeNames, std::string name) {

	for (int i = 0; i < typeNames.size(); i++) {
		if (name == typeNames[i]) {
			return i;
		}
	}
	return -1;
}

struct ChannelType {
	ChannelType() {
		reset();
	}
	void reset() {

		typeNames.clear();

		hasColor = false;
		hasAlbedo = false;
		hasNormal = false;
		hasShadingNormal = false;
		hasPosition = false;
		hasVariance = false;
		hasRelativeVariance = false;
		hasTangent = false;
		hasTextureUV = false;
		hasRoughness = false;

		ColorChannels = 0;
		AlbedoChannels = 0;
		NormalChannels = 0;
		ShadingNormalChannels = 0;
		PositionChannels = 0;
		VarianceChannels = 0;
		RelativeVarianceChannels = 0;
		TangentChannels = 0;
		TextureUVChannels = 0;
		RoughnessChannels = 0;
	}

	void setChannel(std::string channelName) {
		// Color
		if (channelName == "R") { hasColor = true; ColorChannels++; if(!isNameInVector(typeNames, "Color"))typeNames.push_back("Color"); }
		else if (channelName == "G") { hasColor = true; ColorChannels++; if(!isNameInVector(typeNames, "Color"))typeNames.push_back("Color"); }
		else if (channelName == "B") { hasColor = true; ColorChannels++; if(!isNameInVector(typeNames, "Color"))typeNames.push_back("Color");}
		// Alebdo
		else if (channelName == "Albedo.R") { hasAlbedo = true; AlbedoChannels++; if(!isNameInVector(typeNames, "Albedo"))typeNames.push_back("Albedo");}
		else if (channelName == "Albedo.G") { hasAlbedo = true; AlbedoChannels++; if(!isNameInVector(typeNames, "Albedo"))typeNames.push_back("Albedo");}
		else if (channelName == "Albedo.B") { hasAlbedo = true; AlbedoChannels++; if(!isNameInVector(typeNames, "Albedo"))typeNames.push_back("Albedo");}
		// Normal
		else if (channelName == "N.X") { hasNormal = true; NormalChannels++; if(!isNameInVector(typeNames, "Normal"))typeNames.push_back("Normal");}
		else if (channelName == "N.Y") { hasNormal = true; NormalChannels++; if(!isNameInVector(typeNames, "Normal"))typeNames.push_back("Normal"); }
		else if (channelName == "N.Z") { hasNormal = true; NormalChannels++; if(!isNameInVector(typeNames, "Normal"))typeNames.push_back("Normal");}
		// Position
		else if (channelName == "P.X") { hasPosition = true; PositionChannels++; if(!isNameInVector(typeNames, "Position"))typeNames.push_back("Position");}
		else if (channelName == "P.Y") { hasPosition = true; PositionChannels++; if(!isNameInVector(typeNames, "Position"))typeNames.push_back("Position"); }
		else if (channelName == "P.Z") { hasPosition = true; PositionChannels++; if(!isNameInVector(typeNames, "Position"))typeNames.push_back("Position");}
		// ShadingNormal
		else if (channelName == "Ns.X") { hasShadingNormal = true; ShadingNormalChannels++; if(!isNameInVector(typeNames, "ShadingNormal"))typeNames.push_back("ShadingNormal");}
		else if (channelName == "Ns.Y") { hasShadingNormal = true; ShadingNormalChannels++; if(!isNameInVector(typeNames, "ShadingNormal"))typeNames.push_back("ShadingNormal");}
		else if (channelName == "Ns.Z") { hasShadingNormal = true; ShadingNormalChannels++; if(!isNameInVector(typeNames, "ShadingNormal"))typeNames.push_back("ShadingNormal");}
		// Variance
		else if (channelName == "Variance.R") { hasVariance = true; VarianceChannels++; if(!isNameInVector(typeNames, "Variance"))typeNames.push_back("Variance");}
		else if (channelName == "Variance.G") { hasVariance = true; VarianceChannels++; if(!isNameInVector(typeNames, "Variance"))typeNames.push_back("Variance");}
		else if (channelName == "Variance.B") { hasVariance = true; VarianceChannels++; if(!isNameInVector(typeNames, "Variance"))typeNames.push_back("Variance");}
		// RelativeVariance
		else if (channelName == "RelativeVariance.R") { hasRelativeVariance = true; RelativeVarianceChannels++; if(!isNameInVector(typeNames, "RelativeVariance"))typeNames.push_back("RelativeVariance");}
		else if (channelName == "RelativeVariance.G") { hasRelativeVariance = true; RelativeVarianceChannels++; if(!isNameInVector(typeNames, "RelativeVariance"))typeNames.push_back("RelativeVariance");}
		else if (channelName == "RelativeVariance.B") { hasRelativeVariance = true; RelativeVarianceChannels++; if(!isNameInVector(typeNames, "RelativeVariance"))typeNames.push_back("RelativeVariance");}
		// Tangent
		else if (channelName == "dzdx") { hasTangent = true; TangentChannels++; if(!isNameInVector(typeNames, "Tangent"))typeNames.push_back("Tangent");}
		else if (channelName == "dzdy") { hasTangent = true; TangentChannels++; if(!isNameInVector(typeNames, "Tangent"))typeNames.push_back("Tangent");}
		// Roughness
		else if (channelName == "Roughness") { hasRoughness = true; VarianceChannels++; if(!isNameInVector(typeNames, "Roughness"))typeNames.push_back("Roughness");}
		// TextureUV
		else if (channelName == "u") { hasTextureUV = true; TextureUVChannels++; if(!isNameInVector(typeNames, "TextureUV"))typeNames.push_back("TextureUV");}
		else if (channelName == "v") { hasTextureUV = true; TextureUVChannels++; if(!isNameInVector(typeNames, "TextureUV"))typeNames.push_back("TextureUV");}	
	}
	void getTypeOrder(std::string channelName, int& typeIndex, int& channel) {
		// Color
		if (channelName == "R") { typeIndex = getNameIndexInVector(typeNames, "Color"); channel = 0; }
		else if (channelName == "G") { typeIndex = getNameIndexInVector(typeNames, "Color"); channel = 1; }
		else if (channelName == "B") { typeIndex = getNameIndexInVector(typeNames, "Color"); channel = 2; }
		// Alebdo
		else if (channelName == "Albedo.R") { typeIndex = getNameIndexInVector(typeNames, "Albedo"); channel = 0; }
		else if (channelName == "Albedo.G") { typeIndex = getNameIndexInVector(typeNames, "Albedo"); channel = 1; }
		else if (channelName == "Albedo.B") { typeIndex = getNameIndexInVector(typeNames, "Albedo"); channel = 2; }
		// Normal
		else if (channelName == "N.X") { typeIndex = getNameIndexInVector(typeNames, "Normal"); channel = 0; }
		else if (channelName == "N.Y") { typeIndex = getNameIndexInVector(typeNames, "Normal"); channel = 1; }
		else if (channelName == "N.Z") { typeIndex = getNameIndexInVector(typeNames, "Normal"); channel = 2; }
		// Position
		else if (channelName == "P.X") { typeIndex = getNameIndexInVector(typeNames, "Position"); channel = 0; }
		else if (channelName == "P.Y") { typeIndex = getNameIndexInVector(typeNames, "Position"); channel = 1; }
		else if (channelName == "P.Z") { typeIndex = getNameIndexInVector(typeNames, "Position"); channel = 2; }
		// ShadingNormal
		else if (channelName == "Ns.X") { typeIndex = getNameIndexInVector(typeNames, "ShadingNormal"); channel = 0; }
		else if (channelName == "Ns.Y") { typeIndex = getNameIndexInVector(typeNames, "ShadingNormal"); channel = 1; }
		else if (channelName == "Ns.Z") { typeIndex = getNameIndexInVector(typeNames, "ShadingNormal"); channel = 2; }
		// Variance
		else if (channelName == "Variance.R") { typeIndex = getNameIndexInVector(typeNames, "Variance"); channel = 0; }
		else if (channelName == "Variance.G") { typeIndex = getNameIndexInVector(typeNames, "Variance"); channel = 1; }
		else if (channelName == "Variance.B") { typeIndex = getNameIndexInVector(typeNames, "Variance"); channel = 2; }
		// RelativeVariance
		else if (channelName == "RelativeVariance.R") { typeIndex = getNameIndexInVector(typeNames, "RelativeVariance"); channel = 0; }
		else if (channelName == "RelativeVariance.G") { typeIndex = getNameIndexInVector(typeNames, "RelativeVariance"); channel = 1; }
		else if (channelName == "RelativeVariance.B") { typeIndex = getNameIndexInVector(typeNames, "RelativeVariance"); channel = 2; }
		// Tangent
		else if (channelName == "dzdx") { typeIndex = getNameIndexInVector(typeNames, "Tangent"); channel = 0; }
		else if (channelName == "dzdy") { typeIndex = getNameIndexInVector(typeNames, "Tangent"); channel = 1; }
		// Roughness
		else if (channelName == "Roughness") { typeIndex = getNameIndexInVector(typeNames, "Roughness"); channel = 0; }
		// TextureUV
		else if (channelName == "u") { typeIndex = getNameIndexInVector(typeNames, "TextureUV"); channel = 0; }
		else if (channelName == "v") { typeIndex = getNameIndexInVector(typeNames, "TextureUV"); channel = 1; }
		else {
			typeIndex = -1;
			channel = -1;
		}
	}

	int getTypeOrder(std::string TypeName) {
		for (int i = 0; i < typeNames.size(); i++) {
			if (TypeName == typeNames[i]) {
				return i;
			}
		}
		return -1;
	}

	std::vector<std::string> typeNames;
	void PrintTypeNames() {
		std::cout << "Type Names : ";
		for (int i = 0; i < typeNames.size(); i++) {
			std::cout << typeNames[i] << " ";
		}
		std::cout << std::endl;
	}

	bool hasColor;
	int ColorChannels;
	bool hasAlbedo;
	int AlbedoChannels;
	bool hasNormal;
	int NormalChannels;
	bool hasPosition;
	int PositionChannels;
	bool hasShadingNormal;
	int ShadingNormalChannels;
	bool hasVariance;
	int VarianceChannels;
	bool hasRelativeVariance;
	int RelativeVarianceChannels;
	bool hasTangent;
	int TangentChannels;
	bool hasTextureUV;
	int TextureUVChannels;
	bool hasRoughness;
	int RoughnessChannels;

	int getTypesCount() {
		int count = 0;
		if (hasColor) ++count;
		if (hasAlbedo) ++count;
		if (hasNormal) ++count;
		if (hasPosition) ++count;
		if (hasShadingNormal) ++count;
		if (hasVariance) ++count;
		if (hasRelativeVariance) ++count;
		if (hasTangent) ++count;
		if (hasTextureUV) ++count;
		if (hasRoughness) ++count;
		return count;
	}

	int getChannelsCount() {
		int count = 0;
		count += ColorChannels;
		count += AlbedoChannels;
		count += NormalChannels;
		count += PositionChannels;
		count += ShadingNormalChannels;
		count += VarianceChannels;
		count += RelativeVarianceChannels;
		count += TangentChannels;
		count += TextureUVChannels;
		count += RoughnessChannels;
		return count;
	}

	void PrintInfo() {
		std::cout << "ChannelType" << std::endl;
		if (hasColor) std::cout << "hasColor-Channels = [ " << ColorChannels << " ]" << std::endl;
		if (hasAlbedo) std::cout << "hasAlbedo-Channels = [ " << AlbedoChannels << " ]" << std::endl;
		if (hasNormal) std::cout << "hasNormal-Channels = [ " << NormalChannels << " ]" << std::endl;
		if (hasShadingNormal) std::cout << "hasShadingNormal-Channels = [ " << ShadingNormalChannels << " ]" << std::endl;
		if (hasPosition) std::cout << "hasPosition-Channels = [ " << PositionChannels << " ]" << std::endl;
		if (hasVariance) std::cout << "hasVariance-Channels = [ " << VarianceChannels << " ]" << std::endl;
		if (hasRelativeVariance) std::cout << "hasVariance-Channels = [ " << RelativeVarianceChannels << " ]" << std::endl;
		if (hasTangent) std::cout << "hasTangent-Channels = [ " << TangentChannels << " ]" << std::endl;
		if (hasTextureUV) std::cout << "hasTextureUV-Channels = [ " << TextureUVChannels << " ]" << std::endl;
		if (hasRoughness) std::cout << "hasRoughness-Channels = [ " << RoughnessChannels << " ]" << std::endl;
	}

};

struct ExrDataStructure {
	ChannelType channeltype;
	std::vector<float4*> dataPointers;
	gdt::vec2i ImageSize;

	void clear() {
		for (int i = 0; i < dataPointers.size(); i++) {
			delete[] dataPointers[i];
			dataPointers[i] = nullptr;
		}
		dataPointers.clear();
		channeltype.reset();
		ImageSize.x = 0;
		ImageSize.y = 0;
	}
};




#endif





