/**
Copyright (C) <2023>  <Dezeming> <Blossom>
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

#ifndef __EXRLOAD_H__
#define __EXRLOAD_H__


#include <iostream>
#include <string>
#include <vector>

#include <gdt/gdt.h>
#include <gdt/math/vec.h>
#include <vector_types.h>

#include "DataInfo.h"

namespace exrLoad {

	
	/** load GBuffers from .exr file generated from pbrt
	*
	*/
	int loadGBuffers(std::string filename, gdt::vec2i& resolution, ExrDataStructure& datastruct, bool illumOnly = false);

	/** load RGB from .exr file generated from pbrt
	* (There is currently no maintenance available)
	*/
	int loadRGB(std::string filename, gdt::vec2i& resolution, std::vector<float4*>& dataPointers);


	int AlbedoVisualize(ExrDataStructure& datastruct, uint32_t h_pixels[]);

	int NormalVisualize(ExrDataStructure& datastruct, uint32_t h_pixels[]);

	int PositionVisualize(ExrDataStructure& datastruct, uint32_t h_pixels[]);

}


#endif



