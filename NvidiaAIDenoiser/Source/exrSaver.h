/**
Copyright (C) <2023>  <Blossom> <Dezeming>

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


#ifndef __EXRSaver_H__
#define __EXRSaver_H__

#include <iostream>
#include <string>
#include <vector>

#include <gdt/gdt.h>
#include <gdt/math/vec.h>
#include <vector_types.h>

#include "DataInfo.h"

namespace exrLoad {

	int float2exr(std::string outputFile, ExrDataStructure& datastruct, const gdt::vec2i& framesize);

	

}




#endif





