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

#include <ImfArray.h>
#include <iostream>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <algorithm>
#include <cmath>
#include <ImathColor.h>
#include <ImathColorAlgo.h>
#include <ImfChannelList.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <half.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#define _CRT_SECURE_NO_WARNINGS

#include "exrSaver.h"
#include "exrHeader.h"
#include "DataInfo.h"

#include <gdt/gdt.h>
#include <gdt/math/vec.h>
#include <vector_types.h>


namespace exrLoad {

    // using namespace Imf;
    using Imf::InputFile;
    using Imf::ChannelList;
    using Imf::Header;
    using Imf::FrameBuffer;
    using Imf::HALF;
    using Imf::Channel;
    using Imf::OutputFile;
    using Imf::Slice;

    using Imath::Box2i;
    using Imath::V2i;

    //using namespace std;
    using std::vector;
    using std::string;
    using std::cout;
    using std::endl;


    int float2exr(std::string outputFile, ExrDataStructure& datastruct, const gdt::vec2i& framesize) {

        int height = framesize.y;
        int width = framesize.x;

        // Set channel
        int channelnum = datastruct.channeltype.getChannelsCount();  // channels
        string* channelnames = new string[channelnum]; // channels name
        int count = 0;
        if (datastruct.channeltype.hasColor) { channelnames[count++] = "R";channelnames[count++] = "G"; channelnames[count++] = "B";}
        if (datastruct.channeltype.hasAlbedo){ channelnames[count++] = "Albedo.R";channelnames[count++] = "Albedo.G";channelnames[count++] = "Albedo.B"; }
        if (datastruct.channeltype.hasNormal){ channelnames[count++] = "N.X";channelnames[count++] = "N.Y";channelnames[count++] = "N.Z"; }
        if (datastruct.channeltype.hasPosition) { channelnames[count++] = "P.X"; channelnames[count++] = "P.Y"; channelnames[count++] = "P.Z"; }
        if (datastruct.channeltype.hasVariance) { channelnames[count++] = "Variance.R";channelnames[count++] = "Variance.G";channelnames[count++] = "Variance.B"; }
        if (datastruct.channeltype.hasShadingNormal) { channelnames[count++] = "Ns.X";channelnames[count++] = "Ns.Y";channelnames[count++] = "Ns.Z"; }
        if (datastruct.channeltype.hasRelativeVariance) { channelnames[count++] = "RelativeVariance.R";channelnames[count++] = "RelativeVariance.G";channelnames[count++] = "RelativeVariance.B"; }
        if (datastruct.channeltype.hasTangent) { channelnames[count++] = "dzdx";channelnames[count++] = "dzdy"; }
        if (datastruct.channeltype.hasTextureUV) { channelnames[count++] = "u";channelnames[count++] = "v"; }
        if (datastruct.channeltype.hasRoughness) { channelnames[count++] = "Roughness"; }

        if (count != channelnum) {
            std::cout << "count != channelnum in float2exr(...)." << std::endl;
            return -1;
        }

        
        Header header(width, height);
        Box2i data_window(V2i(0, 0),
            V2i(width - 1, height - 1));
        header.dataWindow() = data_window; //beuark.
        for (int i = 0; i < channelnum; i++) {
            header.channels().insert(channelnames[i], Channel(HALF));
        }

        // float to half, save to halfdata
        vector<half*> halfdata;
        for (int i = 0; i < channelnum; i++) {
            half* onechannel = new half[width * height];
            halfdata.push_back(onechannel);
        }

        for (int i = 0; i < channelnum; i++) {
            int offset = 0;
            int typeIndex, channelIndex;
            datastruct.channeltype.getTypeOrder(channelnames[i], typeIndex, channelIndex);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {

                    if (channelIndex == 0)
                        halfdata[i][offset] = f32_to_f16(datastruct.dataPointers[typeIndex][offset].x);
                    else if (channelIndex == 1)
                        halfdata[i][offset] = f32_to_f16(datastruct.dataPointers[typeIndex][offset].y);
                    else if (channelIndex == 2)
                        halfdata[i][offset] = f32_to_f16(datastruct.dataPointers[typeIndex][offset].z);
                    else if (channelIndex == 3)
                        halfdata[i][offset] = f32_to_f16(datastruct.dataPointers[typeIndex][offset].w);
                    offset++;
                }
            }
        }

        // halfdata input to FrameBuffer
        FrameBuffer fb;
        for (int i = 0; i < channelnum; i++) {
            fb.insert(channelnames[i], Slice(HALF, (char*)(&halfdata[i][0] - data_window.min.x - data_window.min.y * width), sizeof(half),
                width * sizeof(half)));
        }

        // write
        try {
            OutputFile outfile(outputFile.c_str(), header);
            outfile.setFrameBuffer(fb);
            //y_count() rows to write
            outfile.writePixels(height);
        }
        catch (const std::exception& e) {
            std::cerr << "Unable to write image file " << outputFile << " : " << e.what() << std::endl;
            return 1;
        }

        // free memory
        for (int i = 0; i < halfdata.size(); i++) {
            delete halfdata[i];
        }
        return 0;
    }




}


