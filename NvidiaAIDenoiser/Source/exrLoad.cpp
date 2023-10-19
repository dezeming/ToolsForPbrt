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

#include "exrLoad.h"
#include "exrHeader.h"
#include "DataInfo.h"

namespace exrLoad {

    // using namespace Imf;
    using Imf::InputFile;
    using Imf::ChannelList;

    //using namespace std;
    using std::vector;
    using std::string;
    using std::cout;
    using std::endl;

    //using namespace IMATH_NAMESPACE;
    using IMATH_NAMESPACE::Box2i;


bool hasChannel_GBuffers(std::string name) {
    if (name == "Albedo.B")return true;
    if (name == "Albedo.G")return true;
    if (name == "Albedo.R")return true;
    if (name == "B")return true;
    if (name == "G")return true;
    if (name == "R")return true;
    if (name == "N.X")return true;
    if (name == "N.Y")return true;
    if (name == "N.Z")return true;
    return false;
}

bool hasChannel_RGB(std::string name) {
    if (name == "B")return true;
    if (name == "G")return true;
    if (name == "R")return true;
    return false;
}

int loadGBuffers(std::string filename, gdt::vec2i& resolution, ExrDataStructure & datastruct, bool illumOnly)
{
    //string channelnames[] = {
    // "Albedo.B","Albedo.G","Albedo.R",
    // "B","G","R",
    // "N.X","N.Y","N.Z",
    // "Ns.X","Ns.Y","Ns.Z",
    // "P.X","P.Y","P.Z",
    // "RelativeVariance.B","RelativeVariance.G","RelativeVariance.R",
    // "Variance.B","Variance.G","Variance.R",
    // "dzdx","dzdy",
    // "u","v"};
    //

    bool needPrintInfo = false;

    try {
        // It cannot be read with rgba inputfile, it is rgb's
        InputFile file(filename.c_str());
        Imath::Box2i dw = file.header().dataWindow();
        int                width = dw.max.x - dw.min.x + 1;
        int                height = dw.max.y - dw.min.y + 1;
        std::cout << width << " " << height << std::endl;
        
        resolution = gdt::vec2i(width, height);
        datastruct.ImageSize = resolution;

        // Channel name
        vector<string> channelnames;
        const ChannelList& channels = file.header().channels();
        int numChannels = 0;
        for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
        {
            channelnames.push_back(i.name());
            numChannels++;
        }
        //std::cout << numChannels << std::endl;

        // Allocate buffers for channels
        vector<half*> allpixels;
        for (int i = 0; i < numChannels; i++) {
            half* onechannel = new half[width * height];
            allpixels.push_back(onechannel);
        }

        //Read channel to buffer
        Imf::FrameBuffer frameBuffer;
        // Each channel is a slice in the buffer
        for (int i = 0; i < numChannels; i++) {
            frameBuffer.insert(channelnames[i], Imf::Slice(Imf::HALF, (char*)(&allpixels[i][0] -
                dw.min.x - dw.min.y * width),
                sizeof(half),
                width * sizeof(half),
                1, 1, // x/y sampling
                0.0));
        }
        file.setFrameBuffer(frameBuffer);
        file.readPixels(dw.min.y, dw.max.y);
            
        // Output a single channel to a file
        if (false) {
            for (int c = 0; c < numChannels; c++) {
                FILE* fp;
                string filename = "../channeldata/" + channelnames[c] + ".txt";
                cout << filename << endl;
                if ((fp = fopen(filename.c_str(), "wb")) == NULL) {
                    printf("cant open the file");
                    exit(0);
                }
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        float p = f16_to_f32(allpixels[c][i * width + j]);
                        p = clamp(p, 0, 1);
                        //cout << p << ",";
                        fprintf(fp, "%f,", p);

                    }fprintf(fp, "\n");
                }
                fclose(fp);
            }
        }

        datastruct.channeltype.reset();

        // Print channel name
        int dataChannels_Acc = 0;
        if (true) {
            for (int c = 0; c < numChannels; c++) {
                string channelName = channelnames[c];

                if (hasChannel_GBuffers(channelName))dataChannels_Acc++;

                datastruct.channeltype.setChannel(channelName);
            }
            if (needPrintInfo){
                std::cout << std::endl;
            }
        }

        datastruct.channeltype.PrintInfo();
        datastruct.channeltype.PrintTypeNames();

        // Not meeting the current need for Color/Abedo/Normal
        if (dataChannels_Acc != 3*3) 
        {
            std::cout << "Better to include information on Color/Abedo/Normal with 3 channels" << std::endl;
            //return -1;
        }

        if (!datastruct.channeltype.hasColor || datastruct.channeltype.ColorChannels != 3) {
            std::cout << "Must to have Color with 3 channels" << std::endl;
            return -1;
        }

        // Generate memory of pointers, all initialized to 0.0f
        for (int index = 0; index < datastruct.channeltype.getTypesCount(); index++) {
            float4* data = new float4[width * height];

            datastruct.dataPointers.push_back(data);

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    datastruct.dataPointers[index][i * width + j].x = 0.0f;
                    datastruct.dataPointers[index][i * width + j].y = 0.0f;
                    datastruct.dataPointers[index][i * width + j].z = 0.0f;
                    datastruct.dataPointers[index][i * width + j].w = 0.0f;
                }
            }
        }

        // Fill memory
        if (true) {
            for (int c = 0; c < numChannels; c++) {
                string channelName = channelnames[c];

                int TypeIndex, channel;
                datastruct.channeltype.getTypeOrder(channelName, TypeIndex, channel);
                if (needPrintInfo) {
                   std::cout << "channelName " << channelName << " : typeIndex = [" << TypeIndex << "] channel = [" << channel << "]" << std::endl;
                }

                if (TypeIndex == -1 || channel == -1) continue;

                if (channel == 0)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float p = f16_to_f32(allpixels[c][i * width + j]);  
                            datastruct.dataPointers[TypeIndex][i * width + j].x = p;
                        }
                    }
                else if (channel == 1)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float p = f16_to_f32(allpixels[c][i * width + j]);
                            datastruct.dataPointers[TypeIndex][i * width + j].y = p;
                        }
                    }
                else if (channel == 2)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float p = f16_to_f32(allpixels[c][i * width + j]);
                            datastruct.dataPointers[TypeIndex][i * width + j].z = p;
                        }
                    }
                else if (channel == 3)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float p = f16_to_f32(allpixels[c][i * width + j]);
                            datastruct.dataPointers[TypeIndex][i * width + j].w = p;
                        }
                    }
            }
            if (needPrintInfo) {
                std::cout << std::endl;
            }

        }


        // Consider forcing an Albedo to be written when there is no Albedo
        if (!datastruct.channeltype.hasAlbedo) {
            std::cout << "Better to have Albedo, now I generate one with all pixels'value equel 1.0f." << std::endl;
            datastruct.channeltype.setChannel("Albedo.R");
            datastruct.channeltype.setChannel("Albedo.G");
            datastruct.channeltype.setChannel("Albedo.B");

            float4* data = new float4[width * height];
            datastruct.dataPointers.push_back(data);

            // fill Albedo 
            int TypeIndex, channel;
            datastruct.channeltype.getTypeOrder("Albedo.R", TypeIndex, channel);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    datastruct.dataPointers[TypeIndex][i * width + j].x = 1.0;
                    datastruct.dataPointers[TypeIndex][i * width + j].y = 1.0;
                    datastruct.dataPointers[TypeIndex][i * width + j].z = 1.0;
                }
            }
        }

        // Erase the albedo from radiance
        if (illumOnly) {
            if ((datastruct.channeltype.hasAlbedo && datastruct.channeltype.AlbedoChannels == 3)) {
                int albedoTypeIndex, colorTypeOrder;
                albedoTypeIndex = datastruct.channeltype.getTypeOrder("Albedo");
                colorTypeOrder = datastruct.channeltype.getTypeOrder("Color");

                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                    
                        if (datastruct.dataPointers[albedoTypeIndex][i * width + j].x != 0.0f)
                            datastruct.dataPointers[colorTypeOrder][i * width + j].x /= datastruct.dataPointers[albedoTypeIndex][i * width + j].x;
                        if (datastruct.dataPointers[albedoTypeIndex][i * width + j].y != 0.0f)
                            datastruct.dataPointers[colorTypeOrder][i * width + j].y /= datastruct.dataPointers[albedoTypeIndex][i * width + j].y;
                        if (datastruct.dataPointers[albedoTypeIndex][i * width + j].z != 0.0f)
                            datastruct.dataPointers[colorTypeOrder][i * width + j].z /= datastruct.dataPointers[albedoTypeIndex][i * width + j].z;
                    }
                }
            }
            else {
                std::cout << "Error requiest: do not have Albedo" << std::endl;
            }
        }

        for (int i = 0; i < allpixels.size(); i++) {
            delete allpixels[i];
        }

    }
    catch (const std::exception& e) {
        std::cerr << "error reading image file hello.exr:" << e.what() << std::endl;
        return 1;
    }

    if (datastruct.dataPointers.size() < 1) {
        std::cout << "loadGBuffers Error with dataPointers.size() equals 0" << std::endl;
        return -1;
    }

    return 0;
}

int loadRGB(std::string filename, gdt::vec2i& resolution, std::vector<float4*>& dataPointers)
{
    //string channelnames[] = {
    // "Albedo.B","Albedo.G","Albedo.R",
    // "B","G","R",
    // "N.X","N.Y","N.Z",
    // "Ns.X","Ns.Y","Ns.Z",
    // "P.X","P.Y","P.Z",
    // "RelativeVariance.B","RelativeVariance.G","RelativeVariance.R",
    // "Variance.B","Variance.G","Variance.R",
    // "dzdx","dzdy",
    // "u","v"};
    //

    try {
        InputFile file(filename.c_str());
        Imath::Box2i dw = file.header().dataWindow();
        int                width = dw.max.x - dw.min.x + 1;
        int                height = dw.max.y - dw.min.y + 1;
        std::cout << width << " " << height << std::endl;

        resolution = gdt::vec2i(width, height);

        std::vector<std::string> channelnames;
        const ChannelList& channels = file.header().channels();
        int numChannels = 0;
        for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i)
        {
            channelnames.push_back(i.name());
            numChannels++;
        }
        //std::cout << numChannels << std::endl;

        vector<half*> allpixels;
        for (int i = 0; i < numChannels; i++) {
            half* onechannel = new half[width * height];
            allpixels.push_back(onechannel);
        }

        Imf::FrameBuffer frameBuffer;
        for (int i = 0; i < numChannels; i++) {
            frameBuffer.insert(channelnames[i], Imf::Slice(Imf::HALF, (char*)(&allpixels[i][0] -
                dw.min.x - dw.min.y * width),
                sizeof(half),
                width * sizeof(half),
                1, 1, // x/y sampling
                0.0));
        }
        file.setFrameBuffer(frameBuffer);
        file.readPixels(dw.min.y, dw.max.y);

        int dataChannels_Acc = 0;
        if (true) {
            for (int c = 0; c < numChannels; c++) {
                string channelName = channelnames[c];

                if (hasChannel_RGB(channelName))dataChannels_Acc++;



                std::cout << channelName << " ";
            }
            std::cout << std::endl;
        }

        if (dataChannels_Acc != 3) {
            std::cout << "Need to include information on Color with 3 channels" << std::endl;
            return -1;
        }

        // color albedo normal   
        float4* data = new float4[width * height];
        dataPointers.push_back(data);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dataPointers[0][i * width + j].w = 1.0f;
            }
        }
        
        if (true) {
            for (int c = 0; c < numChannels; c++) {
                string channelName = channelnames[c];

                int pointNum = -1; int xyz = -1;
                if (channelName == "R") { pointNum = 0; xyz = 0; }
                else if (channelName == "G") { pointNum = 0; xyz = 1; }
                else if (channelName == "B") { pointNum = 0; xyz = 2; }

                if (pointNum == -1) continue;

                if (xyz == 0)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float p = f16_to_f32(allpixels[c][i * width + j]);
                            dataPointers[pointNum][i * width + j].x = p;
                        }
                    }
                else if (xyz == 1)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float p = f16_to_f32(allpixels[c][i * width + j]);
                            dataPointers[pointNum][i * width + j].y = p;
                        }
                    }
                else if (xyz == 2)
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            float p = f16_to_f32(allpixels[c][i * width + j]);
                            dataPointers[pointNum][i * width + j].z = p;
                        }
                    }
            }
            std::cout << std::endl;
        }

        

    }
    catch (const std::exception& e) {
        std::cerr << "error reading image file hello.exr:" << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int AlbedoVisualize(ExrDataStructure& datastruct, uint32_t h_pixels[], bool usingNormalization) {

    if (!datastruct.channeltype.hasAlbedo || datastruct.channeltype.AlbedoChannels < 3) {
        std::cout << "No Albedo with three channel data" << std::endl;
        return -1;
    }

    int typeIndex_R, typeIndex_G, typeIndex_B;
    int channel;
    datastruct.channeltype.getTypeOrder("Albedo.R", typeIndex_R, channel);
    if (channel != 0) {
        std::cout << "Error channel about Albedo.R." << std::endl;
        return -1;
    }
    datastruct.channeltype.getTypeOrder("Albedo.G", typeIndex_G, channel);
    if (channel != 1) {
        std::cout << "Error channel about Albedo.G." << std::endl;
        return -1;
    }
    datastruct.channeltype.getTypeOrder("Albedo.B", typeIndex_B, channel);
    if (channel != 2) {
        std::cout << "Error channel about Albedo.B." << std::endl;
        return -1;
    }
    if (typeIndex_R != typeIndex_G || typeIndex_R != typeIndex_B) {
        std::cout << "The array index of Albedo's three channels is inconsistent." << std::endl;
        return -1;
    }

    float4* datp = datastruct.dataPointers[typeIndex_R];
    // Method: Normalize the values to the range of [0,1]
    if (usingNormalization) {
        float max_x = -99999.0f, max_y = -99999.0f, max_z = -99999.0f;
        float min_x = 99999.0f, min_y = 99999.0f, min_z = 99999.0f;
        // Calculate the maximum value for each dimension
        for (int i = 0; i < datastruct.ImageSize.x; i++) {
            for (int j = 0; j < datastruct.ImageSize.y; j++) {
                int offset = i + j * datastruct.ImageSize.x;
                max_x = max_x > datp[offset].x ? max_x : datp[offset].x;
                min_x = min_x < datp[offset].x ? min_x : datp[offset].x;
                max_y = max_y > datp[offset].y ? max_y : datp[offset].y;
                min_y = min_y < datp[offset].y ? min_y : datp[offset].y;
                max_z = max_z > datp[offset].z ? max_z : datp[offset].z;
                min_z = min_z < datp[offset].z ? min_z : datp[offset].z;
            }
        }

        std::cout << "In AlbedoVisualize: x[" << min_x << "," << max_y << "]"
            << " y[" << min_y << "," << max_y << "]"
            << " z[" << min_z << "," << max_z << "]" << std::endl;

        // Numerical normalization
        float x_intervel = max_x - min_x;
        x_intervel = x_intervel > 0.0f ? x_intervel : 1.f;
        float x_intervel_inv = 1.0f / x_intervel;
        float y_intervel = max_y - min_y;
        y_intervel = y_intervel > 0.0f ? y_intervel : 1.f;
        float y_intervel_inv = 1.0f / y_intervel;
        float z_intervel = max_z - min_z;
        z_intervel = z_intervel > 0.0f ? z_intervel : 1.f;
        float z_intervel_inv = 1.0f / z_intervel;

        for (int i = 0; i < datastruct.ImageSize.x; i++) {
            for (int j = 0; j < datastruct.ImageSize.y; j++) {

                int offset = i + j * datastruct.ImageSize.x;
                uint32_t rgba = 0;
                rgba |= (uint32_t)((datp[offset].x - min_x) * x_intervel_inv * 255.9f) << 0;
                rgba |= (uint32_t)((datp[offset].y - min_y) * y_intervel_inv * 255.9f) << 8;
                rgba |= (uint32_t)((datp[offset].z - min_z) * z_intervel_inv * 255.9f) << 16;
                rgba |= (uint32_t)255 << 24;

                h_pixels[offset] = rgba;
            }
        }
    }
    // Method: Clamp the values to the range of [0,1]
    else {
        for (int i = 0; i < datastruct.ImageSize.x; i++) {
            for (int j = 0; j < datastruct.ImageSize.y; j++) {

                int offset = i + j * datastruct.ImageSize.x;
                uint32_t rgba = 0;
                float x_val, y_val, z_val; 
                if (datp[offset].x > 1.0f) x_val = 1.0f;
                else if (datp[offset].x < 0.0f) x_val = 0.0f;
                else x_val = datp[offset].x;
                if (datp[offset].y > 1.0f) y_val = 1.0f;
                else if (datp[offset].y < 0.0f) y_val = 0.0f;
                else y_val = datp[offset].y;
                if (datp[offset].z > 1.0f) z_val = 1.0f;
                else if (datp[offset].z < 0.0f) z_val = 0.0f;
                else z_val = datp[offset].z;
                        
                rgba |= (uint32_t)(x_val * 255.9f) << 0;
                rgba |= (uint32_t)(y_val * 255.9f) << 8;
                rgba |= (uint32_t)(z_val * 255.9f) << 16;
                rgba |= (uint32_t)255 << 24;

                h_pixels[offset] = rgba;
            }
        }
    }

    return 0;

}

int NormalVisualize(ExrDataStructure& datastruct, uint32_t h_pixels[]) {

    if (!datastruct.channeltype.hasNormal || datastruct.channeltype.NormalChannels < 3) {
        std::cout << "No Normal with three channel data" << std::endl;
        return -1;
    }

    int typeIndex_R, typeIndex_G, typeIndex_B;
    int channel;
    datastruct.channeltype.getTypeOrder("N.X", typeIndex_R, channel);
    if (channel != 0) {
        std::cout << "Error channel about N.X." << std::endl;
        return -1;
    }
    datastruct.channeltype.getTypeOrder("N.Y", typeIndex_G, channel);
    if (channel != 1) {
        std::cout << "Error channel about N.Y." << std::endl;
        return -1;
    }
    datastruct.channeltype.getTypeOrder("N.Z", typeIndex_B, channel);
    if (channel != 2) {
        std::cout << "Error channel about N.Z." << std::endl;
        return -1;
    }
    if (typeIndex_R != typeIndex_G || typeIndex_R != typeIndex_B) {
        std::cout << "The array index of Normal's three channels is inconsistent." << std::endl;
        return -1;
    }

    float4* datp = datastruct.dataPointers[typeIndex_R];
    float max_x = -99999.0f, max_y = -99999.0f, max_z = -99999.0f;
    float min_x = 99999.0f, min_y = 99999.0f, min_z = 99999.0f;
    // Calculate the maximum value for each dimension
    for (int i = 0; i < datastruct.ImageSize.x; i++) {
        for (int j = 0; j < datastruct.ImageSize.y; j++) {
            int offset = i + j * datastruct.ImageSize.x;
            max_x = max_x > datp[offset].x ? max_x : datp[offset].x;
            min_x = min_x < datp[offset].x ? min_x : datp[offset].x;
            max_y = max_y > datp[offset].y ? max_y : datp[offset].y;
            min_y = min_y < datp[offset].y ? min_y : datp[offset].y;
            max_z = max_z > datp[offset].z ? max_z : datp[offset].z;
            min_z = min_z < datp[offset].z ? min_z : datp[offset].z;
        }
    }

    std::cout << "In NormalVisualize: x[" << min_x << "," << max_y << "]"
        << " y[" << min_y << "," << max_y << "]"
        << " z[" << min_z << "," << max_z << "]" << std::endl;

    // Numerical normalization
    float x_intervel = max_x - min_x;
    x_intervel = x_intervel > 0.0f ? x_intervel : 1.f;
    float x_intervel_inv = 1.0f / x_intervel;
    float y_intervel = max_y - min_y;
    y_intervel = y_intervel > 0.0f ? y_intervel : 1.f;
    float y_intervel_inv = 1.0f / y_intervel;
    float z_intervel = max_z - min_z;
    z_intervel = z_intervel > 0.0f ? z_intervel : 1.f;
    float z_intervel_inv = 1.0f / z_intervel;
    for (int i = 0; i < datastruct.ImageSize.x; i++) {
        for (int j = 0; j < datastruct.ImageSize.y; j++) {

            int offset = i + j * datastruct.ImageSize.x;
            uint32_t rgba = 0;
            rgba |= (uint32_t)((datp[offset].x - min_x) * x_intervel_inv * 255.9f) << 0;
            rgba |= (uint32_t)((datp[offset].y - min_y) * y_intervel_inv * 255.9f) << 8;
            rgba |= (uint32_t)((datp[offset].z - min_z) * z_intervel_inv * 255.9f) << 16;
            rgba |= (uint32_t)255 << 24;

            h_pixels[offset] = rgba;
        }
    }
}

int PositionVisualize(ExrDataStructure& datastruct, uint32_t h_pixels[]) {

    if (!datastruct.channeltype.hasPosition || datastruct.channeltype.PositionChannels < 3) {
        std::cout << "No Position with three channel data" << std::endl;
        return -1;
    }

    int typeIndex_R, typeIndex_G, typeIndex_B;
    int channel;
    datastruct.channeltype.getTypeOrder("P.X", typeIndex_R, channel);
    if (channel != 0) {
        std::cout << "Error channel about P.X." << std::endl;
        return -1;
    }
    datastruct.channeltype.getTypeOrder("P.Y", typeIndex_G, channel);
    if (channel != 1) {
        std::cout << "Error channel about P.Y." << std::endl;
        return -1;
    }
    datastruct.channeltype.getTypeOrder("P.Z", typeIndex_B, channel);
    if (channel != 2) {
        std::cout << "Error channel about P.Z." << std::endl;
        return -1;
    }
    if (typeIndex_R != typeIndex_G || typeIndex_R != typeIndex_B) {
        std::cout << "The array index of Position's three channels is inconsistent." << std::endl;
        return -1;
    }

    float4* datp = datastruct.dataPointers[typeIndex_R];
    float max_x = -99999.0f, max_y = -99999.0f, max_z = -99999.0f;
    float min_x = 99999.0f, min_y = 99999.0f, min_z = 99999.0f;
    // Calculate the maximum value for each dimension
    for (int i = 0; i < datastruct.ImageSize.x; i++) {
        for (int j = 0; j < datastruct.ImageSize.y; j++) {
            int offset = i + j * datastruct.ImageSize.x;
            max_x = max_x > datp[offset].x ? max_x : datp[offset].x;
            min_x = min_x < datp[offset].x ? min_x : datp[offset].x;
            max_y = max_y > datp[offset].y ? max_y : datp[offset].y;
            min_y = min_y < datp[offset].y ? min_y : datp[offset].y;
            max_z = max_z > datp[offset].z ? max_z : datp[offset].z;
            min_z = min_z < datp[offset].z ? min_z : datp[offset].z;
        }
    }

    std::cout << "In PositionVisualize: x[" << min_x << "," << max_y << "]"
        << " y[" << min_y << "," << max_y << "]"
        << " z[" << min_z << "," << max_z << "]" << std::endl;

    // Numerical normalization
    float x_intervel = max_x - min_x;
    x_intervel = x_intervel > 0.0f ? x_intervel : 1.f;
    float x_intervel_inv = 1.0f / x_intervel;
    float y_intervel = max_y - min_y;
    y_intervel = y_intervel > 0.0f ? y_intervel : 1.f;
    float y_intervel_inv = 1.0f / y_intervel;
    float z_intervel = max_z - min_z;
    z_intervel = z_intervel > 0.0f ? z_intervel : 1.f;
    float z_intervel_inv = 1.0f / z_intervel;
    for (int i = 0; i < datastruct.ImageSize.x; i++) {
        for (int j = 0; j < datastruct.ImageSize.y; j++) {

            int offset = i + j * datastruct.ImageSize.x;
            uint32_t rgba = 0;
            rgba |= (uint32_t)((datp[offset].x - min_x) * x_intervel_inv * 255.9f) << 0;
            rgba |= (uint32_t)((datp[offset].y - min_y) * y_intervel_inv * 255.9f) << 8;
            rgba |= (uint32_t)((datp[offset].z - min_z) * z_intervel_inv * 255.9f) << 16;
            rgba |= (uint32_t)255 << 24;

            h_pixels[offset] = rgba;
        }
    }
}


}


