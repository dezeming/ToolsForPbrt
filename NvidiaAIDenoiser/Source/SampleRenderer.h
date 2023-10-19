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

// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include <gdt/math/vec.h>
#include <gdt/gdt.h>
#include <gdt/math/vec/functors.h>

#include "DataInfo.h"

/*! \namespace PbrtOptixDenoiser - Optix Siggraph Course */
namespace PbrtOptixDenoiser {
  
class SampleConvertor {

public:
    SampleConvertor() {};

    /*! resize frame buffer to given resolution */
    void resize(const gdt::vec2i& newSize);
    /*! load Images From float4 CPU Buffer  */
    void loadCpuBuffer(float4* color);

    /*! render one frame */
    void convert();
    void computeConvertedColors();

    /*! download the rendered color buffer */
    void downloadPixels(uint32_t h_pixels[]);

    gdt::vec2i framesize;
    CUDABuffer fbColor;
    CUDABuffer finalColorBuffer;
};

struct FilterOptions {

    FilterOptions() {
        reset();
    }
    void reset() {
        useColor = false;
        useAlbedo = false;
        useNormal = false;
        usePosition = false;
        useShadingNormal = false;
        useVariance = false;
        useRelativeVariance = false;
        useTangent = false;
        useTextureUV = false;
        useRoughness = false;
    }

    int validNum(ChannelType& ct) {
        int count = 0;
        if (useColor)
            if (ct.hasColor) count++;
            else useColor = false;

        if (useAlbedo)
            if (ct.hasAlbedo) count++;
            else useAlbedo = false;

        if (useNormal)
            if (ct.hasNormal) count++;
            else useNormal = false;

        if (usePosition)
            if (ct.hasPosition) count++;
            else usePosition = false;

        if (useShadingNormal)
            if (ct.hasShadingNormal) count++;
            else useShadingNormal = false;

        if (useVariance)
            if (ct.hasVariance) count++;
            else useVariance = false;

        if (useRelativeVariance)
            if (ct.hasRelativeVariance) count++;
            else useRelativeVariance = false;

        if (useTangent)
            if (ct.hasTangent) count++;
            else useTangent = false;

        if (useTextureUV)
            if (ct.hasTextureUV) count++;
            else useTextureUV = false;

        if (useRoughness)
            if (ct.hasRoughness) count++;
            else useRoughness = false;

        return count;
    }

    bool useColor;
    bool useAlbedo;
    bool useNormal;
    bool usePosition;
    bool useShadingNormal;
    bool useVariance;
    bool useRelativeVariance;
    bool useTangent;
    bool useTextureUV;
    bool useRoughness;
};

class SampleDenoiser {
    // ------------------------------------------------------------------
    // publicly accessible interface
    // ------------------------------------------------------------------
public:
    /*! constructor - performs all setup, including initializing
        optix, creates module, pipeline, programs, SBT, etc. */
        SampleDenoiser(bool illumOnly, FilterOptions& fltOps);

        ~SampleDenoiser();

    /** render one frame: iteration 1
    * input: fbColor/fbAlbedo/fbNormal
    * -float4
    * output: denoisedBuffer-float4 is [0,1] after tone mapped
    * -uint_32
    * output: undenoisedColorBuffer-uint_32
    * output: neighborClamppedColorBuffer-uint_32
    * output: finalColorBuffer-uint_32
    */
    void render();

    /** render one frame: iteration 2
    * -float4
    * output: denoisedBuffer2 is [0,1] after tone mapped
    * -uint_32
    * output: finalColorBuffer2£º
    */
    void render2();

    /*! resize frame buffer to given resolution */
    void resize(const gdt::vec2i &newSize);
    /*! load Images From float4 CPU Buffer  */
    void loadCpuBuffer(ExrDataStructure& dataStruct);

    /*! download the rendered color buffer */
    void downloadPixels(uint32_t h_pixels[]);
    void downloadRender2Pixels(uint32_t h_pixels[]);
    void downloadUndenoisedPixels(uint32_t h_pixels[]);
    void downloadNeighbourClamppedPixels(uint32_t h_pixels[]);
    
    /*! download the float buffer */
    void downloadDenoisedFloatBuffer(ExrDataStructure& dataStruct);
    void downloadDenoised2FloatBuffer(ExrDataStructure& dataStruct);


    bool denoiserOn = true;
    bool accumulate = true;

    ChannelType channelType;
    FilterOptions filterOptions;

    gdt::vec2i framesize;

    bool isIlluminOnly() {
        return illuminOnly;
    }

protected:

    bool illuminOnly;
    // ------------------------------------------------------------------
    // internal helper functions
    // ------------------------------------------------------------------

    /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
    void computeFinalPixelColors();
    // Generate the denoising result of the second filtering
    void computeRender2PixelColors();
    
    /*! helper function that initializes optix and checks for errors */
    void initOptix();
  
    /*! creates and configures a optix device context (in this simple
        example, only for the primary GPU device) */
    void createContext();

protected:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

protected:

    OptixDenoiser denoiser = nullptr;
    CUDABuffer    denoiserScratch;
    CUDABuffer    denoiserState;
    CUDABuffer    denoiserIntensity;

    /*********************/
    /* float4 buffers */
    /*********************/

    CUDABuffer fbColor;
    CUDABuffer fbNormal;
    CUDABuffer fbAlbedo;
    CUDABuffer fbPosition;
    
    // The result after denoising once, without neighboring Clamped (the result of neighboring Clamped will be directly output to the buffer of rgba8)
    CUDABuffer denoisedBuffer;
    CUDABuffer denoisedBufferToneMapped;
    CUDABuffer denoisedBuffer2;
    

    /********************/
    /* rgba8 buffers */
    /********************/

    // the actual undenoised color buffer used for display, in rgba8 
    CUDABuffer undenoisedColorBuffer;
    // the actual final color buffer used for display, in rgba8 
    CUDABuffer finalColorBuffer;
    CUDABuffer finalColorBuffer2;
    // the actual neighborClampped buffer used for display, in rgba8 
    CUDABuffer neighborClamppedColorBuffer;

};




} // ::osc
