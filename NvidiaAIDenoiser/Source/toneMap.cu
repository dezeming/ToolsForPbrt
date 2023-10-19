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

#include "SampleRenderer.h"

using namespace PbrtOptixDenoiser;

namespace PbrtOptixDenoiser {

  inline __device__ float4 sqrt(float4 f)
  {
    return make_float4(sqrtf(f.x),
                       sqrtf(f.y),
                       sqrtf(f.z),
                       sqrtf(f.w));
  }
  inline __device__ float  clampf(float f) { return min(1.f,max(0.f,f)); }
  inline __device__ float4 clamp(float4 f)
  {
    return make_float4(clampf(f.x),
                       clampf(f.y),
                       clampf(f.z),
                       clampf(f.w));
  }
  


  /*****************************************************/
  /************    First round denoising    ************/
  /*****************************************************/

  /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
  __global__ void computeFinalPixelColorsKernel(uint32_t *finalColorBuffer,
                                                uint32_t* undenoisedColorBuffer,
                                                float4* denoisedBuffer,
                                                float4* denoisedBufferToneMapped,
                                                float4* AbbedoBuffer,
                                                float4* undenoisedBuffer,
                                                gdt::vec2i     size, bool illumOnly)
  {
    int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
    if (pixelX >= size.x) return;
    if (pixelY >= size.y) return;

    int pixelID = pixelX + size.x*pixelY;

    /************************************************/
    /************************************************/
    float4 f4 = denoisedBuffer[pixelID];

    if (illumOnly)
        f4 = make_float4(
            (AbbedoBuffer[pixelID].x == 0.f) ? f4.x : f4.x * AbbedoBuffer[pixelID].x,
            (AbbedoBuffer[pixelID].y == 0.f) ? f4.y : f4.y * AbbedoBuffer[pixelID].y,
            (AbbedoBuffer[pixelID].z == 0.f) ? f4.z : f4.z * AbbedoBuffer[pixelID].z,
            1.0f);
    else
        f4 = make_float4(f4.x, f4.y, f4.z, 1.0f);

    // tonemap
    f4 = clamp(sqrt(f4));

    denoisedBufferToneMapped[pixelID] = f4;

    uint32_t rgba = 0;
    rgba |= (uint32_t)(f4.x * 255.9f) <<  0;
    rgba |= (uint32_t)(f4.y * 255.9f) <<  8;
    rgba |= (uint32_t)(f4.z * 255.9f) << 16;
    rgba |= (uint32_t)255             << 24;
    finalColorBuffer[pixelID] = rgba;


    /************************************************/
    /************************************************/
    f4 = undenoisedBuffer[pixelID];
    if (illumOnly) {
        f4 = make_float4(
            (AbbedoBuffer[pixelID].x == 0.f) ? f4.x : f4.x * AbbedoBuffer[pixelID].x,
            (AbbedoBuffer[pixelID].y == 0.f) ? f4.y : f4.y * AbbedoBuffer[pixelID].y,
            (AbbedoBuffer[pixelID].z == 0.f) ? f4.z : f4.z * AbbedoBuffer[pixelID].z,
            1.0f);
    }
    else
        f4 = make_float4(f4.x, f4.y, f4.z, 1.0f);
    // tonemap
    f4 = clamp(sqrt(f4));

    rgba = 0;
    rgba |= (uint32_t)(f4.x * 255.9f) << 0;
    rgba |= (uint32_t)(f4.y * 255.9f) << 8;
    rgba |= (uint32_t)(f4.z * 255.9f) << 16;
    rgba |= (uint32_t)255 << 24;
    undenoisedColorBuffer[pixelID] = rgba;
  }

  __device__ void getMinMax(float4& minVal, float4& maxVal, float4& f4) {

      if (minVal.x > f4.x) minVal.x = f4.x;
      if (minVal.y > f4.y) minVal.y = f4.y;
      if (minVal.z > f4.z) minVal.z = f4.z;
      if (minVal.w > f4.w) minVal.w = f4.w;

      if (maxVal.x < f4.x) maxVal.x = f4.x;
      if (maxVal.y < f4.y) maxVal.y = f4.y;
      if (maxVal.z < f4.z) maxVal.z = f4.z;
      if (maxVal.w < f4.w) maxVal.w = f4.w;
  }

  __device__ void MinMaxClamp(float4& minVal, float4& maxVal, float4& f4, float ratio = 0.2f) {

      maxVal = make_float4(
          maxVal.x * (1.0f + ratio),
          maxVal.y * (1.0f + ratio),
          maxVal.z * (1.0f + ratio), 1.0f);
      minVal = make_float4(
          minVal.x * (1.0f - ratio),
          minVal.y * (1.0f - ratio),
          minVal.z * (1.0f - ratio), 1.0f);

      if (f4.x > maxVal.x)f4.x = maxVal.x;
      if (f4.y > maxVal.y)f4.y = maxVal.y;
      if (f4.z > maxVal.z)f4.z = maxVal.z;

      if (f4.x < minVal.x)f4.x = minVal.x;
      if (f4.y < minVal.y)f4.y = minVal.y;
      if (f4.z < minVal.z)f4.z = minVal.z;

  }

  __global__ void computeNeighbourClamppedColorsKernel(
      uint32_t* neighbourClamppedColorBuffer,
      float4* denoisedBufferToneMapped,
      float4* AbbedoBuffer,
      gdt::vec2i size, bool illumOnly)
  {
      int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
      int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
      if (pixelX >= size.x) return;
      if (pixelY >= size.y) return;

      int pixelID = pixelX + size.x * pixelY;

      /************************************************/
      /************************************************/
      float4 f4 = denoisedBufferToneMapped[pixelID];
      float4 minVal = make_float4(1000.0f, 1000.0f, 1000.0f, 1.0f),
          maxVal = make_float4(-1000.0f, -1000.0f, -1000.0f, 1.0f);
      int radius = 1;
      for (int pix_x = -radius; pix_x <= radius; pix_x++) {
          for (int pix_y = -radius; pix_y <= radius; pix_y++) {

              if (pix_x == 0 && pix_y == 0) continue;

              int pixelX_t = pixelX + pix_x;
              int pixelY_t = pixelY + pix_y;
              if (pixelX_t >= size.x) continue;
              if (pixelY_t >= size.y) continue;

              int pixelID_t = pixelX_t + size.x * pixelY_t;
              float4 f4_t = denoisedBufferToneMapped[pixelID_t];
              getMinMax(minVal, maxVal, f4_t);
          }
      }
      
      MinMaxClamp(minVal, maxVal, f4, 0.01f);

      f4 = make_float4(f4.x, f4.y, f4.z, 1.0f);

      uint32_t rgba = 0;
      rgba |= (uint32_t)(f4.x * 255.9f) << 0;
      rgba |= (uint32_t)(f4.y * 255.9f) << 8;
      rgba |= (uint32_t)(f4.z * 255.9f) << 16;
      rgba |= (uint32_t)255 << 24;
      neighbourClamppedColorBuffer[pixelID] = rgba;
  }

  void SampleDenoiser::computeFinalPixelColors()
  {
    gdt::vec2i fbSize = framesize;
    gdt::vec2i blockSize = 32;
    gdt::vec2i numBlocks = divRoundUp(fbSize,blockSize);

    computeFinalPixelColorsKernel
      <<<dim3(numBlocks.x,numBlocks.y),dim3(blockSize.x,blockSize.y)>>>
      ((uint32_t*)finalColorBuffer.d_pointer(),
          (uint32_t*)undenoisedColorBuffer.d_pointer(),
          (float4*)denoisedBuffer.d_pointer(),
          (float4*)denoisedBufferToneMapped.d_pointer(),
          (float4*)fbAlbedo.d_pointer(),
          (float4*)fbColor.d_pointer(),
       fbSize, illuminOnly);

    computeNeighbourClamppedColorsKernel
        << <dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y) >> >
            ((uint32_t*)neighborClamppedColorBuffer.d_pointer(),
            (float4*)denoisedBufferToneMapped.d_pointer(),
                (float4*)fbAlbedo.d_pointer(),
            fbSize, illuminOnly);
  }





  /*****************************************************/
  /************    second round denoising    ***********/
  /*****************************************************/

  __global__ void computeRender2ColorsKernel(
      uint32_t* Render2ColorBuffer,
      float4* denoisedBuffer,
      gdt::vec2i size)
  {
      int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
      int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
      if (pixelX >= size.x) return;
      if (pixelY >= size.y) return;

      int pixelID = pixelX + size.x * pixelY;

      /************************************************/
      /************************************************/
      float4 f4 = denoisedBuffer[pixelID];
      f4 = clamp(f4);

      uint32_t rgba = 0;
      rgba |= (uint32_t)(f4.x * 255.9f) << 0;
      rgba |= (uint32_t)(f4.y * 255.9f) << 8;
      rgba |= (uint32_t)(f4.z * 255.9f) << 16;
      rgba |= (uint32_t)255 << 24;
      Render2ColorBuffer[pixelID] = rgba;
  }

  void SampleDenoiser::computeRender2PixelColors() {

      gdt::vec2i fbSize = framesize;
      gdt::vec2i blockSize = 32;
      gdt::vec2i numBlocks = divRoundUp(fbSize, blockSize);

      computeRender2ColorsKernel
          << <dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y) >> >
          ((uint32_t*)finalColorBuffer2.d_pointer(),
              (float4*)denoisedBuffer2.d_pointer(),
              fbSize);

  }





  /*****************************************************/
  /****************     color convert     **************/
  /*****************************************************/

  __global__ void computeConvertedColorsKernel(
      uint32_t* finalColorBuffer,
      float4* inputBuffer,
      gdt::vec2i size)
  {
      int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
      int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
      if (pixelX >= size.x) return;
      if (pixelY >= size.y) return;

      int pixelID = pixelX + size.x * pixelY;

      /************************************************/
      /************************************************/
      float4 f4 = inputBuffer[pixelID];
      f4 = clamp(sqrt(f4));

      uint32_t rgba = 0;
      rgba |= (uint32_t)(f4.x * 255.9f) << 0;
      rgba |= (uint32_t)(f4.y * 255.9f) << 8;
      rgba |= (uint32_t)(f4.z * 255.9f) << 16;
      rgba |= (uint32_t)255 << 24;
      finalColorBuffer[pixelID] = rgba;
  }

  void SampleConvertor::computeConvertedColors() {

      gdt::vec2i fbSize = framesize;
      gdt::vec2i blockSize = 32;
      gdt::vec2i numBlocks = divRoundUp(fbSize, blockSize);

      computeConvertedColorsKernel
          << <dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y) >> >
          ((uint32_t*)finalColorBuffer.d_pointer(),
              (float4*)fbColor.d_pointer(),
              fbSize);
  }


  
} // ::osc
