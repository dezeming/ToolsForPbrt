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
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <iostream>
#include <gdt/gdt.h>

/*! \namespace PbrtOptixDenoiser - Optix Siggraph Course */
namespace PbrtOptixDenoiser {

  extern "C" char embedded_ptx_code[];

  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleDenoiser::SampleDenoiser(bool ifIllumOnly, FilterOptions& fltOps) :
      illuminOnly(ifIllumOnly),
      filterOptions(fltOps)
  {
    initOptix();

    std::cout << "#osc: creating optix context ..." << std::endl;
    createContext();

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
  }

  SampleDenoiser::~SampleDenoiser() {
      fbColor.free();
      fbNormal.free();
      fbAlbedo.free();
      fbPosition.free();

      /*! output of the denoiser pass, in float4 */
      denoisedBuffer.free();
      denoisedBuffer2.free();

      /* the actual undenoised color buffer used for display, in rgba8 */
      undenoisedColorBuffer.free();
      /* the actual final color buffer used for display, in rgba8 */
      finalColorBuffer.free();
      finalColorBuffer2.free();
      /* the actual neighborClampped buffer used for display, in rgba8 */
      neighborClamppedColorBuffer.free();

      denoiserScratch.free();
      denoiserState.free();
      denoiserIntensity.free();
  }

  /*! helper function that initializes optix and checks for errors */
  void SampleDenoiser::initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;
    
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void SampleDenoiser::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }


  /*! render one frame */
  void SampleDenoiser::render()
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (framesize.x == 0) return;

    denoiserIntensity.resize(sizeof(float));

    OptixDenoiserParams denoiserParams;
#if OPTIX_VERSION > 70500
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
#if OPTIX_VERSION >= 70300
    if (denoiserIntensity.sizeInBytes != sizeof(float))
        denoiserIntensity.alloc(sizeof(float));
#endif
    denoiserParams.hdrIntensity = denoiserIntensity.d_pointer();
    if(accumulate)
        denoiserParams.blendFactor  = 1.f/10.0f;
    else
        denoiserParams.blendFactor = 0.0f;

    int inputlayer = filterOptions.validNum(channelType);

    // -------------------------------------------------------
    OptixImage2D inputLayer[10];
    inputLayer[0].data = fbColor.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[0].width = framesize.x;
    /// Height of the image (in pixels)
    inputLayer[0].height = framesize.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[0].rowStrideInBytes = framesize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[0].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[2].data = fbNormal.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[2].width = framesize.x;
    /// Height of the image (in pixels)
    inputLayer[2].height = framesize.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[2].rowStrideInBytes = framesize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[2].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[1].data = fbAlbedo.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[1].width = framesize.x;
    /// Height of the image (in pixels)
    inputLayer[1].height = framesize.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[1].rowStrideInBytes = framesize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[1].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = denoisedBuffer.d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = framesize.x;
    /// Height of the image (in pixels)
    outputLayer.height = framesize.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = framesize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    if (denoiserOn) {
      OPTIX_CHECK(optixDenoiserComputeIntensity
                  (denoiser,
                   /*stream*/0,
                   &inputLayer[0],
                   (CUdeviceptr)denoiserIntensity.d_pointer(),
                   (CUdeviceptr)denoiserScratch.d_pointer(),
                   denoiserScratch.size()));
      
#if OPTIX_VERSION >= 70300
    OptixDenoiserGuideLayer denoiserGuideLayer = {};
    if(filterOptions.useAlbedo)
        denoiserGuideLayer.albedo = inputLayer[1];
    if (filterOptions.useNormal)
        denoiserGuideLayer.normal = inputLayer[2];

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer[0];
    denoiserLayer.output = outputLayer;

      OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                      /*stream*/0,
                                      &denoiserParams,
                                      denoiserState.d_pointer(),
                                      denoiserState.size(),
                                      &denoiserGuideLayer,
                                      &denoiserLayer,1,
                                      /*inputOffsetX*/0,
                                      /*inputOffsetY*/0,
                                      denoiserScratch.d_pointer(),
                                      denoiserScratch.size()));
#else
      OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                      /*stream*/0,
                                      &denoiserParams,
                                      denoiserState.d_pointer(),
                                      denoiserState.size(),
                                      &inputLayer[0],2,
                                      /*inputOffsetX*/0,
                                      /*inputOffsetY*/0,
                                      &outputLayer,
                                      denoiserScratch.d_pointer(),
                                      denoiserScratch.size()));
#endif
    } else {
      cudaMemcpy((void*)outputLayer.data,(void*)inputLayer[0].data,
                 outputLayer.width*outputLayer.height*sizeof(float4),
                 cudaMemcpyDeviceToDevice);
    }
    computeFinalPixelColors();
    
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
  }

  void SampleDenoiser::render2()
  {
      // sanity check: make sure we launch only after first resize is
      // already done:
      if (framesize.x == 0) return;

      denoiserIntensity.resize(sizeof(float));

      OptixDenoiserParams denoiserParams;
#if OPTIX_VERSION > 70500
      denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
#if OPTIX_VERSION >= 70300
      if (denoiserIntensity.sizeInBytes != sizeof(float))
          denoiserIntensity.alloc(sizeof(float));
#endif
      denoiserParams.hdrIntensity = denoiserIntensity.d_pointer();
      if (accumulate)
          denoiserParams.blendFactor = 1.f / 10.0f;
      else
          denoiserParams.blendFactor = 0.0f;

      // -------------------------------------------------------
      OptixImage2D inputLayer[3];
      inputLayer[0].data = denoisedBufferToneMapped.d_pointer();
      /// Width of the image (in pixels)
      inputLayer[0].width = framesize.x;
      /// Height of the image (in pixels)
      inputLayer[0].height = framesize.y;
      /// Stride between subsequent rows of the image (in bytes).
      inputLayer[0].rowStrideInBytes = framesize.x * sizeof(float4);
      /// Stride between subsequent pixels of the image (in bytes).
      /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
      inputLayer[0].pixelStrideInBytes = sizeof(float4);
      /// Pixel format.
      inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

      // ..................................................................
      inputLayer[2].data = fbNormal.d_pointer();
      /// Width of the image (in pixels)
      inputLayer[2].width = framesize.x;
      /// Height of the image (in pixels)
      inputLayer[2].height = framesize.y;
      /// Stride between subsequent rows of the image (in bytes).
      inputLayer[2].rowStrideInBytes = framesize.x * sizeof(float4);
      /// Stride between subsequent pixels of the image (in bytes).
      /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
      inputLayer[2].pixelStrideInBytes = sizeof(float4);
      /// Pixel format.
      inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

      // ..................................................................
      inputLayer[1].data = fbAlbedo.d_pointer();
      /// Width of the image (in pixels)
      inputLayer[1].width = framesize.x;
      /// Height of the image (in pixels)
      inputLayer[1].height = framesize.y;
      /// Stride between subsequent rows of the image (in bytes).
      inputLayer[1].rowStrideInBytes = framesize.x * sizeof(float4);
      /// Stride between subsequent pixels of the image (in bytes).
      /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
      inputLayer[1].pixelStrideInBytes = sizeof(float4);
      /// Pixel format.
      inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;


      // -------------------------------------------------------
      OptixImage2D outputLayer;
      outputLayer.data = denoisedBuffer2.d_pointer();
      /// Width of the image (in pixels)
      outputLayer.width = framesize.x;
      /// Height of the image (in pixels)
      outputLayer.height = framesize.y;
      /// Stride between subsequent rows of the image (in bytes).
      outputLayer.rowStrideInBytes = framesize.x * sizeof(float4);
      /// Stride between subsequent pixels of the image (in bytes).
      /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
      outputLayer.pixelStrideInBytes = sizeof(float4);
      /// Pixel format.
      outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

      // -------------------------------------------------------
      if (denoiserOn) {
          OPTIX_CHECK(optixDenoiserComputeIntensity
          (denoiser,
              /*stream*/0,
              &inputLayer[0],
              (CUdeviceptr)denoiserIntensity.d_pointer(),
              (CUdeviceptr)denoiserScratch.d_pointer(),
              denoiserScratch.size()));

#if OPTIX_VERSION >= 70300
          OptixDenoiserGuideLayer denoiserGuideLayer = {};
          denoiserGuideLayer.albedo = inputLayer[1];
          denoiserGuideLayer.normal = inputLayer[2];

          OptixDenoiserLayer denoiserLayer = {};
          denoiserLayer.input = inputLayer[0];
          denoiserLayer.output = outputLayer;

          OPTIX_CHECK(optixDenoiserInvoke(denoiser,
              /*stream*/0,
              &denoiserParams,
              denoiserState.d_pointer(),
              denoiserState.size(),
              &denoiserGuideLayer,
              &denoiserLayer, 1,
              /*inputOffsetX*/0,
              /*inputOffsetY*/0,
              denoiserScratch.d_pointer(),
              denoiserScratch.size()));
#else
          OPTIX_CHECK(optixDenoiserInvoke(denoiser,
              /*stream*/0,
              &denoiserParams,
              denoiserState.d_pointer(),
              denoiserState.size(),
              &inputLayer[0], 2,
              /*inputOffsetX*/0,
              /*inputOffsetY*/0,
              &outputLayer,
              denoiserScratch.d_pointer(),
              denoiserScratch.size()));
#endif
      }
      else {
          cudaMemcpy((void*)outputLayer.data, (void*)inputLayer[0].data,
              outputLayer.width * outputLayer.height * sizeof(float4),
              cudaMemcpyDeviceToDevice);
      }
      computeRender2PixelColors();

      // sync - make sure the frame is rendered before we download and
      // display (obviously, for a high-performance application you
      // want to use streams and double-buffering, but for this simple
      // example, this will have to do)
      CUDA_SYNC_CHECK();
  }


  /*! resize frame buffer to given resolution */
  void SampleDenoiser::resize(const gdt::vec2i &newSize)
  {
    if (denoiser) {
      OPTIX_CHECK(optixDenoiserDestroy(denoiser));
    };


    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};
#if OPTIX_VERSION >= 70300
    OPTIX_CHECK(optixDenoiserCreate(optixContext,OPTIX_DENOISER_MODEL_KIND_LDR,&denoiserOptions,&denoiser));
#else
    // denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;

    // as suggested by LobnerO, in https://github.com/ingowald/optix7course/issues/41 :
    denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
#if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

    OPTIX_CHECK(optixDenoiserCreate(optixContext,&denoiserOptions,&denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(denoiser,OPTIX_DENOISER_MODEL_KIND_LDR,NULL,0));
#endif

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser,newSize.x,newSize.y,
                                                    &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
    denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
#else
    denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                                    denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
#endif
    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);
    
    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    denoisedBuffer.resize(newSize.x*newSize.y*sizeof(float4));
    denoisedBufferToneMapped.resize(newSize.x * newSize.y * sizeof(float4));
    denoisedBuffer2.resize(newSize.x * newSize.y * sizeof(float4));
    fbColor.resize(newSize.x*newSize.y*sizeof(float4));
    fbNormal.resize(newSize.x*newSize.y*sizeof(float4));
    fbAlbedo.resize(newSize.x*newSize.y*sizeof(float4));
    fbPosition.resize(newSize.x * newSize.y * sizeof(float4));

    finalColorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));
    finalColorBuffer2.resize(newSize.x * newSize.y * sizeof(uint32_t));
    neighborClamppedColorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
    undenoisedColorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

    // update the launch parameters that we'll pass to the optix
    // launch:
    framesize = newSize;

    // ------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(denoiser,0,
                                   newSize.x,newSize.y,
                                   denoiserState.d_pointer(),
                                   denoiserState.size(),
                                   denoiserScratch.d_pointer(),
                                   denoiserScratch.size()));
  }
  
  void SampleDenoiser::loadCpuBuffer(ExrDataStructure& dataStruct) {

      channelType = dataStruct.channeltype;
      int ColorChannelType = channelType.getTypeOrder("Color");
      if (ColorChannelType != -1)
        fbColor.upload(dataStruct.dataPointers[ColorChannelType], framesize.x * framesize.y);
      int AlbedoChannelType = channelType.getTypeOrder("Albedo");
      if (AlbedoChannelType != -1)
          fbAlbedo.upload(dataStruct.dataPointers[AlbedoChannelType], framesize.x * framesize.y);
      int NormalChannelType = channelType.getTypeOrder("Normal");
      if (NormalChannelType != -1)
          fbNormal.upload(dataStruct.dataPointers[NormalChannelType], framesize.x * framesize.y);
  }

  /*! download the rendered color buffer */
  void SampleDenoiser::downloadPixels(uint32_t h_pixels[])
  {
      finalColorBuffer.download(h_pixels,
        framesize.x* framesize.y);
  }
  void SampleDenoiser::downloadRender2Pixels(uint32_t h_pixels[])
  {
      finalColorBuffer2.download(h_pixels,
          framesize.x * framesize.y);
  }

  void SampleDenoiser::downloadUndenoisedPixels(uint32_t h_pixels[]) {
      undenoisedColorBuffer.download(h_pixels,
          framesize.x * framesize.y);
  }
  void SampleDenoiser::downloadNeighbourClamppedPixels(uint32_t h_pixels[]) {
      neighborClamppedColorBuffer.download(h_pixels,
          framesize.x * framesize.y);
  }



  /*! download the float4 buffer */
  void SampleDenoiser::downloadDenoisedFloatBuffer(ExrDataStructure& dataStruct) {

      int typeIndex = dataStruct.channeltype.getTypeOrder("Color");
      if (dataStruct.dataPointers.size() <= typeIndex) {
          std::cout << "The channel is not present in the exr file! " << std::endl;
          return;
      }
      denoisedBuffer.download(dataStruct.dataPointers[typeIndex], framesize.x * framesize.y);
  }
  void SampleDenoiser::downloadDenoised2FloatBuffer(ExrDataStructure& dataStruct) {

      int typeIndex = dataStruct.channeltype.getTypeOrder("Color");
      if (dataStruct.dataPointers.size() <= typeIndex) {
          std::cout << "The channel is not present in the exr file! " << std::endl;
          return;
      }
      denoisedBuffer2.download(dataStruct.dataPointers[typeIndex], framesize.x * framesize.y);
  }




  void SampleConvertor::convert() {

      computeConvertedColors();

      // sync - make sure the frame is rendered before we download and
      // display (obviously, for a high-performance application you
      // want to use streams and double-buffering, but for this simple
      // example, this will have to do)
      CUDA_SYNC_CHECK();

  }
  void SampleConvertor::downloadPixels(uint32_t h_pixels[])
  {
      finalColorBuffer.download(h_pixels,
          framesize.x * framesize.y);
  }
  void SampleConvertor::loadCpuBuffer(float4* color) {

      fbColor.upload(color, framesize.x * framesize.y);
  }
  void SampleConvertor::resize(const gdt::vec2i& newSize)
  {
      // ------------------------------------------------------------------
      // resize our cuda frame buffer

      fbColor.resize(newSize.x * newSize.y * sizeof(float4));
      finalColorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));


      // update the launch parameters that we'll pass to the optix
      // launch:
      framesize = newSize;
  }








  
} // ::osc
