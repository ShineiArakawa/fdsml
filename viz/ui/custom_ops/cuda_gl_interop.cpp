/**
 * @file cuda_gl_interop.cpp
 * @author Shinei Arakawa (sarakawalab@gmail.com)
 * @brief CUDA-OpenGL inter-operability functions
 *
 * @copyright Copyright (c) 2026 Shinei Arakawa
 *  Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
 *  To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
 *  This code is a derivative work based on code from the harskish/pyviewer repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/harskish/pyviewer.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#ifdef WIN32
// clang-format off
#include <windows.h>
#include <gl/GL.h>
// clang-format on
#else
#include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>

#include <iostream>

// --------------------------------------------------------------------------------------------------------------------------------
// Utilities

#define cudaErrors(...)                                                                  \
  do {                                                                                   \
    cudaError_t error = __VA_ARGS__;                                                     \
    if (error) {                                                                         \
      std::cout << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << "\n"; \
      std::cout << "while running " #__VA_ARGS__ "\n";                                   \
    }                                                                                    \
  } while (false)

// --------------------------------------------------------------------------------------------------------------------------------
// Texture interop functions

uint64_t register_texture_resource(const GLuint tex) {
  cudaGraphicsResource_t resource = nullptr;
  cudaErrors(cudaGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
  return (uint64_t)resource;
}

void unregister_texture_resource(uint64_t ptr) {
  auto resource = (cudaGraphicsResource_t)ptr;
  cudaErrors(cudaGraphicsUnregisterResource(resource));
}

void upload_texture(uint64_t data_ptr, int width, int height, int pitch, uint64_t ptr) {
  auto resource = (cudaGraphicsResource_t)ptr;
  cudaErrors(cudaGraphicsMapResources(1, &resource));
  cudaArray_t array;
  cudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
  cudaErrors(cudaMemcpyToArray(array, 0, 0, (const void*)data_ptr, width * height * pitch, cudaMemcpyDeviceToDevice));
  cudaErrors(cudaGraphicsUnmapResources(1, &resource));
}

// --------------------------------------------------------------------------------------------------------------------------------
// VBO interop functions (only for points)

uint64_t register_vbo_resource(const GLuint vbo) {
  cudaGraphicsResource_t resource = nullptr;
  cudaErrors(cudaGraphicsGLRegisterBuffer(&resource, vbo, cudaGraphicsRegisterFlagsNone));
  return (uint64_t)resource;
}

void unregister_vbo_resource(uint64_t ptr) {
  auto resource = (cudaGraphicsResource_t)ptr;
  cudaErrors(cudaGraphicsUnregisterResource(resource));
}

void* map_vbo(uint64_t ptr) {
  auto resource = (cudaGraphicsResource_t)ptr;
  cudaErrors(cudaGraphicsMapResources(1, &resource));
  void* data_ptr = nullptr;
  size_t size = 0;
  cudaErrors(cudaGraphicsResourceGetMappedPointer(&data_ptr, &size, resource));
  return data_ptr;
}

void unmap_vbo(uint64_t ptr) {
  auto resource = (cudaGraphicsResource_t)ptr;
  cudaErrors(cudaGraphicsUnmapResources(1, &resource));
}

void upload_vbo(uint64_t data_ptr, size_t num_bytes, uint64_t ptr) {
  auto resource = (cudaGraphicsResource_t)ptr;
  cudaErrors(cudaGraphicsMapResources(1, &resource));
  void* mapped_ptr = nullptr;
  size_t mapped_size = 0;
  cudaErrors(cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &mapped_size, resource));
  if (num_bytes > mapped_size) {
    std::cout << "upload_vbo: requested bytes exceed mapped buffer size\n";
    num_bytes = mapped_size;
  }
  cudaErrors(cudaMemcpy(mapped_ptr, (const void*)data_ptr, num_bytes, cudaMemcpyDeviceToDevice));
  cudaErrors(cudaGraphicsUnmapResources(1, &resource));
}

// --------------------------------------------------------------------------------------------------------------------------------
// Python bindings

PYBIND11_MODULE(cuda_gl_interop, m) {
  m.def("register_texture", &register_texture_resource, "register texture resource");
  m.def("unregister_texture", &unregister_texture_resource, "unregister texture resource");
  m.def("upload_texture", &upload_texture, "upload texture data");
  m.def("register_vbo", &register_vbo_resource, "register VBO resource");
  m.def("unregister_vbo", &unregister_vbo_resource, "unregister VBO resource");
  m.def("map_vbo", &map_vbo, "map VBO resource");
  m.def("unmap_vbo", &unmap_vbo, "unmap VBO resource");
  m.def("upload_vbo", &upload_vbo, "upload VBO data");
}
