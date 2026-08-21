#define _GNU_SOURCE

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef EGL_CUDA_DEVICE_NV
#  define EGL_CUDA_DEVICE_NV 0x323A
#endif

typedef EGLDisplay (*EglGetDisplayFn)(EGLNativeDisplayType);
typedef void *(*DlsymFn)(void *, const char *);
typedef __eglMustCastToProperFunctionPointerType (*EglGetProcAddressFn)(const char *);

static EglGetDisplayFn real_egl_get_display = NULL;
static EglGetProcAddressFn real_egl_get_proc_address = NULL;

static void fail_closed(const char *message);

static DlsymFn real_dlsym(void)
{
  static DlsymFn function = NULL;
  if (function == NULL) {
    function = (DlsymFn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (function == NULL) {
      fprintf(stderr, "PhysSweep EGL selector: could not resolve the real dlsym\n");
      fflush(stderr);
      _exit(86);
    }
  }
  return function;
}

static void *real_egl_symbol(const char *name)
{
  static void *handle = NULL;
  if (handle == NULL) {
    handle = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == NULL) {
      fail_closed("could not load libEGL.so.1");
    }
  }
  return real_dlsym()(handle, name);
}

static void fail_closed(const char *message)
{
  fprintf(stderr, "PhysSweep EGL selector: %s\n", message);
  fflush(stderr);
  _exit(86);
}

EGLDisplay eglGetDisplay(EGLNativeDisplayType native_display)
{
  if (real_egl_get_display == NULL) {
    real_egl_get_display = (EglGetDisplayFn)real_egl_symbol("eglGetDisplay");
    if (real_egl_get_display == NULL) {
      fail_closed("could not resolve the real eglGetDisplay");
    }
  }

  const char *requested = getenv("PHYSWEEP_EGL_CUDA_DEVICE");
  if (requested == NULL || requested[0] == '\0' || native_display != EGL_DEFAULT_DISPLAY) {
    return real_egl_get_display(native_display);
  }

  errno = 0;
  char *end = NULL;
  const long requested_index = strtol(requested, &end, 10);
  if (errno != 0 || end == requested || *end != '\0' || requested_index < 0) {
    fail_closed("PHYSWEEP_EGL_CUDA_DEVICE must be a non-negative integer");
  }

  PFNEGLQUERYDEVICESEXTPROC query_devices =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
  PFNEGLQUERYDEVICEATTRIBEXTPROC query_attribute =
      (PFNEGLQUERYDEVICEATTRIBEXTPROC)eglGetProcAddress("eglQueryDeviceAttribEXT");
  PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
  if (query_devices == NULL || query_attribute == NULL || get_platform_display == NULL) {
    fail_closed("required EGL device extensions are unavailable");
  }

  EGLDeviceEXT devices[32];
  EGLint device_count = 0;
  if (!query_devices(32, devices, &device_count)) {
    fail_closed("eglQueryDevicesEXT failed");
  }
  for (EGLint index = 0; index < device_count; ++index) {
    EGLAttrib cuda_index = -1;
    if (query_attribute(devices[index], EGL_CUDA_DEVICE_NV, &cuda_index) &&
        cuda_index == requested_index) {
      EGLDisplay display =
          get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[index], NULL);
      if (display == EGL_NO_DISPLAY) {
        fail_closed("selected EGL device did not provide a display");
      }
      setenv("CUDA_VISIBLE_DEVICES", requested, 1);
      setenv("HIP_VISIBLE_DEVICES", requested, 1);
      fprintf(stderr,
              "PhysSweep EGL selector: CUDA device %ld (EGL device %d of %d)\n",
              requested_index,
              index,
              device_count);
      fflush(stderr);
      return display;
    }
  }

  fail_closed("requested CUDA device was not exposed by EGL");
  return EGL_NO_DISPLAY;
}

__eglMustCastToProperFunctionPointerType eglGetProcAddress(const char *name)
{
  if (real_egl_get_proc_address == NULL) {
    real_egl_get_proc_address =
        (EglGetProcAddressFn)real_egl_symbol("eglGetProcAddress");
    if (real_egl_get_proc_address == NULL) {
      fail_closed("could not resolve the real eglGetProcAddress");
    }
  }
  if (getenv("PHYSWEEP_EGL_CUDA_DEVICE") != NULL &&
      strcmp(name, "eglGetDisplay") == 0) {
    return (__eglMustCastToProperFunctionPointerType)eglGetDisplay;
  }
  return real_egl_get_proc_address(name);
}

void *dlsym(void *handle, const char *name)
{
  void *resolved = real_dlsym()(handle, name);
  if (getenv("PHYSWEEP_EGL_CUDA_DEVICE") == NULL) {
    return resolved;
  }
  if (strcmp(name, "eglGetDisplay") == 0) {
    real_egl_get_display = (EglGetDisplayFn)resolved;
    return (void *)eglGetDisplay;
  }
  if (strcmp(name, "eglGetProcAddress") == 0) {
    real_egl_get_proc_address = (EglGetProcAddressFn)resolved;
    return (void *)eglGetProcAddress;
  }
  return resolved;
}
