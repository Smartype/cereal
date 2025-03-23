/*
A common mechanism to share memory buffers across different devices
  • ION converted to produce DMA-BUF handles
  • CMEM converted to produce handles converted to CMEM handles
  • Userspace applications moving to use DMA-BUFs
    – OpenCL: cl_arm_import_memory_dma_buf
    – EGL: EGL_EXT_image_dma_buf_import
    – V4L2: V4L2_MEMORY_FD
    – DRM: DRM_IOCTL_PRIME
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <CL/cl_ext.h>
#include <gralloc/gr_priv_handle.h>
#include <linux/ion.h>
#include <linux/msm_ion.h>
#include "visionbuf.h"

#define MAP_SHARED  0x01    /* Share changes */

// keep trying if x gets interrupted by a signal
#define HANDLE_EINTR(x)                                       \
  ({                                                          \
    decltype(x) ret;                                          \
    int try_cnt = 0;                                          \
    do {                                                      \
      ret = (x);                                              \
    } while (ret == -1 && errno == EINTR && try_cnt++ < 100); \
    ret;                                                      \
  })

// just hard-code these for convenience
// size_t device_page_size = 0;
// clGetDeviceInfo(device_id, CL_DEVICE_PAGE_SIZE_QCOM,
//                 sizeof(device_page_size), &device_page_size,
//                 NULL);

// size_t padding_cl = 0;
// clGetDeviceInfo(device_id, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM,
//                 sizeof(padding_cl), &padding_cl,
//                 NULL);

#define DEVICE_PAGE_SIZE_CL 4096
#define PADDING_CL 0

struct IonFileHandle {
  IonFileHandle() {
    fd = open("/dev/ion", O_RDWR | O_NONBLOCK);
    assert(fd >= 0);
  }
  ~IonFileHandle() {
    close(fd);
  }
  int fd = -1;
};

int ion_fd() {
  static IonFileHandle fh;
  return fh.fd;
}

void VisionBuf::allocate(size_t length) {
  int err;

  struct ion_allocation_data ion_alloc = {0};
  ion_alloc.len = length + PADDING_CL + sizeof(uint64_t);
  ion_alloc.heap_id_mask = 1 << ION_SYSTEM_HEAP_ID;
  //ion_alloc.heap_id_mask = 1 << ION_HEAP_TYPE_DMA;
  ion_alloc.flags = ION_FLAG_CACHED;

  err = HANDLE_EINTR(ioctl(ion_fd(), ION_IOC_ALLOC, &ion_alloc));
  assert(err == 0);

  /*
   In kernel 4.12, the ION_IOC_ALLOC ioctl directly outputs dma-buf fds. The intermediate ION handle state has been removed, along with all ioctls that consume or produce ION handles. Because dma-buf fds aren't tied to specific ION clients, the ION_IOC_SHARE ioctl is no longer needed, and all ION client infrastructure has been removed.
  */

  void *mmap_addr = mmap(NULL, ion_alloc.len,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, ion_alloc.fd, 0);
  assert(mmap_addr != MAP_FAILED);

  memset(mmap_addr, 0, ion_alloc.len);

  this->len = length;
  this->mmap_len = ion_alloc.len;
  this->addr = mmap_addr;
  this->fd = ion_alloc.fd;
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len + PADDING_CL);
}

void VisionBuf::import(){
  assert(this->fd >= 0);
  this->addr = mmap(NULL, this->mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
  assert(this->addr != MAP_FAILED);
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len + PADDING_CL);
}

// alternatively: OpenCL: cl_arm_import_memory_dma_buf
void VisionBuf::init_cl(cl_device_id device_id, cl_context ctx) {
  // https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_ext_host_ptr.txt
  int err;
  assert(((uintptr_t)this->addr % DEVICE_PAGE_SIZE_CL) == 0);
  cl_mem_ion_host_ptr ion_cl = {0};
  ion_cl.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
  ion_cl.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
  ion_cl.ion_filedesc = this->fd;
  ion_cl.ion_hostptr = this->addr;
  this->buf_cl = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, this->len, &ion_cl, &err);
  assert(err == 0);
}

int VisionBuf::sync(int dir) {
  /*
   Kernel 4.12 replaced ION_IOC_SYNC with the DMA_BUF_IOCTL_SYNC ioctl defined in linux/dma-buf.h. Call DMA_BUF_IOCTL_SYNC at the start and end of every CPU access, with flags specifying whether these accesses are reads and/or writes. Although DMA_BUF_IOCTL_SYNC is more verbose than ION_IOC_SYNC, it gives userspace more control over the underlying cache maintenance operations.
DMA_BUF_IOCTL_SYNC is part of the kernel's stable ABI and is usable with all dma-buf fds, whether or not they were allocated by ION.
  */
  struct dma_buf_sync sync;
  assert(dir == VISIONBUF_SYNC_FROM_DEVICE || dir == VISIONBUF_SYNC_TO_DEVICE);
  sync.flags = (dir == VISIONBUF_SYNC_FROM_DEVICE) ? DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW : DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
  return HANDLE_EINTR(ioctl(this->fd, DMA_BUF_IOCTL_SYNC, &sync));
}

int VisionBuf::free() {
  int err = 0;
  if (this->buf_cl){
    err = clReleaseMemObject(this->buf_cl);
    if (err != 0)
      return err;
  }

  err = munmap(this->addr, this->mmap_len);
  if (err != 0)
    return err;

  return close(this->fd);
}
