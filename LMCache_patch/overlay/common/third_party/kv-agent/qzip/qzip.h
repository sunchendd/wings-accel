/***************************************************************************
 *
 *   BSD LICENSE
 *
 *   Copyright(c) 2024 Intel Corporation. All rights reserved.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ***************************************************************************/

#ifndef _QZIP_KV_H
#define _QZIP_KV_H

#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* __cplusplus */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <getopt.h>
#include <limits.h>
#include <assert.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <utime.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <zlib.h>
#include <libgen.h>

/* for kv cache processing */
typedef struct KvCacheInput_S {
	unsigned char *addr;
	unsigned int len;
}KvCacheInfo_T;

int doProcessKvCache(KvCacheInfo_T *cache_array, int cache_num,
				const char *target_file_name, int is_compress);

/* the memory for target data which carried by outputs arrary
 * should be pre-allocated by uplevel */
int kv_agent_block_compress(char *inputs[], char *outputs[],
				int in_data_sizes[], int out_data_sizes[], int num);

/* the memory for target data which carried by outputs arrary
 * should be pre-allocated by uplevel */
int kv_agent_block_decompress(char *inputs[], char *outputs[],
				int in_data_sizes[], int out_data_sizes[], int num);

/* function to replace the implementation in uplevel lib to
 * indicate the usage of compression/decompression */
int kv_agent_compress_enabled(void);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif
