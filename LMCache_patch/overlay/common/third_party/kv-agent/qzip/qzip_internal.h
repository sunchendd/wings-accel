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

#ifndef _QZIP_INTERNAL_H
#define _QZIP_INTERNAL_H

#include <stdio.h>
#include <stdarg.h>
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
#include <qatzip.h> /* new QATzip interface */
#include <pthread.h>

#define OK      0
#define ERROR   1

#define SUFFIX_GZ      ".gz"
#define SUFFIX_7Z      ".7z"
#define SUFFIX_LZ4     ".lz4"
#define SUFFIX_LZ4S    ".lz4s"

#define QZIP_GET_LOWER_32BITS(v)  ((v) & 0xFFFFFFFF)

#define SRC_BUFF_LEN         (512 * 1024 * 1024)

typedef enum QzipDataFormat_E {
    QZIP_DEFLATE_4B = 0,
    /**< Data is in raw deflate format with 4 byte header */
    QZIP_DEFLATE_GZIP,
    /**< Data is in deflate wrapped by GZip header and footer */
    QZIP_DEFLATE_GZIP_EXT,
    /**< Data is in deflate wrapped by GZip extended header and footer */
    QZIP_DEFLATE_RAW,
    /**< Data is in raw deflate format */
    QZIP_LZ4_FH,
    /**< Data is in LZ4 format with frame headers */
    QZIP_LZ4S_BK,
    /**< Data is in LZ4s format with block headers */
} QzipDataFormat_T;

typedef struct QzipParams_S {
    QzHuffmanHdr_T huffman_hdr;
    QzDirection_T direction;
    QzipDataFormat_T data_fmt;
    unsigned int comp_lvl;
    unsigned char comp_algorithm;
    unsigned char force;
    unsigned char keep;
    unsigned int hw_buff_sz;
    unsigned int polling_mode;
    unsigned int recursive_mode;
    unsigned int req_cnt_thrshold;
    char *output_filename;
} QzipParams_T;

typedef struct RunTimeList_S {
    struct timeval time_s;
    struct timeval time_e;
    struct RunTimeList_S *next;
} RunTimeList_T;

#define  NANO_SEC     1000000000UL
#define  TICKS_PER_SEC  10000000UL

/*
 * internal api functions
 */
void freeTimeList(RunTimeList_T *time_list);
void displayStats(RunTimeList_T *time_list,
                  off_t insize, off_t outsize, int is_compress);
int qatzipSetup(QzSession_T *sess, QzipParams_T *params);
int qatzipClose(QzSession_T *sess);

#ifdef QATZIP_DEBUG
static inline void QZ_DEBUG(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
}
#else
#define QZ_DEBUG(...)
#endif

static inline void QZ_PRINT(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
}

static inline void QZ_ERROR(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

#endif
