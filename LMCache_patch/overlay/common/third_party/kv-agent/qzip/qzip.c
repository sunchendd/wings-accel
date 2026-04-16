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
#include "qzip_internal.h"
#include "qzip.h"

QzipParams_T g_params_th = {
    .huffman_hdr = QZ_HUFF_HDR_DEFAULT,
    .direction = QZ_DIRECTION_DEFAULT,
    .data_fmt = QZIP_DEFLATE_GZIP_EXT,
    .comp_lvl = QZ_COMP_LEVEL_DEFAULT,
    .comp_algorithm = QZ_COMP_ALGOL_DEFAULT,
    .hw_buff_sz = QZ_HW_BUFF_SZ,
    .polling_mode = QZ_BUSY_POLLING,
    .req_cnt_thrshold = 32
};

static int doProcessBuffer_new(QzSession_T *sess,
		unsigned char *src, unsigned int *src_len,
		unsigned char *dst, unsigned int *dst_len,
		int is_compress)
{
	int ret = QZ_FAIL;
	unsigned int done = 0;
	unsigned int buf_processed = 0;
	unsigned int buf_remaining = *src_len;
	unsigned int dst_processed = 0;
	unsigned int valid_dst_buf_len = *dst_len;
	unsigned int single_dst_len = valid_dst_buf_len;
	int i = 0;

	while (!done) {
		/* Do actual work */
		if (is_compress) {
			//printf("here1,sess:%p, internal:%p, src:%p, src_len:%u, dst:%p, dst_len:%d.\n",
			//		(void *)sess, sess->internal, src, *src_len, dst, single_dst_len);
			ret = qzCompress(sess, src, src_len, dst, &single_dst_len, 1);
			if (QZ_BUF_ERROR == ret && 0 == *src_len) {
				done = 1;
			}
		} else {
			ret = qzDecompress(sess, src, src_len, dst, &single_dst_len);

			if (QZ_DATA_ERROR == ret ||
				(QZ_BUF_ERROR == ret && 0 == *src_len)) {
				done = 1;
			}
		}

		if (ret != QZ_OK &&
			ret != QZ_BUF_ERROR &&
			ret != QZ_DATA_ERROR) {
			const char *op = (is_compress) ? "Compression" : "Decompression";
			QZ_ERROR("doProcessBuffer:%s failed with error: %d\n", op, ret);
			break;
		}

		//printf("%d, process_dst_len:%d\n", i, single_dst_len);
		i++;
		dst += single_dst_len;
		dst_processed += single_dst_len;
		valid_dst_buf_len -= single_dst_len;
		assert(dst_processed <= *dst_len);

		buf_processed += *src_len;
		buf_remaining -= *src_len;
		if (0 == buf_remaining) {
			done = 1;
		}
		src += *src_len;
		QZ_DEBUG("src_len is %u ,buf_remaining is %u\n", *src_len,
			buf_remaining);
		*src_len = buf_remaining;
		single_dst_len = valid_dst_buf_len;
	}

	*src_len = buf_processed;
	*dst_len = dst_processed;
	return ret;
}

static int doProcessBlocks(char *inputs[], char *outputs[],
            int in_data_sizes[], int out_data_sizes[],
            int num, int is_compress)
{
	QzSession_T sess = {0};
	unsigned int dst_total = 0;
	unsigned int dst_bytes = 0;
	unsigned int src_bytes = 0;
	int ret = 0;
	int i;

	g_params_th.req_cnt_thrshold = 1;
	ret = qatzipSetup(&sess, &g_params_th);
	if (ret) {
		QZ_PRINT("init session failed for op:%d.\n", is_compress);
		return -1;
	}

	for (i = 0; i < num; ++i)
	{
        if (!outputs[i]) {
            QZ_PRINT("the outputs entry for process %s for target %p, %d invalid.\n",
                    is_compress == 1 ? "compression": "decompression",
                    inputs[i], i);
            ret = -1;
            goto exit;
        }
        dst_bytes = out_data_sizes[i];
        src_bytes += in_data_sizes[i];
        ret = doProcessBuffer_new(&sess, (unsigned char *)inputs[i], (unsigned int *)&in_data_sizes[i],
                            (unsigned char *)outputs[i], &dst_bytes, is_compress);
        if (ret) {
            QZ_PRINT("process %s for target %p, %d failed.\n", is_compress == 1 ? "compression":
                    "decompression", inputs[i], i);
            ret = -1;
            goto exit;
        }
        out_data_sizes[i] = dst_bytes;
        dst_total += dst_bytes;
	}
	ret = 0;

exit:
	qatzipClose(&sess);

	return ret;
}


int kv_agent_block_compress(char *inputs[], char *outputs[],
			int in_data_sizes[], int out_data_sizes[], int num)
{
	return doProcessBlocks(inputs, outputs, in_data_sizes,
                 out_data_sizes, num, 1);
}

int kv_agent_block_decompress(char *inputs[], char *outputs[],
			int in_data_sizes[], int out_data_sizes[], int num)
{
	return doProcessBlocks(inputs, outputs, in_data_sizes,
                out_data_sizes, num, 0);
}

int qzipSetupSessionDeflate(QzSession_T *sess, QzipParams_T *params)
{
    int status;
    QzSessionParamsDeflate_T deflate_params;

    status = qzGetDefaultsDeflate(&deflate_params);
    if (status < 0) {
        QZ_ERROR("Session setup failed with error: %d\n", status);
        return ERROR;
    }

    switch (params->data_fmt) {
    case QZIP_DEFLATE_4B:
        deflate_params.data_fmt = QZ_DEFLATE_4B;
        break;
    case QZIP_DEFLATE_GZIP:
        deflate_params.data_fmt = QZ_DEFLATE_GZIP;
        break;
    case QZIP_DEFLATE_GZIP_EXT:
        deflate_params.data_fmt = QZ_DEFLATE_GZIP_EXT;
        break;
    case QZIP_DEFLATE_RAW:
        deflate_params.data_fmt = QZ_DEFLATE_RAW;
        break;
    default:
        QZ_ERROR("Unsupported data format\n");
        return ERROR;
    }

    deflate_params.huffman_hdr = params->huffman_hdr;
    deflate_params.common_params.direction = params->direction;
    deflate_params.common_params.comp_lvl = params->comp_lvl;
    deflate_params.common_params.comp_algorithm = params->comp_algorithm;
    deflate_params.common_params.hw_buff_sz = params->hw_buff_sz;
    deflate_params.common_params.polling_mode = params->polling_mode;
    deflate_params.common_params.req_cnt_thrshold = params->req_cnt_thrshold;

    status = qzSetupSessionDeflate(sess, &deflate_params);
    if (status < 0) {
        QZ_ERROR("Session setup failed with error: %d\n", status);
        return ERROR;
    }

    return OK;
}

int qzipSetupSessionLZ4(QzSession_T *sess, QzipParams_T *params)
{
    int status;
    QzSessionParamsLZ4_T lz4_params;

    status = qzGetDefaultsLZ4(&lz4_params);
    if (status < 0) {
        QZ_ERROR("Session setup failed with error: %d\n", status);
        return ERROR;
    }

    lz4_params.common_params.direction = params->direction;
    lz4_params.common_params.comp_lvl = params->comp_lvl;
    lz4_params.common_params.comp_algorithm = params->comp_algorithm;
    lz4_params.common_params.hw_buff_sz = params->hw_buff_sz;
    lz4_params.common_params.polling_mode = params->polling_mode;
    lz4_params.common_params.req_cnt_thrshold = params->req_cnt_thrshold;

    status = qzSetupSessionLZ4(sess, &lz4_params);
    if (status < 0) {
        QZ_ERROR("Session setup failed with error: %d\n", status);
        return ERROR;
    }

    return OK;
}

int qzipSetupSessionLZ4S(QzSession_T *sess, QzipParams_T *params)
{
    int status;
    QzSessionParamsLZ4S_T lz4s_params;

    status = qzGetDefaultsLZ4S(&lz4s_params);
    if (status < 0) {
        QZ_ERROR("Session setup failed with error: %d\n", status);
        return ERROR;
    }

    lz4s_params.common_params.direction = params->direction;
    lz4s_params.common_params.comp_lvl = params->comp_lvl;
    lz4s_params.common_params.comp_algorithm = params->comp_algorithm;
    lz4s_params.common_params.hw_buff_sz = params->hw_buff_sz;
    lz4s_params.common_params.polling_mode = params->polling_mode;
    lz4s_params.common_params.req_cnt_thrshold = params->req_cnt_thrshold;

    status = qzSetupSessionLZ4S(sess, &lz4s_params);
    if (status < 0) {
        QZ_ERROR("Session setup failed with error: %d\n", status);
        return ERROR;
    }

    return OK;
}

int qatzipSetup(QzSession_T *sess, QzipParams_T *params)
{
    int status;

    QZ_DEBUG("mw>>> sess=%p\n", sess);
    status = qzInit(sess, 1);
    if (status != QZ_OK &&
        status != QZ_DUPLICATE) {
        QZ_ERROR("QAT init failed with error: %d\n", status);
        return ERROR;
    }
    QZ_DEBUG("QAT init OK with error: %d\n", status);

    switch (params->data_fmt) {
    case QZIP_DEFLATE_4B:
    case QZIP_DEFLATE_GZIP:
    case QZIP_DEFLATE_GZIP_EXT:
    case QZIP_DEFLATE_RAW:
        status = qzipSetupSessionDeflate(sess, params);
        if (status != OK) {
            QZ_ERROR("qzipSetupSessionDeflate fail with error: %d\n", status);
        }
        break;
    case QZIP_LZ4_FH:
        status = qzipSetupSessionLZ4(sess, params);
        if (status != OK) {
            QZ_ERROR("qzipSetupSessionLZ4 fail with error: %d\n", status);
        }
        break;
    case QZIP_LZ4S_BK:
        status = qzipSetupSessionLZ4S(sess, params);
        if (status != OK) {
            QZ_ERROR("qzipSetupSessionLZ4S fail with error: %d\n", status);
        }
        break;
    default:
        QZ_ERROR("Unsupported data format\n");
        return ERROR;
    }

    QZ_DEBUG("Session setup OK with error: %d\n", status);
    return 0;
}

int qatzipClose(QzSession_T *sess)
{
    qzTeardownSession(sess);
    qzClose(sess);

    return 0;
}

/* function to replace the implementation in uplevel lib to
 * indicate the usage of compression/decompression */
int kv_agent_compress_enabled(void)
{
    return 1;
}
