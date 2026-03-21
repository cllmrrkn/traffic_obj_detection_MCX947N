/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MODIFIED: Added persistent class counters for 2x2 grid display
 */

#include <stdio.h>
#include <string.h>
#include "fsl_debug_console.h"
#include "image.h"
#include "image_utils.h"
#include "model.h"
#include "output_postproc.h"
#include "timer.h"
#include "video.h"
#include "fomo_post_processing.h"

extern "C" {

#define MODEL_IN_W   144
#define MODEL_IN_H   144
#define MODEL_IN_C   1
#define MODEL_IN_COLOR_BGR 0

#define BOX_SCORE_THRESHOLD 0.8
#define MAX_OD_BOX_CNT  10

typedef struct tagODResult_t
{
    union {
        int16_t xyxy[4];
        struct {
            int16_t x1;
            int16_t y1;
            int16_t x2;
            int16_t y2;
        };
    };
    float score;
    int label;
} ODResult_t;

ODResult_t s_odRets[MAX_OD_BOX_CNT];
int s_odRetCnt = 0;

ODResult_t s_displayRets[MAX_OD_BOX_CNT];
int s_displayRetCnt = 0;

__attribute__((section (".model_input_buffer")))
static uint8_t model_input_buf[MODEL_IN_W * MODEL_IN_H * MODEL_IN_C] = {0};

uint32_t s_infUs = 0;
volatile uint8_t g_isImgBufReady = 0;

#define WND_X0 0
#define WND_Y0 0

static fomo::FomoDetection s_detResults[fomo::MAX_DETECTIONS];
static int s_detResultCount = 0;

/*******************************************************************************
 * NEW: Persistent class counters - these accumulate over time
 * Classes: 0=car, 1=heavy_vehicle, 2=person, 3=two_wheeler
 ******************************************************************************/
#define NUM_CLASSES 4
static volatile uint32_t s_persistentClassCounts[NUM_CLASSES] = {0, 0, 0, 0};

// Flag to track if we already counted detections this frame
// (prevents counting the same object multiple times per frame)
static volatile uint8_t s_frameProcessed = 0;

// Optional: Reset counters function (call from button/command if needed)
void ResetClassCounters(void)
{
    for (int i = 0; i < NUM_CLASSES; i++) {
        s_persistentClassCounts[i] = 0;
    }
}

// NEW: Function called by display.c to get current counts
void GetClassCounts(int* counts)
{
    for (int i = 0; i < NUM_CLASSES; i++) {
        counts[i] = (int)s_persistentClassCounts[i];
    }
}

/*******************************************************************************
 * Helper Functions
 ******************************************************************************/
static inline int clampi(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/*******************************************************************************
 * Safe Rectangle Drawing on Slice Buffer
 ******************************************************************************/
void draw_rect_on_slice_buffer(uint16_t* pCam, int srcW,
    int curY, int stride, ODResult_t *pODRet, int retCnt, int slice_height)
{
    (void)stride;

    for (int i = 0; i < retCnt; i++, pODRet++)
    {
        int x1 = pODRet->x1;
        int x2 = pODRet->x2;
        int y1 = pODRet->y1;
        int y2 = pODRet->y2;

        if (x2 < x1) { int t = x1; x1 = x2; x2 = t; }
        if (y2 < y1) { int t = y1; y1 = y2; y2 = t; }

        if (srcW <= 0 || slice_height <= 0) continue;
        if (srcW < 4) continue;

        x1 = clampi(x1, 0, srcW - 1);
        x2 = clampi(x2, 0, srcW - 1);

        int w = x2 - x1;
        if (w <= 0) continue;

        uint32_t color = 0xF800F800;

        int stripeY1 = y1 - curY;
        if (stripeY1 >= 0 && stripeY1 < slice_height) {
            for (int j = 0; j < 4 && (stripeY1 + j) < slice_height; j++) {
                uint16_t *pHorLine = pCam + srcW * (stripeY1 + j) + x1;
                memset(pHorLine, 0xF8, (size_t)w * 2);
            }
        }

        int stripeY2 = y2 - curY;
        if (stripeY2 >= 0 && stripeY2 < slice_height) {
            for (int j = 0; j < 4 && (stripeY2 + j) < slice_height; j++) {
                uint16_t *pHorLine = pCam + srcW * (stripeY2 + j) + x1;
                memset(pHorLine, 0xF8, (size_t)w * 2);
            }
        }

        int xL = clampi(x1, 0, srcW - 4);
        int xR = clampi(x2, 0, srcW - 4);

        uint16_t *pVtcLineL = pCam + xL;
        uint16_t *pVtcLineR = pCam + xR;

        int sliceY0 = curY;
        int sliceY1 = curY + slice_height;

        int drawY0 = (y1 + 1 > sliceY0) ? (y1 + 1) : sliceY0;
        int drawY1 = (y2 < sliceY1) ? y2 : sliceY1;

        int advance = drawY0 - curY;
        if (advance > 0) {
            pVtcLineL += advance * srcW;
            pVtcLineR += advance * srcW;
        }

        for (int y = drawY0; y < drawY1; y++) {
            memset(pVtcLineL, 0xF8, 4);
            memset(pVtcLineR, 0xF8, 4);
            pVtcLineL += srcW;
            pVtcLineR += srcW;
        }
    }
}

void FACEDET_DrawResults(uint16_t* pCam, int srcW, int curY, int sliceH)
{
    if (s_displayRetCnt > 0) {
        draw_rect_on_slice_buffer(pCam, srcW, curY, 1,
                                  s_displayRets, s_displayRetCnt, sliceH);
    }
}

/*******************************************************************************
 * Image Conversion Functions
 ******************************************************************************/
void Rgb565StridedToGray(const uint16_t* pIn, int srcW,
    int wndW, int wndH, int wndX0, int wndY0,
    uint8_t* pGray, int stride, uint8_t isSub128)
{
    const uint16_t* pSrc;
    uint8_t* pOut = pGray;

    for (int y = wndY0; y < wndH; y += stride)
    {
        pSrc = pIn + srcW * y + wndX0;
        for (int x = 0; x < wndW; x += stride)
        {
            uint16_t datIn = *pSrc;
            pSrc += stride;

            uint8_t r = (datIn >> 8) & 0xF8;
            uint8_t g = (datIn >> 3) & 0xFC;
            uint8_t b = (datIn << 3) & 0xF8;

            uint8_t gray = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);

            if (isSub128) gray ^= 0x80;

            *pOut++ = gray;
        }
    }
}

void Rgb565StridedToRgb888(const uint16_t* pIn, int srcW,
    int wndW, int wndH, int wndX0, int wndY0,
    uint8_t* p888, int stride, uint8_t isSub128)
{
    const uint16_t* pSrc;
    uint8_t* p888out = p888;

    for (int y = wndY0; y < wndH; y += stride)
    {
        pSrc = pIn + srcW * y + wndX0;

        for (int x = 0; x < wndW; x += stride)
        {
            uint16_t datIn = *pSrc;
            pSrc += stride;

            uint8_t r = (datIn >> 8) & 0xF8;
            uint8_t g = (datIn >> 3) & 0xFC;
            uint8_t b = (datIn << 3) & 0xF8;

            if (isSub128) { r ^= 0x80; g ^= 0x80; b ^= 0x80; }

            *p888out++ = r;
            *p888out++ = g;
            *p888out++ = b;
        }
    }
}

void Rgb565StridedToBgr888(const uint16_t* pIn, int srcW,
    int wndW, int wndH, int wndX0, int wndY0,
    uint8_t* p888, int stride, uint8_t isSub128)
{
    const uint16_t* pSrc;
    uint8_t* p888out = p888;

    for (int y = wndY0; y < wndH; y += stride)
    {
        pSrc = pIn + srcW * y + wndX0;

        for (int x = 0; x < wndW; x += stride)
        {
            uint16_t datIn = *pSrc;
            pSrc += stride;

            uint8_t r = (datIn >> 8) & 0xF8;
            uint8_t g = (datIn >> 3) & 0xFC;
            uint8_t b = (datIn << 3) & 0xF8;

            if (isSub128) { r ^= 0x80; g ^= 0x80; b ^= 0x80; }

            *p888out++ = b;
            *p888out++ = g;
            *p888out++ = r;
        }
    }
}

/*******************************************************************************
 * Slice Copy to Model Input
 ******************************************************************************/
void ezh_copy_slice_to_model_input(uint32_t idx,
                                  uint32_t cam_slice_buffer,
                                  uint32_t cam_slice_width,
                                  uint32_t cam_slice_height,
                                  uint32_t max_idx)
{
    static uint8_t* pCurDat;
    uint32_t curY;

    uint32_t s_imgStride = cam_slice_width / MODEL_IN_W;

    if (s_imgStride == 0) return;
    if (idx >= max_idx) return;

    uint32_t ndx = idx;
    curY = ndx * cam_slice_height;

    int wndY = (int)((s_imgStride - ((curY - WND_Y0) % s_imgStride)) % s_imgStride);

    if (idx + 1 >= max_idx)
        g_isImgBufReady = 1;

    const uint32_t inWndW = MODEL_IN_W * s_imgStride;
    const uint32_t outStartRow = (uint32_t)((cam_slice_height * ndx + (uint32_t)wndY) / s_imgStride);
    const uint32_t outW = MODEL_IN_W;

    pCurDat = model_input_buf + (size_t)1 * outW * outStartRow;

    const size_t BUF_SZ = sizeof(model_input_buf);

    uint32_t outH = 0;
    if (cam_slice_height > (uint32_t)wndY) {
        uint32_t effH = cam_slice_height - (uint32_t)wndY;
        outH = (effH + s_imgStride - 1) / s_imgStride;
    }

    size_t off = (size_t)(pCurDat - model_input_buf);
    size_t bytesToWrite = (size_t)outW * (size_t)outH * 1u;

    volatile uint32_t g_oob_count = 0;

    if (off > BUF_SZ || (off + bytesToWrite) > BUF_SZ) {
        g_oob_count++;
        return;
    }

    if (curY + cam_slice_height >= WND_Y0)
    {
        Rgb565StridedToGray((uint16_t*)cam_slice_buffer,
                            (int)cam_slice_width,
                            (int)inWndW,
                            (int)cam_slice_height,
                            WND_X0, wndY,
                            pCurDat,
                            (int)s_imgStride, 0);
        if (s_odRetCnt) {
            draw_rect_on_slice_buffer((uint16_t*)cam_slice_buffer,
                                      (int)cam_slice_width,
                                      (int)(idx * cam_slice_height),
                                      1,
                                      s_odRets,
                                      s_odRetCnt,
                                      (int)cam_slice_height);
        }
    }
}

/*******************************************************************************
 * UI String - Now simplified (grid handles display)
 ******************************************************************************/
static const char* const s_classShort[] = {
    "car", "hvy", "prs", "two"
};

// GetBriefString now returns minimal info - grid shows counts
const char* GetBriefString(void)
{
    // Return inference time only - class counts shown in grid
    static char sz[14];

    int ms = (int)(s_infUs / 1000);
    if (ms > 9999) ms = 9999;

    // Format: "NNNNms"
    char digits[5];
    int dLen = 0;
    int temp = ms;
    if (temp == 0) {
        digits[dLen++] = '0';
    } else {
        while (temp > 0 && dLen < 4) {
            digits[dLen++] = '0' + (temp % 10);
            temp /= 10;
        }
    }

    // Zero-pad to 4 digits
    int pos = 0;
    int pad = 4 - dLen;
    while (pad-- > 0) sz[pos++] = '0';
    for (int i = dLen - 1; i >= 0; i--) sz[pos++] = digits[i];
    sz[pos++] = 'm';
    sz[pos++] = 's';
    sz[pos] = '\0';

    return sz;
}

void MODEL_ODPrintResult(const ODResult_t *p, int retCnt)
{
    // Debug output disabled - can enable if needed
    // PRINTF("Found boxes count %d\r\n", retCnt);
}

/*******************************************************************************
 * Main Inference Loop
 ******************************************************************************/
void face_det()
{
    tensor_dims_t inputDims;
    tensor_type_t inputType;
    uint8_t* inputData;

    tensor_dims_t outputDims;
    tensor_type_t outputType;
    uint8_t* outputData;
    size_t arenaSize;

    if (MODEL_Init() != kStatus_Success)
    {
        PRINTF("Failed initializing model");
        for (;;) {}
    }

    size_t usedSize = MODEL_GetArenaUsedBytes(&arenaSize);
    PRINTF("\r\n%d/%d kB (%0.2f%%) tensor arena used\r\n",
           (int)(usedSize / 1024), (int)(arenaSize / 1024), 100.0 * usedSize / arenaSize);

    inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
    outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);

    uint32_t out_size = MODEL_GetOutputSize();
    if (inputDims.size != 4) {
        PRINTF("INPUTDIMERROR\r\n");
        for (;;) {}
    }

    if (out_size != 1) {
        PRINTF("Unexpected outputs_size=%lu. FOMO export is usually 1 output.\r\n", out_size);
    }

    TfLiteTensor* outputTensor0 = MODEL_GetOutputTensor(0);
    PRINTF("Output type=%d dims=", outputTensor0->type);
    for (int i = 0; i < outputTensor0->dims->size; i++) {
        PRINTF("%d ", outputTensor0->dims->data[i]);
    }
    PRINTF("\r\n");

    fomo::FomoPostProcessParams postProcessParams = {
        .threshold = BOX_SCORE_THRESHOLD,
        .peakOnly  = true
    };

    fomo::FomoPostProcess postProcess(
        outputTensor0,
        s_detResults,
        s_detResultCount,
        postProcessParams);

    while (1)
    {
        if (g_isImgBufReady == 0)
            continue;

        int H = (int)inputDims.data[1];
        int W = (int)inputDims.data[2];
        int C = (int)inputDims.data[3];

        int topPad = (H - MODEL_IN_H) / 2;
        uint8_t* buf = inputData + (topPad * MODEL_IN_W * MODEL_IN_C);

        memcpy(buf, model_input_buf, MODEL_IN_W * MODEL_IN_H * MODEL_IN_C);

        MODEL_ConvertInput(inputData, &inputDims, inputType);
        g_isImgBufReady = 0;

        auto startTime = TIMER_GetTimeInUS();
        MODEL_RunInference();
        auto endTime = TIMER_GetTimeInUS();

        s_infUs = (uint32_t)(endTime - startTime);

        s_detResultCount = 0;
        if (!postProcess.DoPostProcess()) {
            PRINTF("Post-processing failed.\r\n");
            s_odRetCnt = 0;
        }

        // Process detections and update persistent counters
        s_odRetCnt = 0;

        // Temporary counts for this frame (to avoid double-counting)
        int frameClassCounts[NUM_CLASSES] = {0, 0, 0, 0};

        for (int i = 0; i < s_detResultCount && s_odRetCnt < MAX_OD_BOX_CNT; i++) {
            const fomo::FomoDetection& det = s_detResults[i];

            const float scaleX = (float)CAMERA_WIDTH  / (float)MODEL_IN_W;
            const float scaleY = (float)CAMERA_HEIGHT / (float)MODEL_IN_H;

            const float camCx = det.cx * scaleX;
            const float camCy = det.cy * scaleY;

            const int16_t halfBoxX = 8;
            const int16_t halfBoxY = 8;

            s_odRets[s_odRetCnt].x1    = (int16_t)(camCx - halfBoxX);
            s_odRets[s_odRetCnt].x2    = (int16_t)(camCx + halfBoxX);
            s_odRets[s_odRetCnt].y1    = (int16_t)(camCy - halfBoxY);
            s_odRets[s_odRetCnt].y2    = (int16_t)(camCy + halfBoxY);
            s_odRets[s_odRetCnt].score = det.score;
            s_odRets[s_odRetCnt].label = det.cls - 1;  // Convert 1-4 to 0-3

            // Count this detection for the frame
            int classIdx = det.cls - 1;  // FOMO uses 1-4, we use 0-3
            if (classIdx >= 0 && classIdx < NUM_CLASSES) {
                frameClassCounts[classIdx]++;
            }

            s_odRetCnt++;
        }

        // Update persistent counters with this frame's detections
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (frameClassCounts[c] > 0) {
                s_persistentClassCounts[c] += frameClassCounts[c];

                // Cap at reasonable maximum (e.g., 99999)
                if (s_persistentClassCounts[c] > 99999) {
                    s_persistentClassCounts[c] = 99999;
                }
            }
        }

        if (s_odRetCnt > 0) {
            MODEL_ODPrintResult(s_odRets, s_odRetCnt);
            memcpy(s_displayRets, s_odRets, sizeof(ODResult_t) * s_odRetCnt);
            s_displayRetCnt = s_odRetCnt;
        }
        else {
            s_displayRetCnt = 0;
        }
    }
}

} // extern "C"
