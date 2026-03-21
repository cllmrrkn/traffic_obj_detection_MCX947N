/*
 * YOLOv26 post-processing — heap-free implementation for MCXN947.
 * No std::vector, no dynamic allocation anywhere.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <string.h>
#include "fsl_debug_console.h"
#include "yolo26_post_processing.h"

namespace yolo26 {

// ---------------------------------------------------------------------------
// Internal static buffers — allocated once in BSS, never freed
// ---------------------------------------------------------------------------

struct Candidate {
    float x1, y1, x2, y2;
    float score;
    int   cls;
};

static Candidate  s_candidates[MAX_CANDIDATES];
static uint8_t    s_suppressed[MAX_CANDIDATES];
static int        s_candidateCount;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline float dequant(const TfLiteTensor* t, int idx)
{
    const float scale = (t->params.scale != 0.0f) ? t->params.scale : 1.0f;
    const int32_t zp  = t->params.zero_point;

    if (t->type == kTfLiteFloat32) return t->data.f[idx];
    if (t->type == kTfLiteInt8)
        return (static_cast<int32_t>(t->data.int8[idx])  - zp) * scale;
    if (t->type == kTfLiteUInt8)
        return (static_cast<int32_t>(t->data.uint8[idx]) - zp) * scale;
    return 0.0f;
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float iou(const Candidate& a, const Candidate& b)
{
    const float ix1 = a.x1 > b.x1 ? a.x1 : b.x1;
    const float iy1 = a.y1 > b.y1 ? a.y1 : b.y1;
    const float ix2 = a.x2 < b.x2 ? a.x2 : b.x2;
    const float iy2 = a.y2 < b.y2 ? a.y2 : b.y2;

    const float iw = ix2 > ix1 ? ix2 - ix1 : 0.0f;
    const float ih = iy2 > iy1 ? iy2 - iy1 : 0.0f;
    const float inter = iw * ih;

    const float aA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float aB = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = aA + aB - inter;

    return (uni > 1e-6f) ? (inter / uni) : 0.0f;
}

// Simple insertion sort — N is small (≤500), mostly unsorted
static void sortCandidatesDesc(int count)
{
    for (int i = 1; i < count; i++) {
        Candidate key = s_candidates[i];
        int j = i - 1;
        while (j >= 0 && s_candidates[j].score < key.score) {
            s_candidates[j + 1] = s_candidates[j];
            j--;
        }
        s_candidates[j + 1] = key;
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DetectorPostProcess::DetectorPostProcess(const TfLiteTensor*      outputTensor,
                                         DetectionResult*         resultsBuffer,
                                         int&                     resultsCount,
                                         const PostProcessParams& params)
    : m_outputTensor(outputTensor),
      m_results(resultsBuffer),
      m_resultCount(resultsCount),
      m_params(params)
{
}

// ---------------------------------------------------------------------------
// DoPostProcess
// ---------------------------------------------------------------------------

bool DetectorPostProcess::DoPostProcess()
{
    m_resultCount = 0;

    if (!m_outputTensor || !m_outputTensor->dims ||
        m_outputTensor->dims->size != 3) {
        return false;
    }

    const int d1 = m_outputTensor->dims->data[1];  // C = 10
    const int d2 = m_outputTensor->dims->data[2];  // N = 500
    const int nc = m_params.numClasses;

    // Detect layout: (1, C, N) or (1, N, C)
    const bool isCN = (d1 == (4 + nc)) || (d1 == (5 + nc));
    const bool isNC = (d2 == (4 + nc)) || (d2 == (5 + nc));
    if (!isCN && !isNC) return false;

    const int C    = isCN ? d1 : d2;
    const int N    = isCN ? d2 : d1;
    const int cls0 = (C == (5 + nc)) ? 5 : 4;  // hasObj offset

    // Cap N to buffer size
    const int numAnchors = N < MAX_CANDIDATES ? N : MAX_CANDIDATES;

    const float origW = static_cast<float>(m_params.originalImageWidth);
    const float origH = static_cast<float>(m_params.originalImageHeight);
    const float maxX  = origW - 1.0f;
    const float maxY  = origH - 1.0f;

    // Lambda: get value at (anchor n, channel c)
    auto get = [&](int n, int c) -> float {
        const int idx = isCN ? (c * N + n) : (n * C + c);
        return dequant(m_outputTensor, idx);
    };

    // --- Decode anchors ---

    s_candidateCount = 0;

    for (int i = 0; i < numAnchors; i++) {
    	if (i < 3) {
    	    float cx = get(i,0), cy = get(i,1), bw = get(i,2), bh = get(i,3);
    	    float c0 = get(i, cls0+0);
    	    float c1 = get(i, cls0+1);
    	    PRINTF("i=%d raw xywh=%f %f %f %f cls0=%f cls1=%f\r\n", i, cx, cy, bw, bh, c0, c1);
    	}
        // Best class score via sigmoid
    	// Best class score (ASSUME already probability 0..1)
    	float bestScore = 0.0f;
    	int   bestCls   = 0;
    	for (int c = 0; c < nc; c++) {
    	    const float prob = get(i, cls0 + c);   // <<< FIX: removed sigmoid
    	    if (prob > bestScore) {
    	        bestScore = prob;
    	        bestCls   = c;
    	    }
    	}

        if (bestScore < m_params.threshold) continue;


        // Box — normalized x1,y1,w,h (TOP-LEFT + SIZE) -> pixel x1,y1,x2,y2
        const float x1n = get(i, 0);
        const float y1n = get(i, 1);
        float wn  = get(i, 2);
        float hn  = get(i, 3);

        // Safety clamps
        if (wn < 0.0f) wn = 0.0f;
        if (hn < 0.0f) hn = 0.0f;
        if (x1n < 0.0f) wn += x1n, (void)(wn < 0 ? wn = 0 : 0), (void)(0); // optional, can omit
        if (y1n < 0.0f) hn += y1n, (void)(hn < 0 ? hn = 0 : 0), (void)(0);
        if (wn > 1.0f) wn = 1.0f;
        if (hn > 1.0f) hn = 1.0f;

        float x1 = x1n * origW;
        float y1 = y1n * origH;
        float x2 = (x1n + wn) * origW;
        float y2 = (y1n + hn) * origH;

        // Clamp
        if (x1 < 0.0f) x1 = 0.0f; if (x1 > maxX) x1 = maxX;
        if (y1 < 0.0f) y1 = 0.0f; if (y1 > maxY) y1 = maxY;
        if (x2 < 0.0f) x2 = 0.0f; if (x2 > maxX) x2 = maxX;
        if (y2 < 0.0f) y2 = 0.0f; if (y2 > maxY) y2 = maxY;

        if (x2 <= x1 || y2 <= y1) continue;

        s_candidates[s_candidateCount++] = { x1, y1, x2, y2, bestScore, bestCls };
    }

    if (s_candidateCount == 0) return true;

    // --- Sort by score descending ---
    sortCandidatesDesc(s_candidateCount);

    // --- Greedy class-aware NMS ---
    for (int i = 0; i < s_candidateCount; i++) s_suppressed[i] = 0;

    const int maxKeep = (m_params.topN > 0) ? m_params.topN : MAX_DETECTIONS;

    for (int i = 0; i < s_candidateCount && m_resultCount < maxKeep; i++) {
        if (s_suppressed[i]) continue;

        // Write result
        const Candidate& c = s_candidates[i];
        DetectionResult& r = m_results[m_resultCount++];
        r.x0 = c.x1;
        r.y0 = c.y1;
        r.w  = c.x2 - c.x1;
        r.h  = c.y2 - c.y1;
        r.score = c.score;
        r.cls   = c.cls;
        // Legacy fields for face_det.cpp compatibility
        r.m_x0           = r.x0;
        r.m_y0           = r.y0;
        r.m_w            = r.w;
        r.m_h            = r.h;
        r.m_class        = r.cls;
        r.m_score        = r.score;
        r.m_normalisedVal = r.score;

        // Suppress overlapping same-class boxes
        for (int j = i + 1; j < s_candidateCount; j++) {
            if (s_suppressed[j]) continue;
            if (s_candidates[j].cls != c.cls) continue;
            if (iou(s_candidates[i], s_candidates[j]) > m_params.nms)
                s_suppressed[j] = 1;
        }
    }

    return true;
}

} // namespace yolo26
