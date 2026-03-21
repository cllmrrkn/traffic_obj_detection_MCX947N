/*
 * FOMO post-processing — heap-free implementation for MCXN947.
 * Model output tensor: (1, 16, 16, 5) — softmax probabilities
 * Layout: [batch=0][row][col][class]
 *
 * Post-processing steps:
 *   1. Scan every grid cell
 *   2. Find the argmax class (skip background=0)
 *   3. Threshold against params.threshold
 *   4. Optionally suppress non-local-maxima (peakOnly=true recommended)
 *   5. Convert surviving cells to image centroids
 */

#include <string.h>
#include "fsl_debug_console.h"
#include "fomo_post_processing.h"

namespace fomo {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

FomoPostProcess::FomoPostProcess(const TfLiteTensor*          outputTensor,
                                 FomoDetection*               resultsBuffer,
                                 int&                         resultsCount,
                                 const FomoPostProcessParams& params)
    : m_outputTensor(outputTensor),
      m_results(resultsBuffer),
      m_resultCount(resultsCount),
      m_params(params)
{
}

// ---------------------------------------------------------------------------
// Dequantize a single tensor element by flat index
// ---------------------------------------------------------------------------

float FomoPostProcess::dequant(int idx) const
{
    const float    scale = (m_outputTensor->params.scale != 0.0f)
                               ? m_outputTensor->params.scale : 1.0f;
    const int32_t  zp    = m_outputTensor->params.zero_point;

    switch (m_outputTensor->type) {
        case kTfLiteFloat32:
            return m_outputTensor->data.f[idx];
        case kTfLiteInt8:
            return (static_cast<int32_t>(m_outputTensor->data.int8[idx]) - zp)
                   * scale;
        case kTfLiteUInt8:
            return (static_cast<int32_t>(m_outputTensor->data.uint8[idx]) - zp)
                   * scale;
        default:
            return 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Get probability for class cls at grid cell (row, col)
// Tensor layout: (1, GRID_H, GRID_W, NUM_CLASSES)
// flat index    = row * GRID_W * NUM_CLASSES + col * NUM_CLASSES + cls
// ---------------------------------------------------------------------------

float FomoPostProcess::getProb(int row, int col, int cls) const
{
    const int idx = row * GRID_W * NUM_CLASSES
                  + col * NUM_CLASSES
                  + cls;
    return dequant(idx);
}

// ---------------------------------------------------------------------------
// Local maximum check — 4-connected neighbors (up, down, left, right)
// A cell is a peak if its score >= all valid neighbors for the same class
// ---------------------------------------------------------------------------

bool FomoPostProcess::isLocalMax(int row, int col, int cls, float score) const
{
    // Up
    if (row > 0 && getProb(row - 1, col, cls) > score) return false;
    // Down
    if (row < GRID_H - 1 && getProb(row + 1, col, cls) > score) return false;
    // Left
    if (col > 0 && getProb(row, col - 1, cls) > score) return false;
    // Right
    if (col < GRID_W - 1 && getProb(row, col + 1, cls) > score) return false;

    return true;
}

// ---------------------------------------------------------------------------
// DoPostProcess — main entry point
// ---------------------------------------------------------------------------

bool FomoPostProcess::DoPostProcess()
{
    m_resultCount = 0;

    // Validate tensor shape: must be (1, 16, 16, 5)
    if (!m_outputTensor || !m_outputTensor->dims ||
        m_outputTensor->dims->size != 4) {
        PRINTF("FOMO: unexpected tensor rank %d\r\n",
               m_outputTensor ? m_outputTensor->dims->size : -1);
        return false;
    }

    const int d1 = m_outputTensor->dims->data[1];  // GRID_H
    const int d2 = m_outputTensor->dims->data[2];  // GRID_W
    const int d3 = m_outputTensor->dims->data[3];  // NUM_CLASSES

    if (d1 != GRID_H || d2 != GRID_W || d3 != NUM_CLASSES) {
        PRINTF("FOMO: shape mismatch got (%d,%d,%d) expected (%d,%d,%d)\r\n",
               d1, d2, d3, GRID_H, GRID_W, NUM_CLASSES);
        return false;
    }

    // Scan every grid cell
    for (int row = 0; row < GRID_H && m_resultCount < MAX_DETECTIONS; row++) {
        for (int col = 0; col < GRID_W && m_resultCount < MAX_DETECTIONS; col++) {

            // Find argmax class, skipping background (index 0)
            float bestScore = 0.0f;
            int   bestCls   = 0;

            for (int c = 1; c < NUM_CLASSES; c++) {  // start at 1, skip background
                const float prob = getProb(row, col, c);
                if (prob > bestScore) {
                    bestScore = prob;
                    bestCls   = c;
                }
            }

            // Threshold check
            if (bestScore < m_params.threshold) continue;

            // Local maximum check (suppresses blobs to single centroid)
            if (m_params.peakOnly && !isLocalMax(row, col, bestCls, bestScore))
                continue;

            // Convert grid cell to image centroid
            // Cell (row, col) covers pixels [col*STRIDE .. col*STRIDE+STRIDE-1]
            // Centroid is at the center of the cell
            const float cx = static_cast<float>(col * STRIDE) + (STRIDE * 0.5f);
            const float cy = static_cast<float>(row * STRIDE) + (STRIDE * 0.5f);

            // Write detection
            FomoDetection& det = m_results[m_resultCount++];
            det.cx    = cx;
            det.cy    = cy;
            det.score = bestScore;
            det.cls   = bestCls;
            det.label = CLASS_NAMES[bestCls];
        }
    }

    return true;
}

} // namespace fomo
