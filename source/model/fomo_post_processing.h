/*
 * FOMO post-processing — heap-free implementation for MCXN947.
 * Model: Edge Impulse FOMO, input 128x128, output (1, 16, 16, 5)
 * Classes: 0=background, 1=car, 2=heavy_vehicle, 3=person, 4=two_wheeler
 *
 * No std::vector, no dynamic allocation — safe for MCXN947 SRAM.
 */

#ifndef FOMO_POST_PROCESSING_H
#define FOMO_POST_PROCESSING_H

#include <stdint.h>
#include "tensorflow/lite/c/common.h"

namespace fomo {

// Grid and image parameters — match your model exactly
static constexpr int GRID_H        = 18;
static constexpr int GRID_W        = 18;
static constexpr int NUM_CLASSES   = 5;   // includes background at index 0
static constexpr int INPUT_W       = 144;
static constexpr int INPUT_H       = 144;
static constexpr int STRIDE        = INPUT_W / GRID_W;  // 8 pixels per cell

// Maximum detections kept after peak suppression
static constexpr int MAX_DETECTIONS = 10;

// Class names — index matches output tensor channel
static constexpr const char* CLASS_NAMES[NUM_CLASSES] = {
    "background",
    "car",
    "heavy_vehicle",
    "person",
    "two_wheeler"
};

struct FomoDetection {
    float cx;       // centroid x in input image pixels
    float cy;       // centroid y in input image pixels
    float score;    // class probability
    int   cls;      // class index 1..4 (0=background, never returned)
    const char* label;  // pointer to class name string
};

struct FomoPostProcessParams {
    float threshold;    // minimum class probability to report (e.g. 0.5)
    bool  peakOnly;     // if true, only report local maxima (recommended)
};

class FomoPostProcess {
public:
    FomoPostProcess(const TfLiteTensor*        outputTensor,
                    FomoDetection*             resultsBuffer,
                    int&                       resultsCount,
                    const FomoPostProcessParams& params);

    bool DoPostProcess();

private:
    const TfLiteTensor*          m_outputTensor;
    FomoDetection*               m_results;
    int&                         m_resultCount;
    const FomoPostProcessParams& m_params;

    float dequant(int idx) const;
    // Get probability for class c at grid cell (row, col)
    float getProb(int row, int col, int cls) const;
    // Check if cell (row, col, cls) is a local maximum among 4-neighbors
    bool  isLocalMax(int row, int col, int cls, float score) const;
};

} // namespace fomo

#endif // FOMO_POST_PROCESSING_H
