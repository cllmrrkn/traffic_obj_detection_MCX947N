/*
 * YOLOv26 (Ultralytics-style) post-processing — heap-free version.
 *
 * All internal buffers are fixed-size static arrays.
 * No std::vector, no malloc, no new — safe for MCXN947 SRAM.
 *
 * Output tensor layout: (1, 10, 500) — (1, C, N), C=4+nc, N=500
 * Box format: normalized cx,cy,w,h in [0..1]
 */

#ifndef YOLO26_POST_PROCESSING_H
#define YOLO26_POST_PROCESSING_H

#include <stdint.h>
#include "tensorflow/lite/c/common.h"

namespace yolo26 {

// Maximum number of candidates before NMS (500 anchors total)
static constexpr int MAX_CANDIDATES = 500;

// Maximum detections kept after NMS
static constexpr int MAX_DETECTIONS = 20;

struct DetectionResult {
    float x0;       // top-left x in original image pixels
    float y0;       // top-left y in original image pixels
    float w;        // width  in original image pixels
    float h;        // height in original image pixels
    float score;    // sigmoid confidence
    int   cls;      // class index [0..nc-1]

    // Legacy fields kept for face_det.cpp compatibility
    float  m_x0{0};
    float  m_y0{0};
    float  m_w{0};
    float  m_h{0};
    int    m_class{0};
    float  m_score{0.0f};
    double m_normalisedVal{0.0};
};

struct PostProcessParams {
    int   originalImageWidth;
    int   originalImageHeight;
    float threshold;        // confidence threshold (post-sigmoid)
    float nms;              // IoU threshold for NMS
    int   numClasses;
    int   topN;             // 0 = keep all; >0 = cap at topN after NMS
};

class DetectorPostProcess {
public:
    DetectorPostProcess(const TfLiteTensor*  outputTensor,
                        DetectionResult*     resultsBuffer,
                        int&                 resultsCount,
                        const PostProcessParams& params);

    bool DoPostProcess();

private:
    const TfLiteTensor*      m_outputTensor;
    DetectionResult*         m_results;
    int&                     m_resultCount;
    const PostProcessParams& m_params;
};

} // namespace yolo26

#endif // YOLO26_POST_PROCESSING_H
