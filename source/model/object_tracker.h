/*
 * Object Tracker & Counter — Centroid-based tracking for MCXN947
 * 
 * Algorithm: Euclidean Distance Centroid Tracking
 * - Lightweight, suitable for embedded systems with limited SRAM
 * - Matches detections across frames using minimum distance
 * - Handles object registration, tracking, and deregistration
 * - Counts objects crossing virtual boundaries
 * 
 * References:
 * - PyImageSearch Centroid Tracking: https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
 * - IOU Tracker paper: http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf
 * - Hackster.io ESP32 FOMO tracking: https://www.hackster.io/sologithu/object-detection-with-centroid-based-tracking-c-way-ddaee0
 */

#ifndef OBJECT_TRACKER_H
#define OBJECT_TRACKER_H

#include <stdint.h>
#include <stdbool.h>
#include "fomo_post_processing.h"

namespace tracker {

// ============================================================================
// Configuration — tune these for your application
// ============================================================================

// Maximum tracked objects (static allocation, no heap)
static constexpr int MAX_TRACKED_OBJECTS = 16;

// Maximum distance (pixels) to associate detection with existing track
// If detection is farther than this from all tracks, it becomes a new object
static constexpr float MAX_ASSOCIATION_DISTANCE = 40.0f;

// Frames before deregistering a "lost" object
static constexpr int MAX_FRAMES_DISAPPEARED = 7;

// Minimum frames an object must be tracked before counting (reduces false counts)
static constexpr int MIN_FRAMES_BEFORE_COUNT = 2;

// Image dimensions (must match FOMO model input)
static constexpr float IMAGE_WIDTH  = 144.0f;
static constexpr float IMAGE_HEIGHT = 144.0f;

// ============================================================================
// Vertical Line Counting Configuration (for left/right movement)
// ============================================================================

// X-coordinate of the counting line (in model pixels, 0-144)
// Objects crossing this line will be counted
static constexpr float COUNT_LINE_X = 72.0f;  // center of image

// Hysteresis band (pixels) to prevent jitter-induced double counts
// Object must move beyond LINE_X ± HYSTERESIS to register a crossing
static constexpr float LINE_HYSTERESIS = 4.0f;

// ============================================================================
// Data Structures
// ============================================================================

// Direction of object travel when crossing the line
enum class CrossDirection : uint8_t {
    NONE = 0,
    RIGHT,  // crossing from left to right (increasing X)
    LEFT    // crossing from right to left (decreasing X)
};

// Position relative to counting line
enum class LinePosition : uint8_t {
    LEFT_SIDE,   // cx < COUNT_LINE_X - HYSTERESIS
    ON_LINE,     // within hysteresis band
    RIGHT_SIDE   // cx > COUNT_LINE_X + HYSTERESIS
};

// Single tracked object structure
// Designed for static allocation (no heap)
struct TrackedObject {
    // Core tracking data
    float cx;                   // current centroid X (pixels)
    float cy;                   // current centroid Y (pixels)
    float prev_cx;              // previous centroid X (for direction)
    float prev_cy;              // previous centroid Y (for direction)
    
    // Object identity
    uint32_t id;                // unique object ID
    int cls;                    // class index (1=car, 2=heavy_vehicle, etc.)
    
    // Tracking state
    int frames_tracked;         // total frames this object has been tracked
    int frames_disappeared;     // consecutive frames not detected
    bool active;                // slot is in use
    bool counted;               // has crossed line (prevent double-count)
    LinePosition last_position; // last known position relative to line
    
    // Direction/velocity estimation (for prediction during occlusion)
    float velocity_x;           // estimated velocity X
    float velocity_y;           // estimated velocity Y
};

// Per-class counting statistics
struct ClassCounts {
    uint32_t total;             // total objects counted for this class
    uint32_t crossed_right;     // objects that crossed line going right
    uint32_t crossed_left;      // objects that crossed line going left
};

// Overall tracker statistics
struct TrackerStats {
    ClassCounts by_class[fomo::NUM_CLASSES];  // counts per class
    uint32_t total_right;                      // all classes going right
    uint32_t total_left;                       // all classes going left
    uint32_t active_tracks;                    // currently tracked objects
    uint32_t next_object_id;                   // next ID to assign
    uint32_t frames_processed;                 // total inference frames
};

// ============================================================================
// Tracker Class
// ============================================================================

class ObjectTracker {
public:
    ObjectTracker();
    
    // Main update function — call after each FOMO inference
    // Returns number of line crossings detected this frame
    int Update(const fomo::FomoDetection* detections, int detection_count);
    
    // Get current statistics
    const TrackerStats& GetStats() const { return m_stats; }
    
    // Get specific class count
    uint32_t GetClassCount(int cls) const;
    
    // Get counts by direction
    uint32_t GetTotalRight() const { return m_stats.total_right; }
    uint32_t GetTotalLeft() const { return m_stats.total_left; }
    
    // Get all active tracked objects
    const TrackedObject* GetTrackedObjects() const { return m_objects; }
    int GetActiveCount() const { return m_stats.active_tracks; }
    
    // Reset all tracking state and counts
    void Reset();
    
    // Debug: print current state
    void DebugPrint() const;

private:
    // Tracked objects pool (static allocation)
    TrackedObject m_objects[MAX_TRACKED_OBJECTS];
    
    // Statistics
    TrackerStats m_stats;
    
    // ---- Internal methods ----
    
    // Find best matching detection for each tracked object
    // Uses Hungarian-like greedy assignment based on Euclidean distance
    void AssociateDetections(const fomo::FomoDetection* detections, 
                             int detection_count,
                             int* assignment,      // output: detection index per track (-1 if none)
                             bool* det_used);      // output: which detections were matched
    
    // Register a new object
    int RegisterObject(float cx, float cy, int cls);
    
    // Deregister an object (mark slot as free)
    void DeregisterObject(int slot);
    
    // Update existing object with new detection
    void UpdateObject(int slot, float cx, float cy, int cls);
    
    // Check if object crossed the counting line
    // Returns crossing direction (NONE if no crossing)
    CrossDirection CheckLineCrossing(int slot);
    
    // Calculate Euclidean distance between two points
    static float Distance(float x1, float y1, float x2, float y2);
    
    // Get position relative to counting line (with hysteresis)
    static LinePosition GetLinePosition(float cx);
};

} // namespace tracker

#endif // OBJECT_TRACKER_H
