/*
 * Object Tracker & Counter — Implementation
 * 
 * Centroid-based tracking with horizontal line-crossing counting.
 * Optimized for embedded systems — no dynamic allocation.
 * 
 * Algorithm Flow:
 * 1. Receive FOMO detections each frame
 * 2. Compute distance matrix between existing tracks and new detections
 * 3. Greedily assign detections to tracks (minimum distance first)
 * 4. Update matched tracks with new positions
 * 5. Increment disappeared counter for unmatched tracks
 * 6. Register new objects for unmatched detections
 * 7. Deregister tracks that exceeded MAX_FRAMES_DISAPPEARED
 * 8. Check line crossings and count objects
 */

#include <string.h>
#include <math.h>
#include <float.h>
#include "fsl_debug_console.h"
#include "object_tracker.h"

namespace tracker {

// ============================================================================
// Constructor
// ============================================================================

ObjectTracker::ObjectTracker() {
    Reset();
}

// ============================================================================
// Reset — clear all state
// ============================================================================

void ObjectTracker::Reset() {
    // Clear all tracked objects
    memset(m_objects, 0, sizeof(m_objects));
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
        m_objects[i].active = false;
        m_objects[i].last_position = LinePosition::ON_LINE;
    }
    
    // Clear statistics
    memset(&m_stats, 0, sizeof(m_stats));
    m_stats.next_object_id = 1;  // Start IDs at 1
}

// ============================================================================
// Main Update — called after each inference
// ============================================================================

int ObjectTracker::Update(const fomo::FomoDetection* detections, int detection_count) {
    m_stats.frames_processed++;
    int crossings_this_frame = 0;
    
    // Temporary arrays for association (stack allocated)
    int assignment[MAX_TRACKED_OBJECTS];        // detection index for each track
    bool det_used[fomo::MAX_DETECTIONS];        // which detections are matched
    
    // Initialize
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) assignment[i] = -1;
    for (int i = 0; i < fomo::MAX_DETECTIONS; i++) det_used[i] = false;
    
    // Step 1: Associate detections with existing tracks
    AssociateDetections(detections, detection_count, assignment, det_used);
    
    // Step 2: Update matched tracks and handle unmatched tracks
    for (int slot = 0; slot < MAX_TRACKED_OBJECTS; slot++) {
        if (!m_objects[slot].active) continue;
        
        if (assignment[slot] >= 0) {
            // Track was matched — update with new detection
            int det_idx = assignment[slot];
            UpdateObject(slot, detections[det_idx].cx, detections[det_idx].cy, 
                        detections[det_idx].cls);
            
            // Check if this object crossed the counting line
            CrossDirection dir = CheckLineCrossing(slot);
            if (dir != CrossDirection::NONE) {
                crossings_this_frame++;
            }
        } else {
            // Track was NOT matched — increment disappeared counter
            m_objects[slot].frames_disappeared++;
            
            // Predict position using velocity (simple linear prediction)
            // This helps maintain track during brief occlusions
            if (m_objects[slot].frames_disappeared <= 2) {
                m_objects[slot].prev_cx = m_objects[slot].cx;
                m_objects[slot].prev_cy = m_objects[slot].cy;
                m_objects[slot].cx += m_objects[slot].velocity_x;
                m_objects[slot].cy += m_objects[slot].velocity_y;
                
                // Clamp to image bounds
                if (m_objects[slot].cx < 0) m_objects[slot].cx = 0;
                if (m_objects[slot].cy < 0) m_objects[slot].cy = 0;
                if (m_objects[slot].cx > IMAGE_WIDTH) m_objects[slot].cx = IMAGE_WIDTH;
                if (m_objects[slot].cy > IMAGE_HEIGHT) m_objects[slot].cy = IMAGE_HEIGHT;
                
                // Still check for line crossing during prediction
                CrossDirection dir = CheckLineCrossing(slot);
                if (dir != CrossDirection::NONE) {
                    crossings_this_frame++;
                }
            }
            
            // Deregister if disappeared too long
            if (m_objects[slot].frames_disappeared > MAX_FRAMES_DISAPPEARED) {
                DeregisterObject(slot);
            }
        }
    }
    
    // Step 3: Register new objects for unmatched detections
    for (int i = 0; i < detection_count; i++) {
        if (!det_used[i]) {
            int slot = RegisterObject(detections[i].cx, detections[i].cy, detections[i].cls);
            if (slot >= 0) {
                PRINTF("TRACKER: New object #%lu registered (class %d) at (%.1f, %.1f)\r\n",
                       m_objects[slot].id, detections[i].cls, 
                       detections[i].cx, detections[i].cy);
            }
        }
    }
    
    // Update active track count
    m_stats.active_tracks = 0;
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
        if (m_objects[i].active) m_stats.active_tracks++;
    }
    
    return crossings_this_frame;
}

// ============================================================================
// Detection-to-Track Association (Greedy Hungarian-like)
// ============================================================================

void ObjectTracker::AssociateDetections(const fomo::FomoDetection* detections,
                                        int detection_count,
                                        int* assignment,
                                        bool* det_used) {
    // Distance matrix: distances[track_slot][detection_idx]
    float distances[MAX_TRACKED_OBJECTS][fomo::MAX_DETECTIONS];
    bool track_assigned[MAX_TRACKED_OBJECTS];
    
    // Initialize
    for (int t = 0; t < MAX_TRACKED_OBJECTS; t++) {
        track_assigned[t] = false;
        assignment[t] = -1;
        for (int d = 0; d < fomo::MAX_DETECTIONS; d++) {
            distances[t][d] = FLT_MAX;
        }
    }
    
    // Compute distances for active tracks
    int active_count = 0;
    for (int t = 0; t < MAX_TRACKED_OBJECTS; t++) {
        if (!m_objects[t].active) continue;
        active_count++;
        
        for (int d = 0; d < detection_count; d++) {
            // Optional: Only match same class (uncomment for stricter matching)
            // if (m_objects[t].cls != detections[d].cls) continue;
            
            distances[t][d] = Distance(m_objects[t].cx, m_objects[t].cy,
                                       detections[d].cx, detections[d].cy);
        }
    }
    
    // Greedy assignment: repeatedly pick minimum distance pair
    int pairs_to_assign = (active_count < detection_count) ? active_count : detection_count;
    
    for (int p = 0; p < pairs_to_assign; p++) {
        float min_dist = FLT_MAX;
        int best_track = -1;
        int best_det = -1;
        
        // Find minimum unassigned pair
        for (int t = 0; t < MAX_TRACKED_OBJECTS; t++) {
            if (!m_objects[t].active || track_assigned[t]) continue;
            
            for (int d = 0; d < detection_count; d++) {
                if (det_used[d]) continue;
                
                if (distances[t][d] < min_dist) {
                    min_dist = distances[t][d];
                    best_track = t;
                    best_det = d;
                }
            }
        }
        
        // Check if minimum distance is within threshold
        if (best_track >= 0 && min_dist <= MAX_ASSOCIATION_DISTANCE) {
            assignment[best_track] = best_det;
            track_assigned[best_track] = true;
            det_used[best_det] = true;
        } else {
            // No more valid pairs
            break;
        }
    }
}

// ============================================================================
// Register New Object
// ============================================================================

int ObjectTracker::RegisterObject(float cx, float cy, int cls) {
    // Find free slot
    int slot = -1;
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
        if (!m_objects[i].active) {
            slot = i;
            break;
        }
    }
    
    if (slot < 0) {
        PRINTF("TRACKER: WARNING — No free slots, cannot register new object\r\n");
        return -1;
    }
    
    // Initialize object
    TrackedObject& obj = m_objects[slot];
    obj.cx = cx;
    obj.cy = cy;
    obj.prev_cx = cx;
    obj.prev_cy = cy;
    obj.id = m_stats.next_object_id++;
    obj.cls = cls;
    obj.frames_tracked = 1;
    obj.frames_disappeared = 0;
    obj.active = true;
    obj.counted = false;
    obj.velocity_x = 0.0f;
    obj.velocity_y = 0.0f;
    
    // Initialize position relative to counting line (using X coordinate for vertical line)
    obj.last_position = GetLinePosition(cx);
    
    return slot;
}

// ============================================================================
// Deregister Object
// ============================================================================

void ObjectTracker::DeregisterObject(int slot) {
    if (slot >= 0 && slot < MAX_TRACKED_OBJECTS) {
        PRINTF("TRACKER: Object #%lu deregistered (tracked %d frames, counted=%d)\r\n",
               m_objects[slot].id, m_objects[slot].frames_tracked, 
               m_objects[slot].counted ? 1 : 0);
        m_objects[slot].active = false;
    }
}

// ============================================================================
// Update Object with New Detection
// ============================================================================

void ObjectTracker::UpdateObject(int slot, float cx, float cy, int cls) {
    TrackedObject& obj = m_objects[slot];
    
    // Store previous position
    obj.prev_cx = obj.cx;
    obj.prev_cy = obj.cy;
    
    // Update position
    obj.cx = cx;
    obj.cy = cy;
    
    // Update velocity estimate (exponential moving average)
    float new_vx = cx - obj.prev_cx;
    float new_vy = cy - obj.prev_cy;
    obj.velocity_x = 0.7f * obj.velocity_x + 0.3f * new_vx;
    obj.velocity_y = 0.7f * obj.velocity_y + 0.3f * new_vy;
    
    // Update class (could use majority voting for robustness)
    obj.cls = cls;
    
    // Update counters
    obj.frames_tracked++;
    obj.frames_disappeared = 0;
}

// ============================================================================
// Check Line Crossing — vertical line for left/right movement
// ============================================================================

CrossDirection ObjectTracker::CheckLineCrossing(int slot) {
    TrackedObject& obj = m_objects[slot];
    
    // Don't count if already counted
    if (obj.counted) return CrossDirection::NONE;
    
    // Don't count if not tracked long enough (reduces noise)
    if (obj.frames_tracked < MIN_FRAMES_BEFORE_COUNT) return CrossDirection::NONE;
    
    // Get current position relative to line (using X coordinate)
    LinePosition current_pos = GetLinePosition(obj.cx);
    
    // Check for crossing: must go from LEFT to RIGHT or vice versa
    // Being ON_LINE doesn't count — must fully cross the hysteresis band
    CrossDirection direction = CrossDirection::NONE;
    
    if (obj.last_position == LinePosition::LEFT_SIDE && current_pos == LinePosition::RIGHT_SIDE) {
        // Crossed rightward (left to right)
        direction = CrossDirection::RIGHT;
        
        // Update statistics
        m_stats.by_class[obj.cls].total++;
        m_stats.by_class[obj.cls].crossed_right++;
        m_stats.total_right++;
        
        obj.counted = true;
        
        PRINTF("TRACKER: Object #%lu (class %d: %s) crossed RIGHT! Class total=%lu\r\n",
               obj.id, obj.cls, fomo::CLASS_NAMES[obj.cls],
               m_stats.by_class[obj.cls].total);
    }
    else if (obj.last_position == LinePosition::RIGHT_SIDE && current_pos == LinePosition::LEFT_SIDE) {
        // Crossed leftward (right to left)
        direction = CrossDirection::LEFT;
        
        // Update statistics
        m_stats.by_class[obj.cls].total++;
        m_stats.by_class[obj.cls].crossed_left++;
        m_stats.total_left++;
        
        obj.counted = true;
        
        PRINTF("TRACKER: Object #%lu (class %d: %s) crossed LEFT! Class total=%lu\r\n",
               obj.id, obj.cls, fomo::CLASS_NAMES[obj.cls],
               m_stats.by_class[obj.cls].total);
    }
    
    // Update last known position (only if not ON_LINE to maintain hysteresis)
    if (current_pos != LinePosition::ON_LINE) {
        obj.last_position = current_pos;
    }
    
    return direction;
}

// ============================================================================
// Utility: Get Position Relative to Vertical Line (using X coordinate)
// ============================================================================

LinePosition ObjectTracker::GetLinePosition(float cx) {
    if (cx < COUNT_LINE_X - LINE_HYSTERESIS) {
        return LinePosition::LEFT_SIDE;
    }
    else if (cx > COUNT_LINE_X + LINE_HYSTERESIS) {
        return LinePosition::RIGHT_SIDE;
    }
    return LinePosition::ON_LINE;
}

// ============================================================================
// Utility: Euclidean Distance
// ============================================================================

float ObjectTracker::Distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

// ============================================================================
// Get Class Count
// ============================================================================

uint32_t ObjectTracker::GetClassCount(int cls) const {
    if (cls >= 0 && cls < fomo::NUM_CLASSES) {
        return m_stats.by_class[cls].total;
    }
    return 0;
}

// ============================================================================
// Debug Print
// ============================================================================

void ObjectTracker::DebugPrint() const {
    PRINTF("\r\n=== TRACKER STATUS (Frame %lu) ===\r\n", m_stats.frames_processed);
    PRINTF("Active tracks: %lu\r\n", m_stats.active_tracks);
    PRINTF("Line crossings — Right: %lu, Left: %lu\r\n",
           m_stats.total_right, m_stats.total_left);
    
    PRINTF("Counts by class:\r\n");
    for (int c = 1; c < fomo::NUM_CLASSES; c++) {
        PRINTF("  %s: %lu (right=%lu, left=%lu)\r\n",
               fomo::CLASS_NAMES[c],
               m_stats.by_class[c].total,
               m_stats.by_class[c].crossed_right,
               m_stats.by_class[c].crossed_left);
    }
    
    PRINTF("Active objects:\r\n");
    for (int i = 0; i < MAX_TRACKED_OBJECTS; i++) {
        if (m_objects[i].active) {
            const char* pos_str = (m_objects[i].last_position == LinePosition::LEFT_SIDE) ? "LEFT" :
                                  (m_objects[i].last_position == LinePosition::RIGHT_SIDE) ? "RIGHT" : "ON_LINE";
            PRINTF("  [%d] ID=%lu cls=%d pos=(%.1f,%.1f) %s frames=%d counted=%d\r\n",
                   i, m_objects[i].id, m_objects[i].cls,
                   m_objects[i].cx, m_objects[i].cy, pos_str,
                   m_objects[i].frames_tracked,
                   m_objects[i].counted ? 1 : 0);
        }
    }
    PRINTF("================================\r\n\r\n");
}

} // namespace tracker
