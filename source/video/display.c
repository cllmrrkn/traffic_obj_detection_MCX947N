/*
 * Copyright (c) 2013 - 2015, Freescale Semiconductor, Inc.
 * Copyright 2016-2017 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MODIFIED: Added 16x16 scaled font (2x), header at top, class grid below with separation
 */

#include "fsl_device_registers.h"
#include "fsl_debug_console.h"
#include "pin_mux.h"
#include "fsl_inputmux.h"
#include "clock_config.h"
#include "board.h"
#include "lcd_impl_flexio.h"
#include "st7796_lcd.h"
#include "video.h"
#include "font8x8_basic.h"
#include <stdio.h>

/****************************************************************
 * LCD display
 ****************************************************************/
__attribute__((section(".bss.$SRAMX"))) volatile uint16_t g_camera_display_buffer[CAMERA_WIDTH*BUFF_HEIGHT];

struct lcd_coordinate {
    uint16_t x_start;
    uint16_t y_start;
    uint16_t x_end;
    uint16_t y_end;
};

struct lcd_coordinate lcd_stripe_coordinate[4] = {
    {0, BUFF_HEIGHT*0, 320-1, BUFF_HEIGHT*1-1},
    {0, BUFF_HEIGHT*1, 320-1, BUFF_HEIGHT*2-1},
    {0, BUFF_HEIGHT*2, 320-1, BUFF_HEIGHT*3-1},
    {0, BUFF_HEIGHT*3, 320-1, BUFF_HEIGHT*4-1},
};

struct lcd_coordinate display_slice_area_320_240_brk[18] = {
    { 0, 40+15*0, 320U-1U, 40+15*1-1 },
    { 0, 40+15*1, 320U-1U, 40+15*2-1 },
    { 0, 40+15*2, 320U-1U, 40+15*3-1 },
    { 0, 40+15*3, 320U-1U, 40+15*4-1 },
    { 0, 40+15*4, 320U-1U, 40+15*5-1 },
    { 0, 40+15*5, 320U-1U, 40+15*6-1 },
    { 0, 40+15*6, 320U-1U, 40+15*7-1 },
    { 0, 40+15*7, 320U-1U, 40+15*8-1 },
    { 0, 40+15*8, 320U-1U, 40+15*9-1 },
    { 0, 40+15*9, 320U-1U, 40+15*10-1 },
    { 0, 40+15*10, 320U-1U, 40+15*11-1 },
    { 0, 40+15*11, 320U-1U, 40+15*12-1 },
    { 0, 40+15*12, 320U-1U, 40+15*13-1 },
    { 0, 40+15*13, 320U-1U, 40+15*14-1 },
    { 0, 40+15*14, 320U-1U, 40+15*15-1 },
    { 0, 40+15*15, 320U-1U, 40+15*16-1 },
};

struct lcd_coordinate display_slice_area_320_240_rotate_0[18] = {
    { 0, 15*0, 320U-1U, 15*1-1 },
    { 0, 15*1, 320U-1U, 15*2-1 },
    { 0, 15*2, 320U-1U, 15*3-1 },
    { 0, 15*3, 320U-1U, 15*4-1 },
    { 0, 15*4, 320U-1U, 15*5-1 },
    { 0, 15*5, 320U-1U, 15*6-1 },
    { 0, 15*6, 320U-1U, 15*7-1 },
    { 0, 15*7, 320U-1U, 15*8-1 },
    { 0, 15*8, 320U-1U, 15*9-1 },
    { 0, 15*9, 320U-1U, 15*10-1 },
    { 0, 15*10, 320U-1U, 15*11-1 },
    { 0, 15*11, 320U-1U, 15*12-1 },
    { 0, 15*12, 320U-1U, 15*13-1 },
    { 0, 15*13, 320U-1U, 15*14-1 },
    { 0, 15*14, 320U-1U, 15*15-1 },
    { 0, 15*15, 320U-1U, 15*16-1 },
};

struct lcd_coordinate Pic_nxp_ai_demo[5] = {
    { 448U, 16U, 800U-1U, 16+448-1U },
    { 224, 48, 480-1, 272-1 },
    { 224+96, 116-15, 480-1, 256-1-15 },
    { 0, 240, 320-1, 480-1 },
    { 0, 0, 480-1, 320-1 },
};

static lcd_impl_flexio_t s_lcd_impl;
static st7796_lcd_t s_lcd = {
    .config = {
#ifdef USE_BOARD_BRK
        .direction = ST7796_DIR_90,
        .pix_fmt   = ST7796_RGB565,
        .bgr_mode  = 1,
        .inversion = 0,
        .mirrored  = 0,
        .vflip     = 1,
#else
        .direction = ST7796_DIR_0,
        .pix_fmt   = ST7796_RGB565,
        .bgr_mode  = 1,
        .inversion = 0,
        .mirrored  = 0,
        .vflip     = 1,
#endif
    },
    .cb = {
        .reset_cb      = lcd_impl_reset,
        .write_cmd_cb  = lcd_impl_write_cmd,
        .write_data_cb = lcd_impl_write_data,
    },
    .user_data = &s_lcd_impl,
};

uint16_t POINT_COLOR = 0x0000;      // Black text
uint16_t BACK_COLOR  = 0xFFFF;      // White background

// Character buffer for 16x16 scaled font (2x scale of 8x8)
__attribute__((aligned(32))) volatile uint16_t g_char_buf_16x16[16][16] = {0};

// Line buffer for drawing separator
__attribute__((aligned(32))) volatile uint16_t g_line_buf[320] = {0};

/*******************************************************************************
 * Function Prototypes
 ******************************************************************************/
void display_show_string_16x16(uint16_t x, uint16_t y, uint16_t max_x, const char *p);
void display_draw_hline(uint16_t x, uint16_t y, uint16_t width, uint16_t color);
void display_window_clear(void);
void display_draw_header_and_grid(void);

__WEAK const char* GetBriefString(void) {
    return 0;
}

// External function to get class counts (implemented in face_det.cpp)
__WEAK void GetClassCounts(int* counts) {
    for (int i = 0; i < 4; i++) counts[i] = 0;
}

/*******************************************************************************
 * Display Initialization
 ******************************************************************************/
void display_init(void)
{
    lcd_impl_init(&s_lcd_impl);
    st7796_lcd_init(&s_lcd);
    display_window_clear();
}

/*******************************************************************************
 * 16x16 Font Conversion (2x scaled from 8x8)
 * Each pixel in 8x8 becomes a 2x2 block in 16x16
 ******************************************************************************/
void display_font_conversion_16x16(uint8_t ch, uint8_t mode)
{
    if (ch < 32 || ch > 127) ch = '?';

    uint8_t idx = ch - 32;
    const uint8_t* glyph = font8x8_basic[idx];

    for (int row = 0; row < 8; row++) {
        uint8_t rowData = glyph[row];
        for (int col = 0; col < 8; col++) {
            uint16_t color;
            // LSB is leftmost pixel in font8x8
            if (rowData & (1 << col)) {
                color = POINT_COLOR;
            } else if (mode == 0) {
                color = BACK_COLOR;
            } else {
                continue;
            }

            // Scale 2x: each 8x8 pixel becomes 2x2 block in 16x16
            int dstRow = row * 2;
            int dstCol = col * 2;
            g_char_buf_16x16[dstRow][dstCol]         = color;
            g_char_buf_16x16[dstRow][dstCol + 1]     = color;
            g_char_buf_16x16[dstRow + 1][dstCol]     = color;
            g_char_buf_16x16[dstRow + 1][dstCol + 1] = color;
        }
    }
}

/*******************************************************************************
 * String Display with 16x16 scaled font
 * max_x is the absolute X coordinate limit (not relative width)
 ******************************************************************************/
void display_show_string_16x16(uint16_t x, uint16_t y, uint16_t max_x, const char *p)
{
    uint16_t x0, x1, y0, y1;

    while (*p != '\0' && *p >= ' ' && *p <= '~') {
        // Check if character fits before drawing
        if (x + 16 > max_x) break;

        display_font_conversion_16x16((uint8_t)*p, 0);

        x0 = x;
        x1 = x + 15;
        y0 = y;
        y1 = y + 15;

        st7796_lcd_load(&s_lcd, (uint8_t*)g_char_buf_16x16, x0, x1, y0, y1);

        x += 16;
        p++;
    }
}

/*******************************************************************************
 * Draw horizontal line (separator)
 ******************************************************************************/
void display_draw_hline(uint16_t x, uint16_t y, uint16_t width, uint16_t color)
{
    // Fill line buffer with color
    for (int i = 0; i < width && i < 320; i++) {
        g_line_buf[i] = color;
    }

    // Draw 2-pixel thick line for visibility
    st7796_lcd_load(&s_lcd, (uint8_t*)g_line_buf, x, x + width - 1, y, y);
    st7796_lcd_load(&s_lcd, (uint8_t*)g_line_buf, x, x + width - 1, y + 1, y + 1);
}

/*******************************************************************************
 * ISR-safe integer to string conversion
 ******************************************************************************/
static int int_to_str(int val, char* buf, int bufSize)
{
    if (bufSize < 2) return 0;
    if (val < 0) val = 0;
    if (val > 99999) val = 99999;

    char tmp[8];
    int len = 0;
    if (val == 0) {
        tmp[len++] = '0';
    } else {
        while (val > 0 && len < 6) {
            tmp[len++] = '0' + (val % 10);
            val /= 10;
        }
    }

    int written = 0;
    for (int i = len - 1; i >= 0 && written < bufSize - 1; i--) {
        buf[written++] = tmp[i];
    }
    buf[written] = '\0';
    return written;
}

/*******************************************************************************
 * Draw Header and 2x2 Class Counter Grid with Separator
 *
 * Layout (for non-BRK board, below 240px camera view):
 *
 *   Y=242:  "Traffic Detect"
 *   Y=260:  "FOMO Model"
 *   Y=280:  ──────────────────── (separator line)
 *   Y=286:  car: XX    hvy: XX
 *   Y=304:  prs: XX    two: XX
 *
 ******************************************************************************/
void display_draw_header_and_grid(void)
{
    int counts[4] = {0};
    GetClassCounts(counts);

    static const char* labels[4] = {"car:", "hvy:", "prs:", "two:"};

#ifdef USE_BOARD_BRK
    const uint16_t INFO_X      = 325;
    const uint16_t HEADER_Y    = 100;
    const uint16_t HEADER2_Y   = 118;
    const uint16_t SEP_Y       = 140;
    const uint16_t GRID_Y      = 148;
    const uint16_t CELL_W      = 75;
    const uint16_t CELL_H      = 20;
    const uint16_t MAX_X       = 479;
    const uint16_t SEP_WIDTH   = 150;
#else
    const uint16_t INFO_X      = 8;
    const uint16_t HEADER_Y    = 242;
    const uint16_t HEADER2_Y   = 260;
    const uint16_t SEP_Y       = 280;      // Separator line Y position
    const uint16_t GRID_Y      = 288;      // Grid starts after separator
    const uint16_t CELL_W      = 120;
    const uint16_t CELL_H      = 25;
    const uint16_t MAX_X       = 319;      // Full screen width for text
    const uint16_t SEP_WIDTH   = 304;      // Separator width
#endif

    // ===== HEADER SECTION =====
#ifdef USE_BOARD_BRK
    display_show_string_16x16(INFO_X, HEADER_Y, MAX_X, "MCXN94x");
    display_show_string_16x16(INFO_X, HEADER2_Y, MAX_X, "FOMO Det");
#else
    // "Traffic Detect" = 14 chars * 16 pixels = 224 pixels needed
    display_show_string_16x16(INFO_X, HEADER_Y, MAX_X, "Traffic Detect");
    display_show_string_16x16(INFO_X, HEADER2_Y, MAX_X, "FOMO Model");
#endif

    // ===== SEPARATOR LINE =====
    display_draw_hline(INFO_X, SEP_Y, SEP_WIDTH, 0x0000);  // Black line

    // ===== GRID SECTION =====
    char numBuf[8];

    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 2; col++) {
            int classIdx = row * 2 + col;

            uint16_t cellX = INFO_X + col * CELL_W;
            uint16_t cellY = GRID_Y + row * CELL_H;

            // Draw label
            display_show_string_16x16(cellX, cellY, cellX + CELL_W, labels[classIdx]);

            // Convert count to string
            int_to_str(counts[classIdx], numBuf, sizeof(numBuf));

            // Draw count (offset by 4 chars * 16 = 64 pixels)
            display_show_string_16x16(cellX + 64, cellY, cellX + CELL_W + 32, numBuf);
        }
    }
}

/*******************************************************************************
 * Display Slice (called per stripe during camera DMA)
 ******************************************************************************/
void display_show_slice(uint32_t g_stripe_index, uint32_t buffer, uint32_t max_idx)
{
#ifdef USE_BOARD_BRK
    st7796_lcd_load(
        &s_lcd,
        (uint8_t*)(buffer),
        display_slice_area_320_240_brk[g_stripe_index].x_start,
        display_slice_area_320_240_brk[g_stripe_index].x_end,
        display_slice_area_320_240_brk[g_stripe_index].y_start,
        display_slice_area_320_240_brk[g_stripe_index].y_end
    );

    if (g_stripe_index + 1 == max_idx) {
        display_draw_header_and_grid();
    }
#else
    st7796_lcd_load(
        &s_lcd,
        (uint8_t*)(buffer),
        display_slice_area_320_240_rotate_0[g_stripe_index].x_start,
        display_slice_area_320_240_rotate_0[g_stripe_index].x_end,
        display_slice_area_320_240_rotate_0[g_stripe_index].y_start,
        display_slice_area_320_240_rotate_0[g_stripe_index].y_end
    );

    if (g_stripe_index + 1 == max_idx) {
        display_draw_header_and_grid();
    }
#endif
}

/*******************************************************************************
 * Clear Display Window
 ******************************************************************************/
void display_show_bg_image(void)
{
}

void display_window_clear(void)
{
    memset((void*)g_camera_display_buffer, 0xff, CAMERA_WIDTH * BUFF_HEIGHT * 2);
    int buffer_h = (CAMERA_WIDTH * BUFF_HEIGHT / 480);

    for (int i = 0; i < 480 / buffer_h; i++) {
        st7796_lcd_load(
            &s_lcd,
            (uint8_t*)(g_camera_display_buffer),
            0,
            480 - 1,
            i * buffer_h,
            i * buffer_h + buffer_h - 1
        );
    }
}
