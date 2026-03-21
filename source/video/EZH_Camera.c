/*
 * Copyright (c) 2013 - 2015, Freescale Semiconductor, Inc.
 * Copyright 2016-2017 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "fsl_device_registers.h"
#include "fsl_debug_console.h"
#include "pin_mux.h"
#include "fsl_inputmux.h"
#include "clock_config.h"
#include "board.h"
#include "lcd_impl_flexio.h"
#include "st7796_lcd.h"
#include "fsl_clock.h"
#include "fsl_smartdma.h"
#include "ov7670.h"
#include "string.h"
#include "video.h"

/*******************************************************************************
 * Prototypes
 ******************************************************************************/
/*******************************************************************************
 * Variables
 ******************************************************************************/
/****************************************************************
SmartDMA camera
****************************************************************/
smartdma_camera_param_t smartdmaParam;                  /*!< SMARTDMA function parameters. */

__attribute__((section (".bss.$SRAMX")))
uint32_t g_camera_buffer[320*15*2*2/4] __attribute((aligned (32)));

__attribute__((section (".ezh_code"))) uint32_t ezh_code[512] __attribute((aligned(4)));
volatile uint32_t __attribute((aligned(4))) p_coord_index=0;
volatile uint8_t  __attribute((aligned(4))) g_samrtdma_stack[64];

volatile uint32_t g_data_ready=0;
//volatile uint8_t g_camera_buffer[480*320*2];
volatile uint32_t g_camera_complete_flag=0;

extern void FACEDET_DrawResults(uint16_t* pCam, int srcW, int curY, int sliceH);




/*******************************************************************************
 * Code
 ******************************************************************************/
static void ezh_camera_callback(void *param){
	g_camera_complete_flag = 1;
	NVIC_SetPendingIRQ(PLU_IRQn);
 }


/*!
 * @brief Main function
 */
void ezh_start(void)
{
    char ch;
    uint32_t address;

    INPUTMUX_Init(INPUTMUX0);
#ifdef USE_BOARD_BRK
    INPUTMUX_AttachSignal(INPUTMUX0, 0, kINPUTMUX_GpioPort0Pin10ToSmartDma);//P0_10/EZH_CAMERA_VSYNC
	INPUTMUX_AttachSignal(INPUTMUX0, 1, kINPUTMUX_GpioPort0Pin11ToSmartDma);//P0_11/EZH_CAMERA_HSYNC
	INPUTMUX_AttachSignal(INPUTMUX0, 2, kINPUTMUX_GpioPort0Pin14ToSmartDma);//P0_14/EZH_CAMERA_PCLK
#else
	INPUTMUX_AttachSignal(INPUTMUX0, 0, kINPUTMUX_GpioPort0Pin4ToSmartDma);//P0_10/EZH_CAMERA_VSYNC
	INPUTMUX_AttachSignal(INPUTMUX0, 1, kINPUTMUX_GpioPort0Pin11ToSmartDma);//P0_11/EZH_CAMERA_HSYNC
	INPUTMUX_AttachSignal(INPUTMUX0, 2, kINPUTMUX_GpioPort0Pin5ToSmartDma);//P0_14/EZH_CAMERA_PCLK
#endif
	/* Turn off clock to inputmux to save power. Clock is only needed to make changes */
	INPUTMUX_Deinit(INPUTMUX0);



    g_camera_complete_flag=0;

	SMARTDMA_InitWithoutFirmware();
	SMARTDMA_InstallFirmware((uint32_t)ezh_code,s_smartdmaCameraFirmware,
						 SMARTDMA_CAMERA_FIRMWARE_SIZE);
	SMARTDMA_InstallCallback(ezh_camera_callback, NULL);
	NVIC_EnableIRQ(SMARTDMA_IRQn);
	NVIC_SetPriority(SMARTDMA_IRQn, 3);

    smartdmaParam.smartdma_stack = (uint32_t*)g_samrtdma_stack;
	smartdmaParam.p_buffer_ping_pong  		 = (uint32_t*)g_camera_buffer;
	smartdmaParam.p_stripe_index  = (uint32_t*)&p_coord_index;
	SMARTDMA_Boot(kEZH_Camera_320240_Div16_Buf, &smartdmaParam, 0x2);


	NVIC_SetPriority(SysTick_IRQn, 1);
	NVIC_SetPriority(SMARTDMA_IRQn, 3);
	NVIC_SetPriority(PLU_IRQn, 7);
	NVIC_EnableIRQ(PLU_IRQn);

}

void ezh_camera_display_callback(void)
{
    uint32_t *buf32;
    uint32_t pixel_count = CAMERA_WIDTH * BUFF_HEIGHT;
    uint32_t word_count = pixel_count / 2; // 2 RGB565 pixels per uint32_t

    // Select the ping/pong half that SMARTDMA just filled
    if ((p_coord_index % 2) == 0) {
        buf32 = (uint32_t *)g_camera_buffer;
    } else {
        buf32 = (uint32_t *)(&g_camera_buffer[2400]); // 2400 x 32-bit words = 4800 pixels
    }

    // In-place conversion: RGB565 -> grayscale RGB565, 2 pixels per 32-bit word
    for (uint32_t i = 0; i < word_count; i++) {
        uint32_t w = buf32[i];

        // Pixel 0
        uint16_t p0 = (uint16_t)(w & 0xFFFFU);
        uint8_t r0 = (uint8_t)((p0 >> 8) & 0xF8U);
        uint8_t g0 = (uint8_t)((p0 >> 3) & 0xFCU);
        uint8_t b0 = (uint8_t)((p0 << 3) & 0xF8U);
        uint8_t y0 = (uint8_t)((r0 * 77U + g0 * 150U + b0 * 29U) >> 8);

        // Convert 8-bit gray back to RGB565 gray
        uint16_t gray0 = (uint16_t)(((y0 & 0xF8U) << 8) |
                                    ((y0 & 0xFCU) << 3) |
                                    ( y0 >> 3));

        // Pixel 1
        uint16_t p1 = (uint16_t)((w >> 16) & 0xFFFFU);
        uint8_t r1 = (uint8_t)((p1 >> 8) & 0xF8U);
        uint8_t g1 = (uint8_t)((p1 >> 3) & 0xFCU);
        uint8_t b1 = (uint8_t)((p1 << 3) & 0xF8U);
        uint8_t y1 = (uint8_t)((r1 * 77U + g1 * 150U + b1 * 29U) >> 8);

        uint16_t gray1 = (uint16_t)(((y1 & 0xF8U) << 8) |
                                    ((y1 & 0xFCU) << 3) |
                                    ( y1 >> 3));

        // Pack 2 grayscale RGB565 pixels back into the same 32-bit word
        buf32[i] = ((uint32_t)gray1 << 16) | gray0;
    }

    FACEDET_DrawResults(
        (uint16_t*)buf32,
        CAMERA_WIDTH,
        (int)(p_coord_index * BUFF_HEIGHT),
        BUFF_HEIGHT
    );

    // Display grayscale image directly from the same main buffer
    display_show_slice(
        p_coord_index,
        (uint32_t)buf32,
        CAMERA_HEIGHT / BUFF_HEIGHT
    );

    /*
     * IMPORTANT:
     * If ezh_copy_slice_to_model_input() expects 8-bit grayscale pixels,
     * you cannot pass buf32 directly anymore as a true 8-bit grayscale buffer.
     * Right now buf32 contains grayscale RGB565 pixels (16 bits/pixel).
     *
     * Only keep this call if your model input function can accept RGB565 grayscale,
     * or if you modify it to extract one gray value per pixel.
     */
    ezh_copy_slice_to_model_input(
        p_coord_index,
        (uint32_t)buf32,
        CAMERA_WIDTH,
        BUFF_HEIGHT,
        CAMERA_HEIGHT / BUFF_HEIGHT
    );

    g_camera_complete_flag = 0;
}

void PLU_DriverIRQHandler(void)
{
	ezh_camera_display_callback();
}
