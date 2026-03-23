/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "board_init.h"
#include "demo_config.h"
#include "demo_info.h"
#include "fsl_debug_console.h"
#include "image.h"
#include "image_utils.h"
#include "model.h"
#include "timer.h"
#include "video.h"
#include "ov7670.h"
#include "fsl_lpi2c.h"
#include "MCXN947_cm33_core0.h"
//#include "fsl_pint.h"
#include "pin_mux.h"

#define CAMERA_I2C LPI2C7

static uint8_t brightness =0x2f;
static status_t OV7670_WriteReg_Direct(uint8_t reg, uint8_t value)
{
    lpi2c_master_transfer_t xfer;
    uint8_t data[2] = {reg, value};

    xfer.slaveAddress   = 0x21;
    xfer.direction      = (lpi2c_direction_t)0;
    xfer.subaddress     = 0;
    xfer.subaddressSize = 0;
    xfer.data           = data;
    xfer.dataSize       = 2;
    xfer.flags          = kLPI2C_TransferDefaultFlag;

    return LPI2C_MasterTransferBlocking(LPI2C7, &xfer);
}

//void PINT0_IRQHandler(void)
//{
//	if(PINT_PinInterruptGetStatus(PINT, kPINT_PinInt0)){
//		brightness += 10;
//		OV7670_WriteReg_Direct(0x55U,brightness);
//		PINT_PinInterruptClrStatus(PINT, kPINT_PinInt0);//sw3
//	}
//	else if(PINT_PinInterruptGetStatus(PINT, kPINT_PinInt1)){
//		brightness -= 10;
//		OV7670_WriteReg_Direct(0x55U,brightness);
//		PINT_PinInterruptClrStatus(PINT, kPINT_PinInt1);//sw2
//	}
//}
int main(void)
{
    BOARD_Init();
    TIMER_Init();
//    PINT_Init(PINT);
//    PINT_PinInterruptConfig(PINT, kPINT_PinInt0, kPINT_PinIntEnableFallEdge);
//    PINT_PinInterruptConfig(PINT, kPINT_PinInt1, kPINT_PinIntEnableFallEdge);
//    EnableIRQ(PINT0_IRQn);


    DEMO_PrintInfo();


    Ov7670_Init();

    display_init();

#if SERVO_ENABLE
    servo_motor_init();
#endif
    ezh_start();

    face_det();

    while(1)
    {

    }
}
