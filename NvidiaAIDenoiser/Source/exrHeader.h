/**
Copyright (C) <2023>  <Blossom> <Dezeming>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef __EXRHEADER_H__
#define __EXRHEADER_H__

#include <iostream>
#include <string>
#include <vector>

#include <gdt/gdt.h>
#include <gdt/math/vec.h>
#include <vector_types.h>

#include <half.h>

namespace exrLoad {

    inline float clamp(float x, float min, float max) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    inline float f16_to_f32(half __x) {
        unsigned short n = *((unsigned short*)&__x);
        unsigned int x = (unsigned int)n;
        x = x & 0xffff;
        unsigned int sign = x & 0x8000;                   //signed
        unsigned int exponent_f16 = (x & 0x7c00) >> 10;   //half Exponential
        unsigned int mantissa_f16 = x & 0x03ff;           //half Decimal
        unsigned int y = sign << 16;
        unsigned int exponent_f32;                        //float Exponential
        unsigned int mantissa_f32;                        //float Decimal
        unsigned int first_1_pos = 0;                     //The position of the highest digit 1 (half decimal place)
        unsigned int mask;
        unsigned int hx;

        hx = x & 0x7fff;

        if (hx == 0) {
            return *((float*)&y);
        }
        if (hx == 0x7c00) {
            y |= 0x7f800000;
            return *((float*)&y);
        }
        if (hx > 0x7c00) {
            y = 0x7fc00000;
            return *((float*)&y);
        }

        exponent_f32 = 0x70 + exponent_f16;
        mantissa_f32 = mantissa_f16 << 13;

        for (first_1_pos = 0; first_1_pos < 10; first_1_pos++) {
            if ((mantissa_f16 >> (first_1_pos + 1)) == 0) {
                break;
            }
        }

        if (exponent_f16 == 0) {
            mask = (1 << 23) - 1;
            exponent_f32 = exponent_f32 - (10 - first_1_pos) + 1;
            mantissa_f32 = mantissa_f32 << (10 - first_1_pos);
            mantissa_f32 = mantissa_f32 & mask;
        }

        y = y | (exponent_f32 << 23) | mantissa_f32;

        return *((float*)&y);
    }

    inline half f32_to_f16(float __x) {
        unsigned int x = *((unsigned int*)&__x);
        unsigned int sign = x & 0x80000000;
        unsigned int exponent_f32 = (x & 0x7f800000) >> 23;
        unsigned int mantissa_f32 = x & 0x007fffff;
        unsigned short y = (unsigned short)(sign >> 16);
        unsigned int exponent_f16;
        unsigned int mantissa_f16;
        unsigned int hx;

        hx = x & 0x7fffffff;

        if (hx < 0x33800000) {
            return *((half*)&y);
        }
        if (hx > 0x7f800000) {
            y = 0x7e00;
            return *((half*)&y);
        }
        if (hx >= 0x477fffff) {
            y |= 0x7c00;
            return *((half*)&y);
        }

        mantissa_f16 = mantissa_f32 >> 13;

        if (exponent_f32 > 0x70) {
            exponent_f16 = exponent_f32 - 0x70;
        }
        else {
            exponent_f16 = 0;
            mantissa_f16 |= 0x400;
            mantissa_f16 = mantissa_f16 >> (0x71 - exponent_f32);
        }
        y = y | (unsigned short)(exponent_f16 << 10) | (unsigned short)mantissa_f16;
        return *((half*)&y);
    }




}




#endif





