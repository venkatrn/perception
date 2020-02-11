#pragma once

#include "cuda_fp16.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>


#ifndef COMMONCUDA_H
#define COMMONCUDA_H

#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)


__device__ void rgb2lab(uint8_t rr,uint8_t gg, uint8_t bbb, float* lab){
    double r = rr / 255.0;
    double g = gg / 255.0;
    double b = bbb / 255.0;
    double x;
    double y;
    double z;
    r = ((r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : (r / 12.92)) * 100.0;
    g = ((g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : (g / 12.92)) * 100.0;
    b = ((b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : (b / 12.92)) * 100.0;

    x = r*0.4124564 + g*0.3575761 + b*0.1804375;
    y = r*0.2126729 + g*0.7151522 + b*0.0721750;
    z = r*0.0193339 + g*0.1191920 + b*0.9503041;

    x = x / 95.047;
    y = y / 100.00;
    z = z / 108.883;

    x = (x > 0.008856) ? cbrt(x) : (7.787 * x + 16.0 / 116.0);
    y = (y > 0.008856) ? cbrt(y) : (7.787 * y + 16.0 / 116.0);
    z = (z > 0.008856) ? cbrt(z) : (7.787 * z + 16.0 / 116.0);
    float l,a,bb;

    l = (116.0 * y) - 16;
    a = 500 * (x - y);
    bb = 200 * (y - z);

    lab[0] = l;
    lab[1] = a;
    lab[2] = bb;
}
__device__ double color_distance(float l1,float a1,float b1,
                      float l2,float a2,float b2){
    double eps = 1e-5;
    double c1 = sqrtf(SQR(a1) + SQR(b1));
    double c2 = sqrtf(SQR(a2) + SQR(b2));
    double meanC = (c1 + c2) / 2.0;
    double meanC7 = POW7(meanC);

    double g = 0.5*(1 - sqrtf(meanC7 / (meanC7 + 6103515625.))); // 0.5*(1-sqrt(meanC^7/(meanC^7+25^7)))
    double a1p = a1 * (1 + g);
    double a2p = a2 * (1 + g);

    c1 = sqrtf(SQR(a1p) + SQR(b1));
    c2 = sqrtf(SQR(a2p) + SQR(b2));
    double h1 = fmodf(atan2f(b1, a1p) + 2*M_PI, 2*M_PI);
    double h2 = fmodf(atan2f(b2, a2p) + 2*M_PI, 2*M_PI);

    // compute deltaL, deltaC, deltaH
    double deltaL = l2 - l1;
    double deltaC = c2 - c1;
    double deltah;

    if (c1*c2 < eps) {
        deltah = 0;
    }
    if (std::abs(h2 - h1) <= M_PI) {
        deltah = h2 - h1;
    }
    else if (h2 > h1) {
        deltah = h2 - h1 - 2* M_PI;
    }
    else {
        deltah = h2 - h1 + 2 * M_PI;
    }

    double deltaH = 2 * sqrtf(c1*c2)*sinf(deltah / 2);

    // calculate CIEDE2000
    double meanL = (l1 + l2) / 2;
    meanC = (c1 + c2) / 2.0;
    meanC7 = POW7(meanC);
    double meanH;

    if (c1*c2 < eps) {
        meanH = h1 + h2;
    }
    if (std::abs(h1 - h2) <= M_PI + eps) {
        meanH = (h1 + h2) / 2;
    }
    else if (h1 + h2 < 2*M_PI) {
        meanH = (h1 + h2 + 2*M_PI) / 2;
    }
    else {
        meanH = (h1 + h2 - 2*M_PI) / 2;
    }

    double T = 1
        - 0.17*cosf(meanH - DegToRad(30))
        + 0.24*cosf(2 * meanH)
        + 0.32*cosf(3 * meanH + DegToRad(6))
        - 0.2*cosf(4 * meanH - DegToRad(63));
    double sl = 1 + (0.015*SQR(meanL - 50)) / sqrtf(20 + SQR(meanL - 50));
    double sc = 1 + 0.045*meanC;
    double sh = 1 + 0.015*meanC*T;
    double rc = 2 * sqrtf(meanC7 / (meanC7 + 6103515625.));
    double rt = -sinf(DegToRad(60 * expf(-SQR((RadToDeg(meanH) - 275) / 25)))) * rc;

    double cur_dist = sqrtf(SQR(deltaL / sl) + SQR(deltaC / sc) + SQR(deltaH / sh) + rt * deltaC / sc * deltaH / sh);
    return cur_dist;
}

#endif