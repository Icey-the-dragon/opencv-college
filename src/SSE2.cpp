#include <stdio.h>
#include <stdlib.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <emmintrin.h>

// include profiler
#include <tracy/Tracy.hpp>

#include "SSE2.h"

int abs(int a)
{
    return a < 0 ? -a : a;
}

void copiarBloqueSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2, int width, int height)
{
    ZoneScoped;
    ZoneColor(tracy::Color::AliceBlue);
    
    int totalBytesPerRow = width * 3;

    for (int fila = 0; fila < height; fila++)
    {
        __m128i *src = (__m128i *)(ImgFrom->imageData + (y1 + fila) * ImgFrom->widthStep + x1 * 3);
        __m128i *dst = (__m128i *)(ImgTo->imageData + (y2 + fila) * ImgTo->widthStep + x2 * 3);

        int i = 0;
        // Copy 16 bytes at a time
        for (; i <= totalBytesPerRow - 16; i += 16)
        {
            *dst++ = *src++;
        }

        // Handle the remaining bytes (if totalBytesPerRow is not a multiple of 16)
        for (; i < totalBytesPerRow - 16; i++)
        {
            *((char *)(dst) + i) = *((char *)(src) + i);
        }
    }
}

unsigned int compararBloqueSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2)
{
    ZoneScoped;
    int fila, cc;
    __m128i registro = _mm_setzero_si128();

    for (fila = 0; fila < 16; fila++)
    {
        // Calculate the starting byte address for this row
        __m128i *p1 = (__m128i *)(ImgFrom->imageData + (y1 + fila) * ImgFrom->widthStep + x1 * 3);
        __m128i *p2 = (__m128i *)(ImgTo->imageData + (y2 + fila) * ImgTo->widthStep + x2 * 3);

        for (cc = 0; cc < 3; cc++)
        {
            // Accumulate SAD
            registro = _mm_add_epi64(registro, _mm_sad_epu8(*p1++, *p2++));
        }
    }

    // Correct way to sum the two 64-bit lanes into a 32-bit int
    unsigned long long resultados[2];
    _mm_storeu_si128((__m128i *)resultados, registro);

    return (unsigned int)(resultados[0] + resultados[1]);
}

void copiarBloqueSinSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2, int width, int height)
{
    for (int fila = 0; fila < height; fila++)
    {
        unsigned char *src = (unsigned char *)(ImgFrom->imageData + (y1 + fila) * ImgFrom->widthStep + x1 * 3);
        unsigned char *dst = (unsigned char *)(ImgTo->imageData + (y2 + fila) * ImgTo->widthStep + x2 * 3);

        // 16 bytes * 3 colors
        for (int i = 0; i < width * 3; i++)
        {
            *dst++ = *src++;
        }
    }
}

unsigned int compararBloqueSinSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2)
{
    int fila, columna;
    unsigned int registro = 0;

    for (fila = 0; fila < 16; fila++)
    {
        // Calculate the starting byte address for this row
        unsigned char *p1 = (unsigned char *)(ImgFrom->imageData + (y1 + fila) * ImgFrom->widthStep + x1 * 3);
        unsigned char *p2 = (unsigned char *)(ImgTo->imageData + (y2 + fila) * ImgTo->widthStep + x2 * 3);

        for (columna = 0; columna < 48; columna++)
        {
            registro += abs(*p1++ - *p2++);
        }
    }

    return registro;
}