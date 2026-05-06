#ifndef SSE_H
#define SSE_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <emmintrin.h>

void copiarBloqueSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2, int width, int height);

unsigned int compararBloqueSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2);

void copiarBloqueSinSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2, int width, int height);

unsigned int compararBloqueSinSSE2(IplImage *ImgFrom, IplImage *ImgTo, int x1, int y1, int x2, int y2);

#endif