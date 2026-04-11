#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printf("Error: Usage %s image_file_name\n", argv[0]);
        return EXIT_FAILURE;
    }

    // CV_LOAD_IMAGE_COLOR = 1 forces the resultant IplImage to be colour.
    // CV_LOAD_IMAGE_GRAYSCALE = 0 forces a greyscale IplImage.
    // CV_LOAD_IMAGE_UNCHANGED = -1
    IplImage *ImgOrigen = cvLoadImage(argv[1], CV_LOAD_IMAGE_UNCHANGED);

    // Always check if the program can find the image file
    if (!ImgOrigen)
    {
        printf("Error: file %s not found\n", argv[1]);
        return EXIT_FAILURE;
    }

    IplImage *ImgDestino = cvLoadImage(argv[2], CV_LOAD_IMAGE_UNCHANGED);

    // Always check if the program can find the image file
    if (!ImgDestino)
    {
        printf("Error: file %s not found\n", argv[2]);
        return EXIT_FAILURE;
    }

    // a visualization window is created with title: image file name
    cvNamedWindow(argv[1], CV_WINDOW_AUTOSIZE);
    cvShowImage(argv[1], ImgOrigen);
    cvNamedWindow(argv[2], CV_WINDOW_AUTOSIZE);
    cvShowImage(argv[2], ImgDestino);
    cvWaitKey(0);

    // Crea la imagen para la componete azul
    IplImage *ImgBLACK = cvCreateImage(cvSize(ImgOrigen->width, ImgOrigen->height), IPL_DEPTH_8U, 3);
    IplImage *ImgFINAL = cvCreateImage(cvSize(ImgOrigen->width, ImgOrigen->height), IPL_DEPTH_8U, 3);

    int i, fila, cc;
    for (fila = 0; fila < ImgOrigen->height; fila++)
    {
        __m128i *pImgBlack = (__m128i *)(ImgBLACK->imageData + fila * ImgBLACK->widthStep);
        for (cc = 0; cc < ImgOrigen->widthStep; cc = cc + 16)
        {
            *pImgBlack++ = _mm_set1_epi8(0x0);
        }
    }

    cvNamedWindow("Animation", CV_WINDOW_AUTOSIZE);

    __m128i one = _mm_set1_epi8(0x01);
    for (i = 0; i < 255; i++)
    {
        for (fila = 0; fila < ImgOrigen->height; fila++)
        {
            __m128i *pImgOrigen = (__m128i *)(ImgOrigen->imageData + fila * ImgOrigen->widthStep);
            __m128i *pImgDestino = (__m128i *)(ImgDestino->imageData + fila * ImgDestino->widthStep);
            __m128i *pImgBlack = (__m128i *)(ImgBLACK->imageData + fila * ImgBLACK->widthStep);
            __m128i *pImgFinal = (__m128i *)(ImgFINAL->imageData + fila * ImgFINAL->widthStep);
            for (cc = 0; cc < ImgOrigen->widthStep; cc = cc + 16)
            {
                *pImgOrigen = _mm_subs_epu8(*pImgOrigen, one);
                *pImgBlack = _mm_adds_epu8(one, *pImgBlack);
                *pImgFinal = _mm_adds_epu8(*pImgOrigen,_mm_min_epu8(*pImgDestino, *pImgBlack));
                // advance pointers
                pImgBlack++;
                pImgDestino++;
                pImgFinal++;
                pImgOrigen++;
            }
        }
        cvShowImage("Animation", ImgFINAL);
        cvWaitKey(1);
    }

    // crea y muestras las ventanas con las im genes
    cvWaitKey(0);

    // memory release for images before exiting the application
    cvReleaseImage(&ImgOrigen);
    cvReleaseImage(&ImgDestino);
    cvReleaseImage(&ImgBLACK);
    cvReleaseImage(&ImgFINAL);

    // Self-explanatory
    cvDestroyWindow(argv[1]);
    cvDestroyWindow(argv[2]);
    cvDestroyWindow("Animation");

    return EXIT_SUCCESS;
}
