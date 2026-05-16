#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <emmintrin.h>

#include <pthread.h>

#include <unistd.h>
#include <stddef.h>

#include <tracy/Tracy.hpp>

#include "SSE2.h"

#define NTHREADS 13
#define TRACY_CALLSTACK 64

// ———— Estructuras ————————————————————————
volatile int atomic_ready = 0;

struct thread_data
{
    int BloqueInicio;
    int BloqueFin;
    IplImage *Img1;
    IplImage *Img2;
    IplImage *ImgMosaico;
};

struct thread_data thread_data_array[NTHREADS];

// ———— Funciones ————————————————————————————

void *mosaico_thread(void *ptr)
{
    ZoneScoped;
    // ptr apunta a un entero que indica el mutiplo de la fila a sustituir
    struct thread_data *data = (struct thread_data *)ptr;
    int BloqueInicio = data->BloqueInicio;
    int BloqueFin = data->BloqueFin;
    IplImage *Img1 = data->Img1;
    IplImage *Img2 = data->Img2;
    IplImage *ImgMosaico = data->ImgMosaico;

    std::string threadName = "Worker " + std::to_string(BloqueInicio) + " - " + std::to_string(BloqueFin);
    tracy::SetThreadName(threadName.c_str());

    int sizefil = Img1->width / 16;

    int fila1_columna1, fila2, columna2, bestX, bestY;
    unsigned int bestValue = -1;
    // start
    for (fila1_columna1 = BloqueInicio; fila1_columna1 < BloqueFin; fila1_columna1++)
    {
        int fila1 = (int)((fila1_columna1 / sizefil) * 16); // fila1_columna1 = nº bloque, fila =fila_columna1 /tamaño
        int columna1 = ((int)(fila1_columna1 % sizefil)) * 16;
        char zoneName[32];
        int size = sprintf(zoneName, "Comparaciónes [%d, %d]", columna1, fila1);
        ZoneScopedC(tracy::Color::Yellow2);
        ZoneName(zoneName,size);

        for (fila2 = 0; fila2 < ImgMosaico->height; fila2 += 16)
        {
            for (columna2 = 0; columna2 < ImgMosaico->width; columna2 += 16)
            {
                // para cada bloque de Img2 iterar por todos los bloques de Img1
                unsigned int currval = compararBloqueSSE2(Img1, Img2, columna1, fila1, columna2, fila2);
                // si un bloque de Img2 es mejor comparación que el ultimo de Img1 Guardarlo
                if (currval < bestValue)
                {
                    bestValue = currval;
                    bestX = columna2;
                    bestY = fila2;
                }
            }
        }
        int size2 = sprintf(zoneName, "Copia [%d, %d] a [%d, %d]", bestX, bestY, columna1, fila1);
        ZoneColor(tracy::Color::Orange2);
        ZoneNamedCS(zoneName2,tracy::Color::Orange2, size2, true);
        copiarBloqueSSE2(Img2, ImgMosaico, bestX, bestY, columna1, fila1, 16, 16);
        bestValue = -1;
        // usleep(1000 * 1);
    }

    // __sync_fetch_and_add(&atomic_ready, 1);
    // fin
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Error: Usage %s image_file_name\n", argv[0]);
        return EXIT_FAILURE;
    }
    struct timespec start, finish;
    float elapsed;

    // CV_LOAD_IMAGE_COLOR = 1 forces the resultant IplImage to be colour.
    // CV_LOAD_IMAGE_GRAYSCALE = 0 forces a greyscale IplImage.
    // CV_LOAD_IMAGE_UNCHANGED = -1
    IplImage *Img1 = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);

    // Always check if the program can find the image file
    if (!Img1)
    {
        printf("Error: file %s not found\n", argv[1]);
        return EXIT_FAILURE;
    }
    IplImage *Img2 = cvLoadImage(argv[2], CV_LOAD_IMAGE_COLOR);

    // Always check if the program can find the image file
    if (!Img2)
    {
        printf("Error: file %s not found\n", argv[2]);
        return EXIT_FAILURE;
    }

    cvNamedWindow(argv[1], CV_WINDOW_AUTOSIZE);
    cvShowImage(argv[1], Img1);
    cvNamedWindow(argv[2], CV_WINDOW_AUTOSIZE);
    cvShowImage(argv[2], Img2);
    cvWaitKey(0);

    // Crea la imagen para la componete azul
    IplImage *ImgMosaico = cvCreateImage(cvSize(Img1->width, Img1->height), IPL_DEPTH_8U, 3);

    cvNamedWindow("MosaicoSIMD", CV_WINDOW_AUTOSIZE);

    // ———— THREADS ———————————————————————

    pthread_t threads[NTHREADS];

    clock_gettime(CLOCK_MONOTONIC, &start);
    TracyMessage("Started spawn of threads", strlen("Started spawn of threads"));

    FrameMark;

    int i;
    for (i = 0; i < NTHREADS; i++)
    {
        struct thread_data *thread_arg = &thread_data_array[i];
        thread_arg->BloqueFin = (1 + i) * (int)((Img1->height * Img2->width) / 256) / NTHREADS; // what block do we end at ?
        thread_arg->BloqueInicio = i * (int)((Img1->height * Img2->width) / 256) / NTHREADS;    // what block do we start at ?
        thread_arg->Img1 = Img1;
        thread_arg->Img2 = Img2;
        thread_arg->ImgMosaico = ImgMosaico;
        pthread_create(&threads[i], NULL, &mosaico_thread, (void *)thread_arg);
    }
    // int is_running = 1;
    // while (is_running) {
    //     cvShowImage("MosaicoSIMD", ImgMosaico);
    //     cvWaitKey(30);
    //     if (atomic_ready >= NTHREADS) {
    //         is_running = 0;
    //     }
    // }

    for (i = 0; i < NTHREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("\nTime elapsed = %f\n", elapsed);
    TracyMessage("Collapsed all threads", strlen("Collapsed all threads"));

    cvShowImage("MosaicoSIMD", ImgMosaico);
    
    // memory release for images before exiting the application
    cvReleaseImage(&Img1);
    cvReleaseImage(&Img2);
    cvReleaseImage(&ImgMosaico);

    // Self-explanatory
    cvDestroyWindow(argv[1]);
    cvDestroyWindow(argv[2]);
    cvDestroyWindow("MosaicoSIMD");
    return EXIT_SUCCESS;
}
