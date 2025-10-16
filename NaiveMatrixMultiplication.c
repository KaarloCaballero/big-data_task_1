#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 10

double a[n][n];
double b[n][n];
double c[n][n];

int main() {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            a[i][j] = (double) rand() / RAND_MAX;
            b[i][j] = (double) rand() / RAND_MAX;
            c[i][j] = 0;
        }

    clock_t start = clock();

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                c[i][j] += a[i][k] * b[k][j];

    clock_t stop = clock();
    double diff = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("%0.6f\n", diff);
}
