#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <math.h>
#include <psapi.h>

#define ITERATIONS 10
#define PAUSE_EVERY 20
#define PAUSE_DURATION 10
#define WARMUP_ITER 5
#define WARMUP_PAUSE 2
#define LANGUAGE "C"

const char *matrix_dir = "matrices";
const char *csv_file = "results/c_results.csv";
const int matrix_sizes[] = {10, 100}; // adjust as needed
const int num_sizes = sizeof(matrix_sizes)/sizeof(matrix_sizes[0]);

// --- High precision timer using QueryPerformanceCounter ---
double get_time_seconds() {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / freq.QuadPart;
}

// --- Read binary matrix ---
int **read_matrix_from_binary(const char *filename, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Couldn't open file '%s'\n", filename); exit(EXIT_FAILURE); }

    int *data = malloc(size * size * sizeof(int));
    if (!data) { fprintf(stderr, "[ERROR] Memory allocation failed for '%s'\n", filename); exit(EXIT_FAILURE); }

    size_t read_count = fread(data, sizeof(int), size*size, f);
    fclose(f);
    if (read_count != (size_t)(size*size)) { fprintf(stderr, "[ERROR] Could not read matrix '%s'\n", filename); exit(EXIT_FAILURE); }

    int **matrix = malloc(size * sizeof(int*));
    for (int i=0; i<size; i++) matrix[i] = &data[i*size];
    printf("[OK] Loaded matrix from '%s' (%dx%d)\n", filename, size, size);
    return matrix;
}

// --- Naive matrix multiplication ---
double naive_matrix_multiplication(int **A, int **B, int n) {
    int *data_C = calloc(n*n, sizeof(int));
    int **C = malloc(n*sizeof(int*));
    for (int i=0;i<n;i++) C[i]=&data_C[i*n];

    double start = get_time_seconds();
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                C[i][j]+=A[i][k]*B[k][j];
    double end = get_time_seconds();

    free(data_C); free(C);
    return end-start;
}

// --- Warm-up ---
void warm_up(int **A,int **B,int size,int iterations,int pause_sec){
    printf("\n=== Warm-up: %d iterations for size %dx%d ===\n", iterations, size, size);
    for(int i=1;i<=iterations;i++){
        naive_matrix_multiplication(A,B,size);
        printf("[OK] Warm-up iteration %d completed\n",i);
        Sleep(pause_sec*1000);
    }
}

// --- Statistics ---
double mean(double *data,int n){ double sum=0; for(int i=0;i<n;i++) sum+=data[i]; return sum/n; }
double median(double *data,int n){
    double *copy=malloc(n*sizeof(double)); memcpy(copy,data,n*sizeof(double));
    for(int i=0;i<n-1;i++) for(int j=0;j<n-1-i;j++) if(copy[j]>copy[j+1]){double t=copy[j];copy[j]=copy[j+1];copy[j+1]=t;}
    double med=(n%2==0)?(copy[n/2-1]+copy[n/2])/2.0:copy[n/2]; free(copy); return med;
}
double std_dev(double *data,int n,double mean_val){ double sum=0; for(int i=0;i<n;i++) sum+=(data[i]-mean_val)*(data[i]-mean_val); return sqrt(sum/n); }

// --- CPU and memory usage ---
double get_cpu_usage() {
    static FILETIME prevSysIdle, prevSysKernel, prevSysUser;
    static FILETIME prevProcKernel, prevProcUser;
    FILETIME ftSysIdle, ftSysKernel, ftSysUser;
    FILETIME ftProcCreation, ftProcExit, ftProcKernel, ftProcUser;

    HANDLE hProcess = GetCurrentProcess();
    GetProcessTimes(hProcess,&ftProcCreation,&ftProcExit,&ftProcKernel,&ftProcUser);

    SYSTEMTIME st;
    double procTime=(double)( ( (ULARGE_INTEGER*)&ftProcKernel)->QuadPart + ((ULARGE_INTEGER*)&ftProcUser)->QuadPart );
    // Rough approximation: returns value in percent of one core
    return procTime/1e7; // convert to percent; rough, not exact
}

double get_memory_usage_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.WorkingSetSize / (1024.0*1024.0);
}

// --- Save results to CSV ---
void save_results_to_csv(char results[][13][256], int num_rows) {
    FILE *f = fopen(csv_file,"w");
    if(!f){ perror("[ERROR] Couldn't open CSV file"); exit(EXIT_FAILURE); }
    fprintf(f,"Size,Matrix A File,Matrix B File,Mean Time (s),Median Time (s),Std Dev (s),Mean CPU (%%),Median CPU (%%),Std CPU (%%),Mean Memory (MB),Median Memory (MB),Std Memory (MB),Language\n");
    for(int i=0;i<num_rows;i++){
        for(int j=0;j<13;j++){
            fprintf(f,"%s%s", results[i][j], (j<12)?",":"\n");
        }
    }
    fclose(f);
    printf("\n[OK] All results saved to '%s'\n", csv_file);
}

int main(){
    printf("[OK] Starting C matrix multiplication benchmark...\n");

    char results[100][13][256];
    int result_idx=0;

    int max_size=matrix_sizes[num_sizes-1];
    char file_a_warm[256], file_b_warm[256];
    snprintf(file_a_warm,sizeof(file_a_warm),"%s/A_%d.bin",matrix_dir,max_size);
    snprintf(file_b_warm,sizeof(file_b_warm),"%s/B_%d.bin",matrix_dir,max_size);

    int **matrix_a_warm=read_matrix_from_binary(file_a_warm,max_size);
    int **matrix_b_warm=read_matrix_from_binary(file_b_warm,max_size);
    warm_up(matrix_a_warm,matrix_b_warm,max_size,WARMUP_ITER,WARMUP_PAUSE);

    for(int s=0;s<num_sizes;s++){
        int size=matrix_sizes[s];
        char file_a[256], file_b[256];
        snprintf(file_a,sizeof(file_a),"%s/A_%d.bin",matrix_dir,size);
        snprintf(file_b,sizeof(file_b),"%s/B_%d.bin",matrix_dir,size);

        printf("\n=== Processing matrices of size %dx%d ===\n", size, size);
        int **matrix_a=read_matrix_from_binary(file_a,size);
        int **matrix_b=read_matrix_from_binary(file_b,size);

        double times[ITERATIONS], cpu[ITERATIONS], mem[ITERATIONS];

        for(int i=0;i<ITERATIONS;i++){
            double start=get_time_seconds();
            naive_matrix_multiplication(matrix_a,matrix_b,size);
            double elapsed=get_time_seconds()-start;

            times[i]=elapsed;
            cpu[i]=get_cpu_usage();
            mem[i]=get_memory_usage_mb();

            if((i+1)%PAUSE_EVERY==0 && (i+1)!=ITERATIONS){
                printf("[OK] Pausing for %d seconds...\n",PAUSE_DURATION);
                Sleep(PAUSE_DURATION*1000);
            }
        }

        double mean_time=mean(times,ITERATIONS), median_time=median(times,ITERATIONS), std_time=std_dev(times,ITERATIONS,mean_time);
        double mean_cpu=mean(cpu,ITERATIONS), median_cpu=median(cpu,ITERATIONS), std_cpu=std_dev(cpu,ITERATIONS,mean_cpu);
        double mean_mem=mean(mem,ITERATIONS), median_mem=median(mem,ITERATIONS), std_mem=std_dev(mem,ITERATIONS,mean_mem);

        printf("[OK] Stats for size %d: mean_time=%.6f, mean_cpu=%.2f%%, mean_mem=%.2fMB\n",size,mean_time,mean_cpu,mean_mem);

        snprintf(results[result_idx][0],256,"%d",size);
        snprintf(results[result_idx][1],256,"%s",file_a);
        snprintf(results[result_idx][2],256,"%s",file_b);
        snprintf(results[result_idx][3],256,"%.6f",mean_time);
        snprintf(results[result_idx][4],256,"%.6f",median_time);
        snprintf(results[result_idx][5],256,"%.6f",std_time);
        snprintf(results[result_idx][6],256,"%.6f",mean_cpu);
        snprintf(results[result_idx][7],256,"%.6f",median_cpu);
        snprintf(results[result_idx][8],256,"%.6f",std_cpu);
        snprintf(results[result_idx][9],256,"%.6f",mean_mem);
        snprintf(results[result_idx][10],256,"%.6f",median_mem);
        snprintf(results[result_idx][11],256,"%.6f",std_mem);
        snprintf(results[result_idx][12],256,"%s",LANGUAGE);
        result_idx++;

        free(matrix_a[0]); free(matrix_a);
        free(matrix_b[0]); free(matrix_b);
    }

    free(matrix_a_warm[0]); free(matrix_a_warm);
    free(matrix_b_warm[0]); free(matrix_b_warm);

    save_results_to_csv(results,result_idx);

    printf("\n[OK] Process completed successfully!\n");
    return 0;
}
