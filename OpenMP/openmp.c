/**
 * Implementation of Parent1 algorithm with OpenMP parallelization
 * This is the first step of the PDC project implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

// Forward declarations (prototypes)
int* Swap(int* v, int x, int n);
int* FindPosition(int* v, int t, int n);
int* Parent1(int* v, int t, int n);
void displayVertex(int* v, int n);
void processVerticesParallel(int** vertices, int** results, int num_vertices, int t, int n);
void processVerticesSequential(int** vertices, int** results, int num_vertices, int t, int n);
int** generateRandomVertices(int n, int num_vertices);
void freeVertices(int** vertices, int num_vertices);

/**
 * Function to swap vertices and find adjacent vertex
 *
 * @param v The vertex represented as array of integers
 * @param x The symbol to swap in the vertex
 * @param n The dimension of Bn
 * @return The vertex adjacent to v in Bn
 */
int* Swap(int* v, int x, int n) {
    // Find the position of x in v
    int i = 0;
    while (i < n && v[i] != x) {
        i++;
    }
   
    // Swap with the next position (cyclically)
    int* p = malloc(n * sizeof(int));
    for (int j = 0; j < n; j++) {
        p[j] = v[j];
    }
   
    // i = π^-1(x), p = η(i)
    if (i < n - 1) {
        // Swap with next position
        int temp = p[i];
        p[i] = p[i + 1];
        p[i + 1] = temp;
    } else {
        // Swap with first position (circular)
        int temp = p[i];
        p[i] = p[0];
        p[0] = temp;
    }
   
    return p;
}

/**
 * Function to find the position of vertex v
 *
 * @param v The vertex represented as array of integers
 * @param t The t-th tree in IST
 * @param n The dimension of Bn
 * @return The vertex adjacent to v in Bn
 */
int* FindPosition(int* v, int t, int n) {
    if (t == 2 && Swap(v, t - 1, n)[n-1] == 1) {
        return Swap(v, t - 1, n);
    } else if (v[n-1] >= t && v[n-1] <= n - 1) {
        int j = v[n-1];
        return Swap(v, j, n);
    } else {
        return Swap(v, t, n);
    }
}

/**
 * Main Parent1 function to find parent of vertex v in tree t
 *
 * @param v The vertex represented as array of integers
 * @param t The t-th tree in IST
 * @param n The dimension of Bn
 * @return The parent of v in Tt
 */
int* Parent1(int* v, int t, int n) {
    int* p;
   
    if (t != n - 1) {
        p = FindPosition(v, t, n);
    } else if (v[n-1] == n) {
        p = Swap(v, n-1, n);
    } else if (v[n-1] == n - 1 && v[n-2] == n && Swap(v, n, n)[n-1] != 1) {
        if (t == 1) {
            p = Swap(v, n, n);
        } else {
            p = Swap(v, t - 1, n);
        }
    } else if (v[n-1] == t) {
        p = Swap(v, n, n);
    } else {
        p = Swap(v, t, n);
    }
   
    return p;
}

/**
 * Function to display a vertex
 *
 * @param v The vertex to display
 * @param n The dimension
 */
void displayVertex(int* v, int n) {
    printf("(");
    for (int i = 0; i < n; i++) {
        printf("%d", v[i]);
        if (i < n - 1) printf(", ");
    }
    printf(")\n");
}

/**
 * Parallel implementation to process multiple vertices using OpenMP
 *
 * @param vertices Array of vertices
 * @param results Array to store parent results
 * @param num_vertices Number of vertices to process
 * @param t The t-th tree in IST
 * @param n The dimension of Bn
 */
void processVerticesParallel(int** vertices, int** results, int num_vertices, int t, int n) {
    // OpenMP parallel region
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_vertices; i++) {
            // Explicit cast to avoid warnings
            results[i] = (int*)Parent1(vertices[i], t, n);
        }
    }
}

/**
 * Sequential implementation to process multiple vertices (for comparison)
 */
void processVerticesSequential(int** vertices, int** results, int num_vertices, int t, int n) {
    for (int i = 0; i < num_vertices; i++) {
        // Explicit cast to avoid warnings
        results[i] = (int*)Parent1(vertices[i], t, n);
    }
}

/**
 * Generate random test vertices
 *
 * @param n The dimension of Bn
 * @param num_vertices Number of vertices to generate
 * @return Array of random vertices
 */
int** generateRandomVertices(int n, int num_vertices) {
    int** vertices = (int**)malloc(num_vertices * sizeof(int*));
   
    for (int i = 0; i < num_vertices; i++) {
        vertices[i] = (int*)malloc(n * sizeof(int));
       
        // Generate a random permutation of 1 to n
        int* used = (int*)calloc(n + 1, sizeof(int));
        for (int j = 0; j < n; j++) {
            int val;
            do {
                val = 1 + rand() % n;
            } while (used[val]);
            used[val] = 1;
            vertices[i][j] = val;
        }
        free(used);
    }
   
    return vertices;
}

/**
 * Free memory for array of vertices
 */
void freeVertices(int** vertices, int num_vertices) {
    for (int i = 0; i < num_vertices; i++) {
        free(vertices[i]);
    }
    free(vertices);
}

/**
 * Main function with benchmarking for sequential and OpenMP versions
 */
int main(int argc, char* argv[]) {
    // Get command line parameters or use defaults
    int n = (argc > 1) ? atoi(argv[1]) : 8;          // Dimension of Bn
    int t = (argc > 2) ? atoi(argv[2]) : 2;          // The t-th tree in IST
    int num_vertices = (argc > 3) ? atoi(argv[3]) : 1000; // Number of vertices to process
    int num_threads = (argc > 4) ? atoi(argv[4]) : 4;     // Number of OpenMP threads
   
    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);
   
    // Seed random number generator
    srand(time(NULL));
   
    // Generate random test vertices
    printf("Generating %d random vertices for dimension %d...\n", num_vertices, n);
    int** vertices = generateRandomVertices(n, num_vertices);
   
    // Allocate result arrays
    int** results_seq = (int**)malloc(num_vertices * sizeof(int*));
    int** results_omp = (int**)malloc(num_vertices * sizeof(int*));
   
    // Sequential processing
    printf("Running sequential processing...\n");
    double start_time = omp_get_wtime();
    processVerticesSequential(vertices, results_seq, num_vertices, t, n);
    double seq_time = omp_get_wtime() - start_time;
   
    // OpenMP parallel processing
    printf("Running parallel processing with %d threads...\n", num_threads);
    start_time = omp_get_wtime();
    processVerticesParallel(vertices, results_omp, num_vertices, t, n);
    double omp_time = omp_get_wtime() - start_time;
   
    // Print results
    printf("\n----- Performance Results -----\n");
    printf("Dimension (n): %d\n", n);
    printf("Tree (t): %d\n", t);
    printf("Number of vertices: %d\n", num_vertices);
    printf("Number of threads: %d\n", num_threads);
    printf("Sequential execution time: %.6f seconds\n", seq_time);
    printf("OpenMP execution time: %.6f seconds\n", omp_time);
    printf("Speedup: %.2f\n", seq_time / omp_time);
    printf("Efficiency: %.2f%%\n", (seq_time / omp_time / num_threads) * 100);
   
    // Validate results (optional, can be disabled for large datasets)
    if (num_vertices <= 100) {  // Only validate for smaller datasets
        printf("\nValidating results...\n");
        int mismatch = 0;
        for (int i = 0; i < num_vertices; i++) {
            for (int j = 0; j < n; j++) {
                if (results_seq[i][j] != results_omp[i][j]) {
                    mismatch++;
                    break;
                }
            }
        }
        if (mismatch > 0) {
            printf("WARNING: %d mismatches found between sequential and parallel results!\n", mismatch);
        } else {
            printf("All results match between sequential and parallel execution.\n");
        }
    }
   
    // Clean up
    freeVertices(vertices, num_vertices);
    freeVertices(results_seq, num_vertices);
    freeVertices(results_omp, num_vertices);
   
    return 0;
}
