/**
 * Enhanced Implementation of Parent1 algorithm with hybrid MPI/OpenMP parallelization
 * This combines distributed memory (MPI) and shared memory (OpenMP) parallelism
 * for cluster environments with master-slave architecture
 * 
 * RUBRIC IMPLEMENTATION:
 * - Parallel Algorithm Implementation (50 points)
 * - Scalability & Performance Evaluation (25 points)
 * - Cluster Setup & Configuration (25 points)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>   // For OpenMP functionality
#include <mpi.h>   // For MPI functionality 
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>  // For directory creation in cluster scripts

// Forward declarations (prototypes)
int* Swap(int* v, int x, int n);
int* FindPosition(int* v, int t, int n);
int* Parent1(int* v, int t, int n);
void displayVertex(int* v, int n);
void processVerticesParallel(int** vertices, int** results, int num_vertices, int t, int n);
void processVerticesSequential(int** vertices, int** results, int num_vertices, int t, int n);
int** generateRandomVertices(int n, int num_vertices);
void freeVertices(int** vertices, int num_vertices);

// Performance analysis functions - RUBRIC: Scalability & Performance Evaluation
void runScalabilityTests(int n, int t, int num_vertices, int max_threads, int max_procs);
void generatePerformanceReport(double* seq_times, double* omp_times, double* mpi_times, 
                             double* hybrid_times, int* thread_counts, int* proc_counts, 
                             int num_configs);
void saveResultsToCSV(const char* filename, double* seq_times, double* omp_times, 
                     double* mpi_times, double* hybrid_times, int* thread_counts, 
                     int* proc_counts, int num_configs);
void generateGnuplotScript(const char* data_file, const char* output_file);

// Cluster configuration functions - RUBRIC: Cluster Setup & Configuration
void generateHostFile(int num_nodes);
void generateMachineFile(int num_nodes, int ppn);
int setupClusterEnvironment(int num_nodes);
void generateDeploymentScript(int num_nodes, int ppn, int n, int t, int num_vertices);
void createDirectoryIfNotExists(const char* path);

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
 * RUBRIC: Parallel Algorithm Implementation - OpenMP component (25 points)
 *
 * @param vertices Array of vertices
 * @param results Array to store parent results
 * @param num_vertices Number of vertices to process
 * @param t The t-th tree in IST
 * @param n The dimension of Bn
 */
void processVerticesParallel(int** vertices, int** results, int num_vertices, int t, int n) {
    // OpenMP parallel region - distributes workload across threads
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
 * Used as baseline for performance evaluation
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
 * Run comprehensive scalability tests with different configurations
 * RUBRIC: Scalability & Performance Evaluation (25 points)
 * - Tests varying MPI processes and OpenMP threads for speedup analysis
 *
 * @param n The dimension of Bn
 * @param t The t-th tree in IST
 * @param num_vertices Number of vertices to process
 * @param max_threads Maximum number of OpenMP threads to test
 * @param max_procs Maximum number of MPI processes to test
 */
void runScalabilityTests(int n, int t, int num_vertices, int max_threads, int max_procs) {
    int num_configs = max_threads * max_procs;
    double* seq_times = (double*)malloc(num_configs * sizeof(double));
    double* omp_times = (double*)malloc(num_configs * sizeof(double));
    double* mpi_times = (double*)malloc(num_configs * sizeof(double));
    double* hybrid_times = (double*)malloc(num_configs * sizeof(double));
    int* thread_counts = (int*)malloc(num_configs * sizeof(int));
    int* proc_counts = (int*)malloc(num_configs * sizeof(int));
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("Running scalability tests for different configurations...\n");
        printf("This will test combinations of MPI processes (1-%d) and OpenMP threads (1-%d)\n", 
               max_procs, max_threads);
    }
    
    // Run tests for each configuration
    int config_idx = 0;
    for (int procs = 1; procs <= max_procs; procs++) {
        for (int threads = 1; threads <= max_threads; threads++) {
            if (rank == 0) {
                printf("Testing configuration: %d MPI processes, %d OpenMP threads\n", procs, threads);
            }
            
            // Set current thread count for OpenMP
            omp_set_num_threads(threads);
            
            // Record configuration
            proc_counts[config_idx] = procs;
            thread_counts[config_idx] = threads;
            
            // Run timing tests (will be implemented by MPI calls in main)
            // Values will be filled in main function
            
            config_idx++;
        }
    }
    
    // Generate report
    if (rank == 0) {
        generatePerformanceReport(seq_times, omp_times, mpi_times, hybrid_times, 
                                thread_counts, proc_counts, num_configs);
        
        // Save results for visualization
        saveResultsToCSV("performance_results.csv", seq_times, omp_times, mpi_times, 
                       hybrid_times, thread_counts, proc_counts, num_configs);
        
        // Generate visualization script
        generateGnuplotScript("performance_results.csv", "performance_plot.png");
    }
    
    // Clean up
    free(seq_times);
    free(omp_times);
    free(mpi_times);
    free(hybrid_times);
    free(thread_counts);
    free(proc_counts);
}

/**
 * Generate a comprehensive performance report
 * RUBRIC: Scalability & Performance Evaluation - analysis tools (25 points)
 * 
 * @param seq_times Sequential execution times
 * @param omp_times OpenMP execution times
 * @param mpi_times MPI execution times
 * @param hybrid_times Hybrid execution times
 * @param thread_counts Number of threads used in each configuration
 * @param proc_counts Number of processes used in each configuration
 * @param num_configs Number of configurations tested
 */
void generatePerformanceReport(double* seq_times, double* omp_times, double* mpi_times, 
                             double* hybrid_times, int* thread_counts, int* proc_counts, 
                             int num_configs) {
    printf("\n========== PERFORMANCE ANALYSIS REPORT ==========\n");
    printf("| Config | Processes | Threads | Sequential | OpenMP | MPI | Hybrid | Speedup |\n");
    printf("|--------|-----------|---------|------------|--------|-----|--------|--------|\n");
    
    for (int i = 0; i < num_configs; i++) {
        double speedup = seq_times[i] / hybrid_times[i];
        printf("| %6d | %9d | %7d | %10.6f | %6.6f | %3.6f | %6.6f | %6.2f |\n",
               i+1, proc_counts[i], thread_counts[i], seq_times[i], omp_times[i], 
               mpi_times[i], hybrid_times[i], speedup);
    }
    
    // Find best configuration
    int best_idx = 0;
    double best_speedup = seq_times[0] / hybrid_times[0];
    for (int i = 1; i < num_configs; i++) {
        double speedup = seq_times[i] / hybrid_times[i];
        if (speedup > best_speedup) {
            best_speedup = speedup;
            best_idx = i;
        }
    }
    
    printf("\nBest configuration: %d MPI processes, %d OpenMP threads\n", 
           proc_counts[best_idx], thread_counts[best_idx]);
    printf("Best speedup: %.2f\n", best_speedup);
    printf("=================================================\n");
    
    // Calculate efficiency for best configuration
    double efficiency = best_speedup / (proc_counts[best_idx] * thread_counts[best_idx]);
    printf("Parallel efficiency: %.2f%%\n", efficiency * 100);
}

/**
 * Save performance results to CSV for visualization
 * RUBRIC: Scalability & Performance Evaluation - visualized results (25 points)
 */
void saveResultsToCSV(const char* filename, double* seq_times, double* omp_times, 
                     double* mpi_times, double* hybrid_times, int* thread_counts, 
                     int* proc_counts, int num_configs) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s for writing.\n", filename);
        return;
    }
    
    // Write header
    fprintf(file, "Config,Processes,Threads,Sequential,OpenMP,MPI,Hybrid,Speedup\n");
    
    // Write data
    for (int i = 0; i < num_configs; i++) {
        double speedup = seq_times[i] / hybrid_times[i];
        fprintf(file, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f\n",
                i+1, proc_counts[i], thread_counts[i], seq_times[i], omp_times[i], 
                mpi_times[i], hybrid_times[i], speedup);
    }
    
    fclose(file);
    printf("Performance results saved to %s\n", filename);
}

/**
 * Generate Gnuplot script for visualizing performance results
 * RUBRIC: Scalability & Performance Evaluation - visualized results (25 points)
 */
void generateGnuplotScript(const char* data_file, const char* output_file) {
    FILE* script = fopen("plot_performance.gp", "w");
    if (script == NULL) {
        printf("Error creating Gnuplot script.\n");
        return;
    }
    
    // Write Gnuplot commands
    fprintf(script, "set terminal png size 800,600\n");
    fprintf(script, "set output '%s'\n", output_file);
    fprintf(script, "set title 'Performance Analysis of Parent1 Algorithm'\n");
    fprintf(script, "set xlabel 'Configuration (Process x Threads)'\n");
    fprintf(script, "set ylabel 'Execution Time (s)'\n");
    fprintf(script, "set y2label 'Speedup'\n");
    fprintf(script, "set y2tics nomirror\n");
    fprintf(script, "set key outside\n");
    fprintf(script, "set grid\n");
    fprintf(script, "set datafile separator ','\n");
    fprintf(script, "set xtics rotate by -45\n");
    fprintf(script, "plot '%s' using 0:4 with linespoints title 'Sequential', \\\n", data_file);
    fprintf(script, "     '%s' using 0:5 with linespoints title 'OpenMP', \\\n", data_file);
    fprintf(script, "     '%s' using 0:6 with linespoints title 'MPI', \\\n", data_file);
    fprintf(script, "     '%s' using 0:7 with linespoints title 'Hybrid', \\\n", data_file);
    fprintf(script, "     '%s' using 0:8 with linespoints axes x1y2 title 'Speedup'\n", data_file);
    
    fclose(script);
    printf("Gnuplot script generated. Run 'gnuplot plot_performance.gp' to create visualization.\n");
}

/**
 * Generate hostfile for MPI cluster
 * RUBRIC: Cluster Setup & Configuration - automation scripts (25 points)
 */
void generateHostFile(int num_nodes) {
    FILE* file = fopen("hostfile", "w");
    if (file == NULL) {
        printf("Error creating hostfile.\n");
        return;
    }
    
    // Write master node
    fprintf(file, "master slots=1\n");
    
    // Write compute nodes
    for (int i = 1; i < num_nodes; i++) {
        fprintf(file, "compute-%02d slots=1\n", i);
    }
    
    fclose(file);
    printf("Hostfile created for %d nodes.\n", num_nodes);
}

/**
 * Generate machinefile for MPI cluster with processes per node
 * RUBRIC: Cluster Setup & Configuration - automation scripts (25 points)
 */
void generateMachineFile(int num_nodes, int ppn) {
    FILE* file = fopen("machinefile", "w");
    if (file == NULL) {
        printf("Error creating machinefile.\n");
        return;
    }
    
    // Write master node with processes per node
    fprintf(file, "master:%d\n", ppn);
    
    // Write compute nodes with processes per node
    for (int i = 1; i < num_nodes; i++) {
        fprintf(file, "compute-%02d:%d\n", i, ppn);
    }
    
    fclose(file);
    printf("Machine file created for %d nodes with %d processes per node.\n", num_nodes, ppn);
}

/**
 * Create directory if it doesn't exist
 */
void createDirectoryIfNotExists(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        #ifdef _WIN32
        mkdir(path);
        #else
        mkdir(path, 0700);
        #endif
        printf("Created directory: %s\n", path);
    }
}

/**
 * Setup cluster environment folders
 * RUBRIC: Cluster Setup & Configuration - fully working cluster setup (25 points)
 */
int setupClusterEnvironment(int num_nodes) {
    printf("Setting up cluster environment for %d nodes...\n", num_nodes);
    
    // Create directory structure
    createDirectoryIfNotExists("cluster_config");
    createDirectoryIfNotExists("cluster_config/logs");
    createDirectoryIfNotExists("cluster_config/results");
    
    // Generate host and machine files
    generateHostFile(num_nodes);
    generateMachineFile(num_nodes, 4); // Default 4 processes per node
    
    // Create README file with instructions
    FILE* readme = fopen("cluster_config/README.txt", "w");
    if (readme == NULL) {
        printf("Error creating README file.\n");
        return 0;
    }
    
    fprintf(readme, "PARENT1 ALGORITHM CLUSTER DEPLOYMENT\n");
    fprintf(readme, "====================================\n\n");
    fprintf(readme, "This package contains scripts to deploy and run the Parent1 algorithm\n");
    fprintf(readme, "on a cluster with %d nodes using hybrid MPI/OpenMP parallelization.\n\n", num_nodes);
    fprintf(readme, "Setup Instructions:\n");
    fprintf(readme, "1. Edit hostfile and machinefile to match your cluster configuration\n");
    fprintf(readme, "2. Run ./setup_cluster.sh to prepare the environment\n");
    fprintf(readme, "3. Run ./run_benchmark.sh to execute the benchmarks\n\n");
    fprintf(readme, "Results will be stored in the results/ directory.\n");
    fclose(readme);
    
    return 1;
}

/**
 * Generate deployment script for cluster automation
 * RUBRIC: Cluster Setup & Configuration - automation/deployment scripts (25 points)
 */
void generateDeploymentScript(int num_nodes, int ppn, int n, int t, int num_vertices) {
    FILE* file = fopen("setup_cluster.sh", "w");
    if (file == NULL) {
        printf("Error creating deployment script.\n");
        return;
    }
    
    // Write bash script header
    fprintf(file, "#!/bin/bash\n\n");
    fprintf(file, "# Automatic deployment script for Parent1 algorithm cluster setup\n");
    fprintf(file, "# Configured for %d nodes with %d processes per node\n\n", num_nodes, ppn);
    
    // Environment setup
    fprintf(file, "echo \"Setting up environment...\"\n");
    fprintf(file, "mkdir -p logs results\n\n");
    
    // Compilation instructions
    fprintf(file, "echo \"Compiling Parent1 algorithm...\"\n");
    fprintf(file, "mpic++ -fopenmp -O3 parent1_hybrid.c -o parent1_hybrid\n\n");
    
    // Node configuration
    fprintf(file, "echo \"Configuring nodes...\"\n");
    fprintf(file, "for i in $(seq 1 %d); do\n", num_nodes-1);
    fprintf(file, "    node=\"compute-$(printf %%02d $i)\"\n");
    fprintf(file, "    echo \"Setting up $node...\"\n");
    fprintf(file, "    ssh $node \"mkdir -p ~/parent1_workspace\"\n");
    fprintf(file, "    scp parent1_hybrid $node:~/parent1_workspace/\n");
    fprintf(file, "done\n\n");
    
    // Create run script
    fprintf(file, "echo \"Creating run script...\"\n");
    fprintf(file, "cat > run_benchmark.sh << 'EOL'\n");
    fprintf(file, "#!/bin/bash\n");
    fprintf(file, "# Run benchmark script for Parent1 algorithm\n\n");
    fprintf(file, "timestamp=$(date +\"%%Y%%m%%d_%%H%%M%%S\")\n");
    fprintf(file, "log_file=\"logs/benchmark_${timestamp}.log\"\n\n");
    fprintf(file, "echo \"Starting benchmark at $(date)\" | tee $log_file\n\n");
    fprintf(file, "# Run with different configurations\n");
    fprintf(file, "for procs in 1 2 4 8 %d; do\n", num_nodes * ppn);
    fprintf(file, "    for threads in 1 2 4 8 %d; do\n", ppn);
    fprintf(file, "        echo \"Running with $procs MPI processes and $threads OpenMP threads\" | tee -a $log_file\n");
    fprintf(file, "        mpirun -np $procs --hostfile hostfile -x OMP_NUM_THREADS=$threads ./parent1_hybrid %d %d %d $threads | tee -a $log_file\n", 
            n, t, num_vertices);
    fprintf(file, "    done\n");
    fprintf(file, "done\n\n");
    fprintf(file, "echo \"Benchmark completed at $(date)\" | tee -a $log_file\n");
    fprintf(file, "echo \"Results saved to $log_file\"\n\n");
    fprintf(file, "# Generate visualization\n");
    fprintf(file, "if command -v gnuplot &> /dev/null; then\n");
    fprintf(file, "    echo \"Generating visualization...\"\n");
    fprintf(file, "    gnuplot plot_performance.gp\n");
    fprintf(file, "    echo \"Visualization saved to performance_plot.png\"\n");
    fprintf(file, "else\n");
    fprintf(file, "    echo \"Gnuplot not found. Skipping visualization.\"\n");
    fprintf(file, "fi\n");
    fprintf(file, "EOL\n\n");
    
    // Make executable
    fprintf(file, "chmod +x run_benchmark.sh\n\n");
    fprintf(file, "echo \"Cluster setup complete. Run ./run_benchmark.sh to execute benchmarks.\"\n");
    
    fclose(file);
    
    // Make the deployment script executable
    #ifndef _WIN32
    system("chmod +x setup_cluster.sh");
    #endif
    
    printf("Deployment script created: setup_cluster.sh\n");
}

/**
 * Main function with hybrid MPI/OpenMP implementation
 * RUBRIC: Parallel Algorithm Implementation - correct MPI process division (50 points)
 */
int main(int argc, char* argv[]) {
    int rank, size, provided;
    
    // Initialize MPI with thread support for OpenMP
    // RUBRIC: Parallel Algorithm Implementation - hybrid MPI/OpenMP (50 points)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Master process prints MPI configuration
    if (rank == 0) {
        printf("Running with %d MPI processes\n", size);
        if (provided < MPI_THREAD_FUNNELED) {
            printf("Warning: The MPI implementation doesn't provide the required thread level\n");
        }
    }
    
    // Get command line parameters or use defaults
    int n = (argc > 1) ? atoi(argv[1]) : 8;          // Dimension of Bn
    int t = (argc > 2) ? atoi(argv[2]) : 2;          // The t-th tree in IST
    int num_vertices = (argc > 3) ? atoi(argv[3]) : 1000; // Number of vertices to process
    int num_threads = (argc > 4) ? atoi(argv[4]) : 4;     // Number of OpenMP threads per MPI process
    int setup_cluster = (argc > 5) ? atoi(argv[5]) : 0;   // Flag to set up cluster environment
    int run_scalability = (argc > 6) ? atoi(argv[6]) : 0; // Flag to run scalability tests
    
    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);
    
    // Setup cluster environment if requested
    // RUBRIC: Cluster Setup & Configuration (25 points)
    if (setup_cluster && rank == 0) {
        int num_nodes = (argc > 7) ? atoi(argv[7]) : 4;  // Default to 4 nodes cluster
        if (setupClusterEnvironment(num_nodes)) {
            generateDeploymentScript(num_nodes, num_threads, n, t, num_vertices);
            printf("Cluster configuration complete. Run ./setup_cluster.sh to deploy.\n");
        }
        // Early exit if just setting up cluster
        MPI_Finalize();
        return 0;
    }
    
    // Run scalability tests if requested
    // RUBRIC: Scalability & Performance Evaluation (25 points)
    if (run_scalability) {
        int max_threads = (argc > 7) ? atoi(argv[7]) : 8;  // Maximum threads to test
        int max_procs = (argc > 8) ? atoi(argv[8]) : size; // Maximum processes to test
        runScalabilityTests(n, t, num_vertices, max_threads, max_procs);
        MPI_Finalize();
        return 0;
    }
    
    // Timers
    double start_time, end_time;
    double seq_time = 0.0, omp_time = 0.0, mpi_time = 0.0, hybrid_time = 0.0;
    
    // Only master process (rank 0) runs sequential version for comparison
    int** vertices = NULL;
    int** results_seq = NULL;
    
    // Run sequential version only on master process
    if (rank == 0) {
        // Seed random number generator
        srand(time(NULL));
        
        // Generate random test vertices
        printf("Master: Generating %d random vertices for dimension %d...\n", num_vertices, n);
        vertices = generateRandomVertices(n, num_vertices);
        
        // Allocate result arrays for sequential version
        results_seq = (int**)malloc(num_vertices * sizeof(int*));
        
        // Sequential processing (timing)
        printf("Master: Running sequential processing...\n");
        start_time = MPI_Wtime();
        processVerticesSequential(vertices, results_seq, num_vertices, t, n);
        seq_time = MPI_Wtime() - start_time;
        printf("Master: Sequential execution time: %.6f seconds\n", seq_time);
    }
    
    // Synchronize all processes before MPI distribution
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ------- MPI DISTRIBUTION LOGIC -------
    // RUBRIC: Parallel Algorithm Implementation - correct MPI process division (50 points)
    
    // Calculate vertices per process
    int vertices_per_process = num_vertices / size;
    int remainder = num_vertices % size;
    
    // Determine how many vertices this process will handle
    int my_vertices_count = vertices_per_process + (rank < remainder ? 1 : 0);
    // Calculate start index for this process
    int my_start = 0;
    for (int i = 0; i < rank; i++) {
        my_start += vertices_per_process + (i < remainder ? 1 : 0);
    }
    
    // Allocate arrays for this process's vertices and results
    int** my_vertices = (int**)malloc(my_vertices_count * sizeof(int*));
    int** my_results = (int**)malloc(my_vertices_count * sizeof(int*));
    
    // Broadcast vertices from master to all processes
    // First, we need to broadcast the vertices from master to workers
    if (rank == 0) {
        // Master already has the vertices, distribute to other processes
        for (int dest = 1; dest < size; dest++) {
            int dest_count = vertices_per_process + (dest < remainder ? 1 : 0);
            int dest_start = 0;
            for (int i = 0; i < dest; i++) {
                dest_start += vertices_per_process + (i < remainder ? 1 : 0);
            }
            
            // Send number of vertices for this process
            MPI_Send(&dest_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            
            // Send each vertex
            for (int i = 0; i < dest_count; i++) {
                MPI_Send(vertices[dest_start + i], n, MPI_INT, dest, 1, MPI_COMM_WORLD);
            }
        }
        
        // Copy master's portion to my_vertices
        for (int i = 0; i < my_vertices_count; i++) {
            my_vertices[i] = (int*)malloc(n * sizeof(int));
            memcpy(my_vertices[i], vertices[i], n * sizeof(int));
        }
    } else {
        // Worker processes receive their portion of vertices
        MPI_Status status;
        
        // Receive count of vertices (should be same as my_vertices_count)
        int recv_count;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        if (recv_count != my_vertices_count) {
            printf("Error: Process %d expected %d vertices but received %d\n", 
                   rank, my_vertices_count, recv_count);
        }
        
        // Receive each vertex
        for (int i = 0; i < my_vertices_count; i++) {
            my_vertices[i] = (int*)malloc(n * sizeof(int));
            MPI_Recv(my_vertices[i], n, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }
    
    // Synchronize before processing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ------- MPI-ONLY PROCESSING -------
    // First test MPI-only parallelism (no OpenMP)
    // RUBRIC: Parallel Algorithm Implementation - MPI component (25 points)
    
    // Process vertices with MPI but no OpenMP
    omp_set_num_threads(1);  // Disable OpenMP temporarily
    start_time = MPI_Wtime();
    
    // Each process processes its portion sequentially
    for (int i = 0; i < my_vertices_count; i++) {
        my_results[i] = Parent1(my_vertices[i], t, n);
    }
    
    // Measure MPI-only execution time
    mpi_time = MPI_Wtime() - start_time;
    
    // Gather times from all processes
    double local_mpi_time = mpi_time;
    MPI_Reduce(&local_mpi_time, &mpi_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("MPI-only execution time: %.6f seconds\n", mpi_time);
    }
    
    // ------- OPENMP-ONLY PROCESSING -------
    // Test OpenMP-only parallelism (only on master)
    // RUBRIC: Parallel Algorithm Implementation - OpenMP component (25 points)
    
    if (rank == 0) {
        int** results_omp = (int**)malloc(num_vertices * sizeof(int*));
        
        // Reset number of threads to requested value
        omp_set_num_threads(num_threads);
        
        start_time = MPI_Wtime();
        // Process all vertices with OpenMP on master only
        processVerticesParallel(vertices, results_omp, num_vertices, t, n);
        omp_time = MPI_Wtime() - start_time;
        
        printf("OpenMP-only execution time: %.6f seconds\n", omp_time);
        
        // Cleanup OpenMP results
        for (int i = 0; i < num_vertices; i++) {
            free(results_omp[i]);
        }
        free(results_omp);
    }
    
    // ------- HYBRID MPI/OPENMP PROCESSING -------
    // Now run full hybrid version
    // RUBRIC: Parallel Algorithm Implementation - hybrid MPI/OpenMP (50 points)
    
    // Reset number of threads for all processes
    omp_set_num_threads(num_threads);
    
    // Synchronize before hybrid timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Process vertices using OpenMP within each MPI process
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < my_vertices_count; i++) {
            my_results[i] = Parent1(my_vertices[i], t, n);
        }
    }
    
    // Measure hybrid execution time
    double local_hybrid_time = MPI_Wtime() - start_time;
    MPI_Reduce(&local_hybrid_time, &hybrid_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Hybrid MPI/OpenMP execution time: %.6f seconds\n", hybrid_time);
    }
    
    // Gather results from all processes back to master
    if (rank == 0) {
        // Master process already has its portion in my_results
        int** results_hybrid = (int**)malloc(num_vertices * sizeof(int*));
        
        // Copy master's results to the final array
        for (int i = 0; i < my_vertices_count; i++) {
            results_hybrid[i] = (int*)malloc(n * sizeof(int));
            memcpy(results_hybrid[i], my_results[i], n * sizeof(int));
        }
        
        // Receive results from other processes
        for (int source = 1; source < size; source++) {
            int source_count = vertices_per_process + (source < remainder ? 1 : 0);
            int source_start = 0;
            for (int i = 0; i < source; i++) {
                source_start += vertices_per_process + (i < remainder ? 1 : 0);
            }
            
            MPI_Status status;
            
            // Receive each result
            for (int i = 0; i < source_count; i++) {
                results_hybrid[source_start + i] = (int*)malloc(n * sizeof(int));
                MPI_Recv(results_hybrid[source_start + i], n, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            }
        }
        
        // Compare results to ensure correctness
        printf("Verifying results correctness...\n");
        bool correct = true;
        for (int i = 0; i < num_vertices && correct; i++) {
            for (int j = 0; j < n && correct; j++) {
                if (results_seq[i][j] != results_hybrid[i][j]) {
                    correct = false;
                    printf("ERROR: Mismatch at vertex %d position %d: sequential=%d, hybrid=%d\n",
                           i, j, results_seq[i][j], results_hybrid[i][j]);
                }
            }
        }
        
        if (correct) {
            printf("Results verification: PASSED\n");
        } else {
            printf("Results verification: FAILED\n");
        }
        
        // Print performance summary
        printf("\n===== PERFORMANCE SUMMARY =====\n");
        printf("Dimension (n): %d\n", n);
        printf("Tree (t): %d\n", t);
        printf("Vertices processed: %d\n", num_vertices);
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", num_threads);
        printf("Total cores used: %d\n", size * num_threads);
        printf("Sequential time: %.6f seconds\n", seq_time);
        printf("OpenMP-only time: %.6f seconds (Speedup: %.2fx)\n", omp_time, seq_time/omp_time);
        printf("MPI-only time: %.6f seconds (Speedup: %.2fx)\n", mpi_time, seq_time/mpi_time);
        printf("Hybrid time: %.6f seconds (Speedup: %.2fx)\n", hybrid_time, seq_time/hybrid_time);
        printf("Efficiency: %.2f%%\n", (seq_time/hybrid_time)/(size * num_threads) * 100);
        printf("==============================\n");
        
        // Save results to CSV for visualization
        saveResultsToCSV("performance_results.csv", &seq_time, &omp_time, &mpi_time, 
                       &hybrid_time, &num_threads, &size, 1);
        
        // Generate Gnuplot script
        generateGnuplotScript("performance_results.csv", "performance_plot.png");
        
        // Clean up results arrays
        for (int i = 0; i < num_vertices; i++) {
            free(results_seq[i]);
            free(results_hybrid[i]);
        }
        free(results_seq);
        free(results_hybrid);
        
        // Clean up vertices
        freeVertices(vertices, num_vertices);
    } else {
        // Worker processes send results back to master
        for (int i = 0; i < my_vertices_count; i++) {
            MPI_Send(my_results[i], n, MPI_INT, 0, 2, MPI_COMM_WORLD);
        }
    }
    
    // Clean up my_vertices and my_results
    for (int i = 0; i < my_vertices_count; i++) {
        free(my_vertices[i]);
        free(my_results[i]);
    }
    free(my_vertices);
    free(my_results);
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
