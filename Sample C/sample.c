/**
 * Implementation of Parent1 algorithm and supporting functions
 * Based on the provided pseudocode
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

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
 * Example usage of the Parent1 algorithm
 */
int main() {
    int n = 4; // Dimension of B4
    int t = 2; // The 2nd tree in IST
   
    // Example vertex: v = v1v2v3v4 = (2,3,1,4)
    int v[4] = {2, 3, 1, 4};
   
    printf("Original vertex: ");
    displayVertex(v, n);
   
    printf("Finding parent in tree T%d...\n", t);
    int* parent = Parent1(v, t, n);
   
    printf("Parent vertex: ");
    displayVertex(parent, n);
   
    // Clean up
    free(parent);
   
    return 0;
}
