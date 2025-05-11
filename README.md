# Parallel Construction of Independent Spanning Trees in Bubble-Sort Networks

## 📚 Overview

This project implements a **non-recursive, parallel algorithm** to construct `n−1` **Independent Spanning Trees (ISTs)** in a **Bubble-Sort Network (Bn)**. It builds upon the work of Kao et al. (2019), who proposed a recursive method that was hard to parallelize. Our approach leverages **OpenMP** and **MPI** to achieve efficient, scalable computation.

## 🧠 Key Concepts

- **Bubble-Sort Network (Bn):** A graph where each node is a permutation of numbers from `1` to `n`, and edges represent adjacent swaps.
- **Independent Spanning Trees (ISTs):** Multiple spanning trees in a graph where paths from any node to the root are disjoint, improving fault tolerance.
- **Graph Partitioning:** We use **METIS** to divide the Bn graph for parallel processing.

## 🚀 Features

- Construct `n−1` ISTs in **O(n·n!)** time (asymptotically optimal).
- Each vertex determines its parent in **constant time**.
- Fault-tolerant and secure communication ensured by disjoint paths.
- Compatible with **shared memory** (OpenMP) and **distributed memory** (MPI) architectures.

## 🛠️ Technologies Used

- `C`
- `MPI` (Message Passing Interface)
- `OpenMP` (Open Multi-Processing)
- `METIS` for graph partitioning

## 🔁 Algorithm Workflow

1. **Graph Generation**: Model the Bn graph where each node is a permutation and edges are swaps.
2. **Partitioning**: Use METIS to split the graph into `k` balanced partitions.
3. **Parallel Parent Calculation**:
   - Each processor receives a partition.
   - Runs `Parent1(v, t, n)` to assign parent nodes.
4. **IST Construction**: Build the trees from the parent mappings.

## 📂 Folder Structure (Sample)

