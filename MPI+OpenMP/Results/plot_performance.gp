set terminal png size 800,600
set output 'performance_plot.png'
set title 'Performance Analysis of Parent1 Algorithm'
set xlabel 'Configuration (Process x Threads)'
set ylabel 'Execution Time (s)'
set y2label 'Speedup'
set y2tics nomirror
set key outside
set grid
set datafile separator ','
set xtics rotate by -45
plot 'performance_results.csv' using 0:4 with linespoints title 'Sequential', \
     'performance_results.csv' using 0:5 with linespoints title 'OpenMP', \
     'performance_results.csv' using 0:6 with linespoints title 'MPI', \
     'performance_results.csv' using 0:7 with linespoints title 'Hybrid', \
     'performance_results.csv' using 0:8 with linespoints axes x1y2 title 'Speedup'
