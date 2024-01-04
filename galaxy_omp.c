/*
  Compile with gcc -O3 galaxy_omp.c -o galaxy_omp -lm
  Run sequentially with srun -n 1 galaxy_omp input.txt input_rand.txt outfile.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define binsperdegree 4 /* Nr of bins per degree */
#define totaldegrees 64 /* Nr of degrees */

/* Count how many lines the input file has */
int count_lines(FILE *infile)
{
    char readline[80]; /* Buffer for file input */
    int lines = 0;
    while (fgets(readline, 80, infile) != NULL)
        lines++;
    rewind(infile); /* Reset the file to the beginning */
    return (lines);
}

/* Read input data from the file, convert to cartesian coordinates
   and write them to arrays x, y and z */
void read_data(FILE *infile, int n, float *x, float *y, float *z)
{
    char readline[80]; /* Buffer for file input */
    float ra, dec;
    int i = 0;
    float dpi = acos(-1.0f);
    while (fgets(readline, 80, infile) != NULL) /* Read a line */
    {
        sscanf(readline, "%f %f", &ra, &dec); /* Read a coordinate pair */
        /* Convert to cartesian coordinates */
        float phi = ra * dpi / 180.0f;
        float theta = (90.0f - dec) * dpi / 180.0f;
        x[i] = sinf(theta) * cosf(phi);
        y[i] = sinf(theta) * sinf(phi);
        z[i] = cosf(theta);
        ++i;
    }
    fclose(infile);
}

/*
  Compute the angle between two observations p and q and add it to the histogram
*/
void add_histogram(float px, float py, float pz,
                   float qx, float qy, float qz, long int *histogram,
                   const float pi, const float costotaldegrees)
{
    float degreefactor = 180.0 / pi * binsperdegree;
    float theta = px * qx + py * qy + pz * qz;
    if (theta >= costotaldegrees)
    { /* Skip if theta < costotaldegrees */
        if (theta > 1.0f)
            theta = 1.0f;
        /* Calculate which bin to increment */
        /* histogram [(int)(acos(theta)*180.0/pi*binsperdegree)] += 1L; */
        
        int bin = (int)(acosf(theta) * degreefactor);

        #pragma omp atomic
        histogram[bin]++;
    }
}

int main(int argc, char *argv[])
{
    int nthreads, tid;
    int nr_of_bins = binsperdegree * totaldegrees;     /* Total number of bins */
    long int *histogramDD, *histogramDR, *histogramRR; /* Arrays for histograms */
    float *xd_real, *yd_real, *zd_real;                /* Arrays for real data */
    float *xd_sim, *yd_sim, *zd_sim;                   /* Arrays for random data */

    double NSimdivNReal, w;
    FILE *infile, *outfile; /* Input and output files */
    double starttime, stoptime;

    /* Check that we have 4 command line arguments */
    if (argc != 4)
    {
        printf("Usage: %s real_data sim_data output_file\n", argv[0]);
        return (0);
    }

    starttime = omp_get_wtime();  /* Master thread measures the execution time */
    
    float pi = acosf(-1.0f);
    float costotaldegrees = (float)(cos(totaldegrees / 180.0f * pi));

    /* Open the real data input file */
    infile = fopen(argv[1], "r");
    if (infile == NULL)
    {
        printf("Unable to open %s\n", argv[1]);
        return (0);
    }

    /* Count how many lines the input file has */
    int Nooflines_Real = count_lines(infile);
    printf("%s contains %d lines\n", argv[1], Nooflines_Real);

    /* Allocate arrays for x, y and z values */
    xd_real = (float *)calloc(Nooflines_Real, sizeof(float));
    yd_real = (float *)calloc(Nooflines_Real, sizeof(float));
    zd_real = (float *)calloc(Nooflines_Real, sizeof(float));

    /* Read the file with real input data */
    read_data(infile, Nooflines_Real, xd_real, yd_real, zd_real);

    /* Open the file with random (simulated) data */
    infile = fopen(argv[2], "r");
    if (infile == NULL)
    {
        printf("Unable to open %s\n", argv[2]);
        return (0);
    }

    /* Count how many lines the file has */
    int Nooflines_Sim = count_lines(infile);
    printf("%s contains %d lines\n", argv[2], Nooflines_Sim);

    /* Allocate arrays for x, y and z values */
    xd_sim = (float *)calloc(Nooflines_Sim, sizeof(float));
    yd_sim = (float *)calloc(Nooflines_Sim, sizeof(float));
    zd_sim = (float *)calloc(Nooflines_Sim, sizeof(float));

    /* Read the input file */
    read_data(infile, Nooflines_Sim, xd_sim, yd_sim, zd_sim);

    /* Allocate arrays for the histograms */
    histogramDD = (long int *)calloc(nr_of_bins + 1, sizeof(long int));
    histogramDR = (long int *)calloc(nr_of_bins + 1, sizeof(long int));
    histogramRR = (long int *)calloc(nr_of_bins + 1, sizeof(long int));

    /* Initialize the histograms to zero */
    for (int i = 0; i <= nr_of_bins; ++i)
    {
        histogramDD[i] = 0L;
        histogramDR[i] = 0L;
        histogramRR[i] = 0L;
    }

    printf("Calculating DD angle histogram...\n");

    int i, j;

    #pragma omp parallel for schedule(dynamic) private(i, j) shared(xd_real, yd_real, zd_real, histogramDD)
    for (i = 0; i < Nooflines_Real; ++i)
    {
        /* Print i every 10000 iterations to monitor progress */
        // if ( i > 0 && (i/10000)*10000 == i ) printf("     %6d\n",i);
        for (j = i + 1; j < Nooflines_Real; ++j)
        {
            // #pragma omp atomic
            add_histogram(xd_real[i], yd_real[i], zd_real[i],
                          xd_real[j], yd_real[j], zd_real[j], histogramDD, pi, costotaldegrees);
        }
    }

    /* Multiply DD histogram with 2 since we only calculate (i,j) pair, not (j,i) */
    #pragma omp parallel for private(i) shared(histogramDD)
    for (i = 0; i <= nr_of_bins; ++i)
        
        #pragma omp atomic
        histogramDD[i] *= 2L;

    // All DD pairs (i,i) have an angle of 0 and are therefore added to bin number zero
    histogramDD[0] += ((long)(Nooflines_Real));

    /* Count the total nr of values in the DD histograms */
    long int TotalCountDD = 0L;

    #pragma omp parallel for reduction(+ : TotalCountDD) shared(histogramDD)
    for (i = 0; i <= nr_of_bins; ++i)
        TotalCountDD += (long)(histogramDD[i]);
        
    printf("  DD histogram count = %ld\n\n", TotalCountDD);

    printf("Calculating DR angle histogram...\n");

    #pragma omp parallel for schedule(dynamic) private(i, j) shared(xd_real, yd_real, zd_real, xd_sim, yd_sim, zd_sim, histogramDR)
    for (i = 0; i < Nooflines_Real; ++i)
    {
        // if ( i > 0 && (i/10000)*10000 == i ) printf("     %6d\n",i);
        //  For DR angles we have to spin over all pairs since we have two different data arrays
        for (j = 0; j < Nooflines_Sim; ++j)
        {
            // #pragma omp atomic
            add_histogram(xd_real[i], yd_real[i], zd_real[i],
                          xd_sim[j], yd_sim[j], zd_sim[j], histogramDR, pi, costotaldegrees);
        }
    }

    /* Count the total nr of values in the DR histograms */
    long int TotalCountDR = 0L;
    
    #pragma omp parallel for reduction(+ : TotalCountDR) shared(histogramDR)
    for (i = 0; i <= nr_of_bins; ++i)
        TotalCountDR += (long)(histogramDR[i]);
    printf("  DR histogram count = %ld\n\n", TotalCountDR);

    printf("Calculating RR angle histogram...\n");

    #pragma omp parallel for schedule(dynamic) private(i, j) shared(xd_sim, yd_sim, zd_sim, histogramRR)
    for (i = 0; i < Nooflines_Sim; ++i)
    {
        // if ( i > 0 && (i/10000)*10000 == i ) printf("     %6d\n",i);
        for (j = i + 1; j < Nooflines_Sim; ++j)
        {
            // #pragma omp atomic
            add_histogram(xd_sim[i], yd_sim[i], zd_sim[i],
                          xd_sim[j], yd_sim[j], zd_sim[j], histogramRR, pi, costotaldegrees);
        }
    }

/* Multiply RR histogram with 2 since we only calculate (i,j) pair, not (j,i) */
    #pragma omp parallel for private(i) shared(histogramRR)
    for (i = 0; i <= nr_of_bins; ++i)
        
        #pragma omp atomic
        histogramRR[i] *= 2L;
    // All RR pairs (i,i) have an angle of 0 and are therefore added to bin number zero
    histogramRR[0] += ((long)(Nooflines_Sim));

    /* Count the total nr of values in the RR histograms */
    long int TotalCountRR = 0L;
    
    #pragma omp parallel for reduction(+ : TotalCountRR) shared(histogramRR)
    for (i = 0; i <= nr_of_bins; ++i)
        TotalCountRR += (long)(histogramRR[i]);
    printf("  RR histogram count = %ld\n\n", TotalCountRR);

    printf("\n\n");

    /* Open the output file */
    outfile = fopen(argv[3], "w");
    if (outfile == NULL)
    {
        printf("Unable to open %s\n", argv[3]);
        return (-1);
    }
    /* Write the histograms both to display and outfile */
    printf("bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
    fprintf(outfile, "bin center\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
    for (int i = 0; i < nr_of_bins; ++i)
    {
        NSimdivNReal = ((double)(Nooflines_Sim)) / ((double)(Nooflines_Real));
        w = 1.0 + NSimdivNReal * NSimdivNReal * histogramDD[i] / histogramRR[i] - 2.0 * NSimdivNReal * histogramDR[i] / ((double)(histogramRR[i]));
        printf(" %6.3f      %3.6f\t%15ld\t%15ld\t%15ld\n", ((float)i + 0.5) / binsperdegree, w,
               histogramDD[i], histogramDR[i], histogramRR[i]);
        fprintf(outfile, "%6.3f\t%15lf\t%15ld\t%15ld\t%15ld\n", ((float)i + 0.5) / binsperdegree, w,
                histogramDD[i], histogramDR[i], histogramRR[i]);
    }

    fclose(outfile);

    /* Free all allocated arrays */
    free(histogramDD);
    free(histogramDR);
    free(histogramRR);
    free(xd_sim);
    free(yd_sim);
    free(zd_sim);
    free(xd_real);
    free(yd_real);
    free(zd_real);

    stoptime = omp_get_wtime(); /* Stop the timer and print time */
    printf("\nTime = %6.1f seconds\n",
           ((double)(stoptime - starttime)));
           
    return (0);
}

