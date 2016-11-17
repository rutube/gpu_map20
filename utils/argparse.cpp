//
// Created by tumbler on 17.11.16.
//


#include <stdio.h>

#include <getopt.h>
#include <stdlib.h>
#include "argparse.h"


void print_usage(char *prog) {
    printf("Usage: %s [options] <matrix_file> <relevance_file> <weights_file>\n", prog);
    printf("Options:\n");
    printf("--factors <int>\tnumber of factors\n");
    printf("--moffset <int>\tmatrix offset\n");
    printf("--roffset <int>\trelevance offset\n");
    printf("--rows <int>\tnumber of rows\n");
    printf("--binary\toutput result as binary array\n");
    exit(1);
}


gpu_map20_args* parse_args(int argc, char **argv) {
    static int binary_flag = 0;
    char *pointers[3];

    int c;
    int option_index = 0;

    gpu_map20_args* args = (gpu_map20_args*)malloc(sizeof(gpu_map20_args)); ;
    args->factors = 48;
    args->matrix_offset = 0;
    args->relevance_offset = 0;
    args->rows = 0;
    args->binary_flag = 0;

    if (argc < 4) {
        print_usage(argv[0]);
        return NULL;
    }

    while (1) {
        static struct option long_options[] =
                {
                        {"factors", required_argument, 0,            'f'},
                        {"moffset", required_argument, 0,            'm'},
                        {"roffset", required_argument, 0,            'r'},
                        {"rows",    required_argument, 0,            'n'},
                        {"binary",  no_argument,       &binary_flag, 1},
                        {0, 0,                         0,            0}
                };
        c = getopt_long(argc, argv, "f:m:r:n:b",
                        long_options, &option_index);
        if (c == -1) break;

        switch (c) {
            case 'f':
                args->factors = atoi(optarg);
                if (args->factors == 0) {
                    print_usage(argv[0]);
                    return NULL;
                }
                break;
            case 'n':
                args->rows = atoi(optarg);
                if (args->rows == 0) {
                    print_usage(argv[0]);
                    return NULL;
                }
                break;
            case 'm':
                args->matrix_offset = atoi(optarg);
                if (args->matrix_offset == 0) {
                    print_usage(argv[0]);
                    return NULL;
                }
                break;
            case 'r':
                args->relevance_offset = atoi(optarg);
                if (args->relevance_offset == 0) {
                    print_usage(argv[0]);
                    return NULL;
                }
                break;

            default:
                break;
        }
    }
    args->binary_flag = binary_flag;

    option_index = 0;
    while (optind < argc) {
        pointers[option_index++] = argv[optind++];

    }
    args->matrix_file = pointers[0];
    args->relevance_file = pointers[1];
    args->weights_file = pointers[2];
    if (option_index < 3) {
        print_usage(argv[0]);
        return NULL;
    }
    return args;
}