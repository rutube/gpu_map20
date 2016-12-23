//
// Created by tumbler on 17.11.16.
//

#ifndef GPU_MAP20_ARGPARSE_H
#define GPU_MAP20_ARGPARSE_H


typedef struct _gpu_map20_args {
    char* matrix_file;
    char* relevance_file;
    char* weights_file;
    char* queries_file;
    int factors;
    size_t matrix_offset;
    size_t relevance_offset;
    size_t queries_offset;
    int rows;
    int append_flag;
} gpu_map20_args;

/// Разбирает аргументы командной строки для программы
/// \param argc число аргументов
/// \param argv массив указателей на входные параметры программы
/// \return указатель на структуру с разобранными параметрами
gpu_map20_args* parse_args(int argc, char **argv);

#endif //GPU_MAP20_ARGPARSE_H
