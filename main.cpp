#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

#include "kernels/constants.h"
#include "utils/utils.h"
#include "utils/argparse.h"
#include "core/ranks.h"
#include "core/map20.h"

using namespace std;

/**
* Программа расчета метрики качества поиска MAP@20 по бинарному представлению
* выдачи, набору релевантных результатов и вектору (набору векторов) весов
**/

off_t fileSize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) == 0)
        return st.st_size;

    return -1;
}

int check_rows(const char *matrix_file, const char *relevance_file,
               const char *weights_file, int matrix_offset,
               int relevance_offset, int rows, int factors) {
    /**
    Проверяет файлы с данными и соответствие их размеров остальным аргументам
    **/

    int final_rows = 0;

    // Проверяем существование всех файлов
    off_t matrix_size = fileSize(matrix_file);
    if (matrix_size <= 0) {
        cout << "Matrix file is empty or does not exist" << endl;
        return 0;
    }
    off_t relevance_size = fileSize(relevance_file);
    if (relevance_size <= 0) {
        cout << "Relevance file is empty or does not exist" << endl;
        return 0;
    }

    off_t weights_size = fileSize(weights_file);
    if (weights_size <= 0) {
        cout << "Weights file is empty or does not exist" << endl;
        return 0;
    }

    // проверяем что файл с весами соответствует указанному числу факторов.
    if (weights_size % (factors * 4)) {
        cout << "Weights matrix size " <<
             weights_size << " does not match factors count" << endl;
        return 0;
    }

    // учитываем смещение в файлах
    matrix_size -= matrix_offset;
    relevance_size -= relevance_offset;

    if (rows == 0) {
        // Размер матрицы должен соответствовать числу факторов
        if (matrix_size % (factors * 4)) {
            cout << "Matrix size " << matrix_size <<
                 " does not match factors count " << endl;
            return 0;
        }
        // Число строк высчитываем по размеру матрицы и числу факторов
        final_rows = (int) (matrix_size / (factors * 4));

        // Проверяем что число строк в матрице и число элементов в relevance
        // совпадают
        if (relevance_size != final_rows * 4) {
            cout << "Relevance size " << relevance_size <<
                 " sdoes not match matrix size " << matrix_size << endl;
            return 0;
        }
    } else {
        // явно задано число строк в выдаче
        final_rows = rows;
    }

    // проверяем что размеров файлов достаточны для чтения указанного числа
    // строк (проверка актуальна при явном указании числа строк)
    if (matrix_size < final_rows * factors * 4) {
        cout << "Matrix size " << matrix_size << " is not enough for " << rows <<
             " rows and " << factors << "factors" << endl;
        return 0;
    }

    if (relevance_size < final_rows * 4) {
        cout << "Relevance size " << relevance_size << " is not enough for " << rows <<
             " rows" << endl;
        return 0;
    }

    if (final_rows > MAX_MATCHES) {
        cout << "Rows count is more than " << MAX_MATCHES << endl;
        return 0;
    }

    return final_rows;
}


int main(int argc, char **argv) {

    gpu_map20_args* args = parse_args(argc, argv);

    int rows = check_rows(args->matrix_file, args->relevance_file, args->weights_file, args->matrix_offset,
                          args->relevance_offset, args->rows, args->factors);
    if (rows == 0)
        return -1;

    off_t weights_size = fileSize(args->weights_file);
    int variants = (int) (weights_size / (4 * args->factors));

    cout << "M: " << args->matrix_file << endl;
    cout << "R: " << args->relevance_file << endl;
    cout << "W: " << args->weights_file << endl;
    if (args->queries_file) {
        cout << "Q: " << args->queries_file << endl;
    }
    cout << "factors: " << args->factors << endl;
    cout << "moffset: " << args->matrix_offset << endl;
    cout << "roffset: " << args->relevance_offset << endl;
    cout << "rows: " << rows << endl;
    cout << "variants:" << variants << endl;
    cout << (args->binary_flag ? "binary" : "text") << endl;

    cout << "Float size: " << sizeof(float) << endl;
    cout << "Initializing GPU..." << endl;
    init_gpu();
    cout << "Initializing CuBLAS..." << endl;
    cublasHandle_t blas_handle = init_cublas();


    float *gpu_ranked = prepare_ranks(blas_handle, args->matrix_file, args->matrix_offset, args->weights_file,
                                      rows, args->factors, variants);

    float *map20 = compute_map20(gpu_ranked, args->relevance_file, args->relevance_offset, rows, variants);

    // FIXME: Отладочное сохранение в файл, надо в бинарном/текстовом виде
    // писать в stdout
    cout << "saving to file..." << endl;
    save_matrix("map20.bin", map20, variants, 1);

    cleanup_gpu(&map20, 1, &gpu_ranked, 1, NULL, true);

    free(args);
    return 0;
}
