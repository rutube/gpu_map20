#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <stdlib.h>

#include "kernels/constants.h"
#include "utils/utils.h"
#include "utils/argparse.h"
#include "core/ranks.h"
#include "core/map20.h"

static const char *const OUTPUT_FILENAME = "map_20.bin";
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

    return final_rows;
}


int read_queries(const char *queries_file, int *queries[], size_t queries_offset) {
    size_t size = (size_t) fileSize(queries_file);
    if (size == 0) {
        cout << "invalid queries file" << endl;
        return 0;
    }
    if (size % sizeof(int)) {
        cout << "invalid queries file size" << endl;
        return 0;
    }
    FILE *f = fopen(queries_file, "rb");
    if (!f) {
        return 0;
    }
    size -= queries_offset;
    if (size)
        fseek(f, queries_offset, 0);
    cudacall(cudaMallocHost((void**) queries, size));
    fread(*queries, sizeof(int), size, f);
    fclose(f);
    return (int) (size / sizeof(int));
}


int main(int argc, char **argv) {

    gpu_map20_args* args = parse_args(argc, argv);
    int *queries;
    int num_queries = 1;
    int total_rows;
    if (args->queries_file) {
        num_queries = read_queries(args->queries_file, &queries, args->queries_offset);
        if (num_queries == 0){
            cout << "queries is empty" << endl;
            return -1;
        }
        total_rows = args->rows;
        args->rows = 0;
        int offset = 0;
        for(int i = 0; i < num_queries; i++) {
            args->rows += queries[i];
            // transform query_row_count -> query_offset
            queries[i] = offset;
            if (total_rows && offset >= total_rows) {
                num_queries = i;
                break;
            }
            offset = args->rows;
        }
        total_rows = offset;

    } else {
        cudacall(cudaMallocHost((void **)&queries, sizeof(queries[0])));

        total_rows = check_rows(args->matrix_file, args->relevance_file, args->weights_file,
                                args->matrix_offset, args->relevance_offset, args->rows, args->factors);
        queries[0] = total_rows;
    }

    if (total_rows == 0) {
        cout << "total rows is zero" << endl;
        return -1;
    }

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
    cout << "qoffset: " << args->queries_offset << endl;
    cout << "queries: " << num_queries << endl;
    cout << "total rows: " << total_rows << endl;
    cout << "variants:" << variants << endl;
    cout << (args->append_flag ? "appending" : "write") << " to "  << OUTPUT_FILENAME << endl;
    cout << endl;
    cout << "Float size: " << sizeof(float) << endl;
    cout << "Int  size: " << sizeof(int) << endl;
    cout << "Initializing GPU..." << endl;
    init_gpu();
    cout << "Initializing CuBLAS..." << endl;
    cublasHandle_t blas_handle = init_cublas();

    cout << "Loading weights file..." << endl;
    // Загружаем матрицу весов ранкера
    // матрица <variants> x <factors> построчно
    float *weights = load_matrix(
            args->weights_file, 0, args->factors, variants);
    float *gpu_weights = upload_to_gpu(
            weights, args->factors * variants);

    float *gpu_map20;

    if (args->append_flag && (fileSize(OUTPUT_FILENAME) > 0)) {
        cout << "Loading " << OUTPUT_FILENAME << "..." << endl;
        float * map20 = load_matrix(OUTPUT_FILENAME, 0, variants, 1);
        cout << "Uploading to gpu..." << endl;
        gpu_map20 = upload_to_gpu(map20, variants);
    } else {
        cudacall(cudaMalloc((void**) &gpu_map20, variants * sizeof(gpu_map20[0])));
        cudacall(cudaMemset(gpu_map20, 0, variants * sizeof(gpu_map20[0])));
    }

    cout << "Preparing ranks..." << endl;
    float *gpu_ranked = prepare_ranks(blas_handle, args->matrix_file, args->matrix_offset, gpu_weights,
                                      total_rows, args->factors, variants);
    cout << "Loading relevance file @ " << args->relevance_offset << " [ " << total_rows << "]" << endl;
    float *relevance = load_matrix(args->relevance_file, args->relevance_offset, 1, total_rows);

    compute_map20(blas_handle, gpu_ranked, gpu_map20, relevance, queries, num_queries, total_rows, variants);
    cleanup_gpu(NULL, 0, &gpu_ranked, 1, NULL, false);


    cout << "Downoading from GPU..." << endl;
    float * map20 = download_from_gpu(gpu_map20, variants);

    cout << "Writing to file..." << endl;
    save_matrix(OUTPUT_FILENAME, map20, variants, 1);

    cleanup_gpu(&map20, 1, &gpu_map20, 1, blas_handle, true);

    free(args);
    return 0;
}
