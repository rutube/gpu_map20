///
/// Симуляция работы поискового движка путем перемножения матрицы признаков найденных документов на вектор(а) весов.
///


/// Подготавливает веса выдачи для всех вариантов и всех результатов поиска
/// \param blas_handle - хендл библиотеки CuBLAS
/// \param matrix_file - путь до файла с матрицей признаков
/// \param matrix_offset - смещение в матрице признаков
/// \param gpu_weights - указатель на gpu-матрицу вариантов
/// \param rows - число объектов выдачи
/// \param factors - число признаков
/// \param variants - число вариантов
/// \return указатель на gpu-массив, содержащий матрицу matrix * weights.T (по столбцам) размера rows * variants
float *prepare_ranks(cublasHandle_t blas_handle, const char *matrix_file, const size_t matrix_offset,
                     const float *gpu_weights, const int rows, const int factors, const int variants);
