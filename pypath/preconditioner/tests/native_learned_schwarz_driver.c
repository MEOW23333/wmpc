#include "ni_gmres_schwarz.h"
#include "spdefs.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int size;
    double *matrix_values;
    double *rhs;
    double *initial_guess;
    double *initial_residual;
    int *ext_to_int_row;
    int *int_to_ext_col;
    ElementPtr *first_in_row;
    struct MatrixElement *elements;
    MatrixFrame matrix_frame;
    SMPmatrix smp_matrix;
    CKTcircuit circuit;
} parity_case_t;

/*
 * The parity driver links ni_gmres_schwarz.c without the rest of ngspice.
 * These two functions are the only external sparse helpers referenced by the
 * module. Rows are constructed linked already, so spcLinkRows is defensive.
 */
int
SMPmatSize(SMPmatrix *matrix)
{
    if (!matrix || !matrix->SPmatrix)
        return 0;
    return matrix->SPmatrix->Size;
}

void
spcLinkRows(MatrixPtr matrix)
{
    if (matrix)
        matrix->RowsLinked = 1;
}

static void
parity_case_clear(parity_case_t *test_case)
{
    if (!test_case)
        return;
    free(test_case->matrix_values);
    free(test_case->rhs);
    free(test_case->initial_guess);
    free(test_case->initial_residual);
    free(test_case->ext_to_int_row);
    free(test_case->int_to_ext_col);
    free(test_case->first_in_row);
    free(test_case->elements);
    memset(test_case, 0, sizeof(*test_case));
}

static int
parse_int(const char *text, int *out)
{
    char *end = NULL;
    long value;
    if (!text || !out)
        return 0;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < 0 || value > 2147483647L)
        return 0;
    *out = (int) value;
    return 1;
}

static int
parse_double(const char *text, double *out)
{
    char *end = NULL;
    double value;
    if (!text || !out)
        return 0;
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value))
        return 0;
    *out = value;
    return 1;
}

static int
read_case_file(const char *path, parity_case_t *test_case)
{
    FILE *fp = NULL;
    double *vectors[3];
    size_t value_count;
    size_t vector_index;
    size_t index;
    int size;

    if (!path || !test_case)
        return 0;
    memset(test_case, 0, sizeof(*test_case));
    fp = fopen(path, "r");
    if (!fp)
        return 0;
    if (fscanf(fp, "%d", &size) != 1 || size <= 0 || size > 100000) {
        fclose(fp);
        return 0;
    }
    if ((size_t) size > ((size_t) -1) / (size_t) size) {
        fclose(fp);
        return 0;
    }
    value_count = (size_t) size * (size_t) size;
    if (value_count > ((size_t) -1) / sizeof(double)) {
        fclose(fp);
        return 0;
    }
    test_case->size = size;
    test_case->matrix_values =
        (double *) malloc(value_count * sizeof(double));
    test_case->rhs = (double *) malloc((size_t) size * sizeof(double));
    test_case->initial_guess =
        (double *) malloc((size_t) size * sizeof(double));
    test_case->initial_residual =
        (double *) malloc((size_t) size * sizeof(double));
    if (!test_case->matrix_values || !test_case->rhs ||
        !test_case->initial_guess || !test_case->initial_residual) {
        fclose(fp);
        parity_case_clear(test_case);
        return 0;
    }
    for (index = 0; index < value_count; index++) {
        if (fscanf(fp, "%lf", &test_case->matrix_values[index]) != 1 ||
            !isfinite(test_case->matrix_values[index])) {
            fclose(fp);
            parity_case_clear(test_case);
            return 0;
        }
    }
    vectors[0] = test_case->rhs;
    vectors[1] = test_case->initial_guess;
    vectors[2] = test_case->initial_residual;
    for (vector_index = 0U; vector_index < 3U; vector_index++) {
        for (index = 0U; index < (size_t) size; index++) {
            if (fscanf(fp, "%lf", &vectors[vector_index][index]) != 1 ||
                !isfinite(vectors[vector_index][index])) {
                fclose(fp);
                parity_case_clear(test_case);
                return 0;
            }
        }
    }
    fclose(fp);
    return 1;
}

static int
build_sparse_view(parity_case_t *test_case)
{
    int row;
    int column;
    int size;
    size_t element_count;

    if (!test_case || test_case->size <= 0 || !test_case->matrix_values)
        return 0;
    size = test_case->size;
    element_count = (size_t) size * (size_t) size;
    test_case->ext_to_int_row =
        (int *) calloc((size_t) size + 1U, sizeof(int));
    test_case->int_to_ext_col =
        (int *) calloc((size_t) size + 1U, sizeof(int));
    test_case->first_in_row =
        (ElementPtr *) calloc((size_t) size + 1U, sizeof(ElementPtr));
    test_case->elements = (struct MatrixElement *) calloc(
        element_count,
        sizeof(struct MatrixElement));
    if (!test_case->ext_to_int_row ||
        !test_case->int_to_ext_col ||
        !test_case->first_in_row ||
        !test_case->elements)
        return 0;

    for (row = 0; row <= size; row++) {
        test_case->ext_to_int_row[row] = row;
        test_case->int_to_ext_col[row] = row;
    }
    for (row = 0; row < size; row++) {
        struct MatrixElement *row_elements =
            test_case->elements + (size_t) row * (size_t) size;
        test_case->first_in_row[row + 1] = row_elements;
        for (column = 0; column < size; column++) {
            struct MatrixElement *element = row_elements + column;
            element->Row = row + 1;
            element->Col = column + 1;
            element->Real = test_case->matrix_values[
                (size_t) row * (size_t) size + (size_t) column];
            element->Imag = 0.0;
            element->NextInRow =
                column + 1 < size ? element + 1 : NULL;
            element->NextInCol = NULL;
        }
    }

    memset(&test_case->matrix_frame, 0, sizeof(test_case->matrix_frame));
    test_case->matrix_frame.Size = size;
    test_case->matrix_frame.CurrentSize = size;
    test_case->matrix_frame.ExtSize = size;
    test_case->matrix_frame.Factored = 0;
    test_case->matrix_frame.Complex = 0;
    test_case->matrix_frame.RowsLinked = 1;
    test_case->matrix_frame.ExtToIntRowMap = test_case->ext_to_int_row;
    test_case->matrix_frame.IntToExtColMap = test_case->int_to_ext_col;
    test_case->matrix_frame.FirstInRow = test_case->first_in_row;

    memset(&test_case->smp_matrix, 0, sizeof(test_case->smp_matrix));
    test_case->smp_matrix.SPmatrix = &test_case->matrix_frame;
    memset(&test_case->circuit, 0, sizeof(test_case->circuit));
    test_case->circuit.CKTmatrix = &test_case->smp_matrix;
    return 1;
}

static void
print_failure(const char *reason)
{
    printf(
        "{\"create_ok\":false,\"apply_ok\":false,\"reason\":\"%s\"}\n",
        reason ? reason : "");
}

static void
print_success(
    const double *output,
    int size,
    const ngspice_gmres_schwarz_metrics_t *metrics
)
{
    int index;
    printf(
        "{\"create_ok\":true,\"apply_ok\":true,\"reason\":\"\","
        "\"output\":[");
    for (index = 0; index < size; index++) {
        if (index > 0)
            putchar(',');
        printf("%.17g", output[index]);
    }
    printf(
        "],\"metrics\":{\"matrix_size\":%d,\"block_count\":%d,"
        "\"total_block_rows\":%d,\"covered_rows\":%d,"
        "\"uncovered_rows\":%d,\"max_block_size\":%d,"
        "\"apply_count\":%d,\"failed_apply_count\":%d,"
        "\"gmin\":%.17g,\"lambda_min\":%.17g,\"lambda_max\":%.17g,"
        "\"retained_bytes\":%lu,\"peak_estimated_bytes\":%lu}}\n",
        metrics->matrix_size,
        metrics->block_count,
        metrics->total_block_rows,
        metrics->covered_rows,
        metrics->uncovered_rows,
        metrics->max_block_size,
        metrics->apply_count,
        metrics->failed_apply_count,
        metrics->gmin,
        metrics->lambda_min,
        metrics->lambda_max,
        (unsigned long) metrics->retained_bytes,
        (unsigned long) metrics->peak_estimated_bytes);
}

int
main(int argc, char **argv)
{
    parity_case_t test_case;
    ngspice_gmres_schwarz_state_t *state = NULL;
    ngspice_gmres_schwarz_metrics_t metrics;
    double *output = NULL;
    double time_value;
    double gmin;
    int newton_iter;
    char reason[256];
    int create_ok;
    int apply_ok;

    if (argc != 8) {
        fprintf(
            stderr,
            "usage: %s SIDECAR CASE TIME GMIN NEWTON_ITER NODE_MAP_HASH INITIAL_GUESS_MODE\n",
            argv[0]);
        return 64;
    }
    if (!parse_double(argv[3], &time_value) ||
        !parse_double(argv[4], &gmin) ||
        !parse_int(argv[5], &newton_iter)) {
        fprintf(stderr, "invalid numeric argument\n");
        return 64;
    }
    if (!read_case_file(argv[2], &test_case) ||
        !build_sparse_view(&test_case)) {
        fprintf(stderr, "failed to construct parity case\n");
        parity_case_clear(&test_case);
        return 65;
    }
    test_case.circuit.CKTtime = time_value;
    test_case.circuit.CKTdiagGmin = gmin;
    reason[0] = '\0';
    create_ok = ngspice_gmres_schwarz_create(
        &test_case.circuit,
        argv[1],
        test_case.size,
        newton_iter,
        argv[6],
        test_case.rhs,
        test_case.initial_guess,
        test_case.initial_residual,
        argv[7],
        &state,
        reason,
        sizeof(reason));
    if (!create_ok) {
        print_failure(reason);
        parity_case_clear(&test_case);
        return 2;
    }

    output = (double *) calloc((size_t) test_case.size, sizeof(double));
    if (!output) {
        ngspice_gmres_schwarz_destroy(state);
        parity_case_clear(&test_case);
        return 66;
    }
    apply_ok = ngspice_gmres_schwarz_apply(
        state,
        test_case.rhs,
        output,
        test_case.size);
    if (!apply_ok) {
        printf(
            "{\"create_ok\":true,\"apply_ok\":false,"
            "\"reason\":\"apply_failed\"}\n");
        free(output);
        ngspice_gmres_schwarz_destroy(state);
        parity_case_clear(&test_case);
        return 3;
    }
    ngspice_gmres_schwarz_get_metrics(state, &metrics);
    print_success(output, test_case.size, &metrics);
    free(output);
    ngspice_gmres_schwarz_destroy(state);
    parity_case_clear(&test_case);
    return 0;
}
