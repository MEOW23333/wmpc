#include "ni_gmres_schwarz.h"
#include "ni_gmres_sha256.h"

#include "ngspice/smpdefs.h"
#include "../sparse/spdefs.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>

#define NGSPICE_SCHWARZ_SCHEMA_VERSION 5
#define NGSPICE_SCHWARZ_MAX_BLOCK_SIZE 32
#define NGSPICE_SCHWARZ_MAX_SIDECAR_BYTES ((size_t) 512U * 1024U * 1024U)
#define NGSPICE_SCHWARZ_WEIGHT_SUM_TOL 1e-9
#define NGSPICE_SCHWARZ_PIVOT_TOL 1e-30
#define NGSPICE_SCHWARZ_HASH_HEX_LENGTH 64
#define NGSPICE_SCHWARZ_FEATURE_CONTRACT \
    "learned_schwarz_v1_abs_rhs_abs_initial_residual"
#define NGSPICE_SCHWARZ_LOCAL_SHIFT_CONTRACT \
    "block_inf_norm_relative_floor_v1"
#define NGSPICE_SCHWARZ_LOCAL_SHIFT_FLOOR_RELATIVE 1e-6
#define NGSPICE_SCHWARZ_INITIAL_RESIDUAL_CONTRACT \
    "linear_rhs - effective_matrix @ initial_guess"
#define NGSPICE_SCHWARZ_INITIAL_RESIDUAL_NORM_ATOL 1e-12
#define NGSPICE_SCHWARZ_INITIAL_RESIDUAL_NORM_RTOL 1e-9

typedef struct {
    int matrix_size;
    int row_index_base;
    int newton_iter;
    int block_count;
    int total_block_rows;
    double time_value;
    double gmin;
    int *block_offsets;
    int *block_rows;
    double *block_lambdas;
    double *block_row_weights;
    char node_map_hash[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char matrix_fingerprint[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char layout_sha256[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char checkpoint_sha256[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char linear_rhs_sha256[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char initial_guess_sha256[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char initial_residual_contract[
        sizeof(NGSPICE_SCHWARZ_INITIAL_RESIDUAL_CONTRACT)];
    double initial_residual_norm_l2;
    double initial_residual_norm_atol;
    double initial_residual_norm_rtol;
    char feature_contract[sizeof(NGSPICE_SCHWARZ_FEATURE_CONTRACT)];
    char local_shift_contract[sizeof(NGSPICE_SCHWARZ_LOCAL_SHIFT_CONTRACT)];
    double local_shift_floor_relative;
    char initial_guess_mode[16];
    size_t file_bytes;
    int max_block_size;
    double lambda_min;
    double lambda_max;
} ngspice_schwarz_sidecar_t;

typedef struct {
    int column;
    double value;
} ngspice_schwarz_row_entry_t;

struct ngspice_gmres_schwarz_state {
    int matrix_size;
    int block_count;
    int total_block_rows;
    int max_block_size;
    int *block_offsets;
    int *block_rows;
    double *block_row_weights;
    size_t *factor_offsets;
    double *factors;
    int *pivots;
    unsigned char *covered;
    double *fallback_scales;
    double *scratch;
    ngspice_gmres_schwarz_metrics_t metrics;
};

static double
ngspice_schwarz_now(void)
{
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0)
        return 0.0;
    return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}

static void
ngspice_schwarz_set_reason(char *reason, size_t reason_size, const char *value)
{
    if (!reason || reason_size == 0U)
        return;
    if (!value)
        value = "";
    strncpy(reason, value, reason_size - 1U);
    reason[reason_size - 1U] = '\0';
}

static void
ngspice_schwarz_set_block_reason(
    char *reason,
    size_t reason_size,
    const char *prefix,
    int block_id
)
{
    if (!reason || reason_size == 0U)
        return;
    snprintf(reason, reason_size, "%s_%d", prefix, block_id);
    reason[reason_size - 1U] = '\0';
}

static int
ngspice_schwarz_checked_bytes(size_t count, size_t item_size, size_t *out)
{
    if (!out)
        return 0;
    if (item_size != 0U && count > SIZE_MAX / item_size)
        return 0;
    *out = count * item_size;
    return 1;
}

static int
ngspice_schwarz_checked_add(size_t lhs, size_t rhs, size_t *out)
{
    if (!out || lhs > SIZE_MAX - rhs)
        return 0;
    *out = lhs + rhs;
    return 1;
}

static int
ngspice_schwarz_is_hex_hash(const char *value)
{
    size_t index;
    if (!value || strlen(value) != NGSPICE_SCHWARZ_HASH_HEX_LENGTH)
        return 0;
    for (index = 0; index < NGSPICE_SCHWARZ_HASH_HEX_LENGTH; index++) {
        if (!isxdigit((unsigned char) value[index]))
            return 0;
    }
    return 1;
}

static int
ngspice_schwarz_is_lower_hex_hash(const char *value)
{
    size_t index;
    if (!value || strlen(value) != NGSPICE_SCHWARZ_HASH_HEX_LENGTH)
        return 0;
    for (index = 0; index < NGSPICE_SCHWARZ_HASH_HEX_LENGTH; index++) {
        unsigned char character = (unsigned char) value[index];
        if (!isdigit(character) &&
            !(character >= (unsigned char) 'a' &&
              character <= (unsigned char) 'f'))
            return 0;
    }
    return 1;
}

static int
ngspice_schwarz_hash_equal_ci(const char *lhs, const char *rhs)
{
    size_t index;
    if (!ngspice_schwarz_is_hex_hash(lhs) || !ngspice_schwarz_is_hex_hash(rhs))
        return 0;
    for (index = 0; index < NGSPICE_SCHWARZ_HASH_HEX_LENGTH; index++) {
        if (tolower((unsigned char) lhs[index]) !=
            tolower((unsigned char) rhs[index]))
            return 0;
    }
    return 1;
}

static int
ngspice_schwarz_double_matches(double actual, double expected)
{
    double scale;
    double tolerance;
    if (!isfinite(actual) || !isfinite(expected))
        return 0;
    scale = fmax(fabs(actual), fabs(expected));
    tolerance = 16.0 * DBL_EPSILON * fmax(scale, DBL_MIN);
    return fabs(actual - expected) <= tolerance;
}

static int
ngspice_schwarz_l2_norm(
    const double *values,
    int count,
    double *out
)
{
    double scale = 0.0;
    double sumsq = 1.0;
    int index;

    if (!values || count <= 0 || !out)
        return 0;
    for (index = 0; index < count; index++) {
        double value = values[index];
        double magnitude;
        if (!isfinite(value))
            return 0;
        magnitude = fabs(value);
        if (magnitude == 0.0)
            continue;
        if (scale < magnitude) {
            double ratio = scale / magnitude;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = magnitude;
        } else {
            double ratio = magnitude / scale;
            sumsq += ratio * ratio;
        }
    }
    *out = scale == 0.0 ? 0.0 : scale * sqrt(sumsq);
    return isfinite(*out);
}

static int
ngspice_schwarz_initial_residual_norm_matches(
    double actual,
    double expected,
    double atol,
    double rtol
)
{
    double tolerance;
    if (!isfinite(actual) || !isfinite(expected) ||
        !isfinite(atol) || !isfinite(rtol) ||
        actual < 0.0 || expected < 0.0 ||
        atol < 0.0 || rtol < 0.0)
        return 0;
    tolerance = atol + rtol * fmax(fabs(actual), fabs(expected));
    return isfinite(tolerance) && fabs(actual - expected) <= tolerance;
}

static const char *
ngspice_schwarz_skip_space(const char *cursor)
{
    while (cursor && *cursor && isspace((unsigned char) *cursor))
        cursor++;
    return cursor;
}

static const char *
ngspice_schwarz_find_unique_json_value(const char *json, const char *key)
{
    char pattern[128];
    const char *found;
    const char *cursor;
    int written;

    if (!json || !key)
        return NULL;
    written = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written <= 0 || (size_t) written >= sizeof(pattern))
        return NULL;
    found = strstr(json, pattern);
    if (!found || strstr(found + (size_t) written, pattern))
        return NULL;
    cursor = ngspice_schwarz_skip_space(found + (size_t) written);
    if (!cursor || *cursor != ':')
        return NULL;
    return ngspice_schwarz_skip_space(cursor + 1);
}

static int
ngspice_schwarz_json_int(const char *json, const char *key, int *out)
{
    const char *cursor = ngspice_schwarz_find_unique_json_value(json, key);
    char *endptr;
    long parsed;
    if (!cursor || !out)
        return 0;
    errno = 0;
    parsed = strtol(cursor, &endptr, 10);
    if (errno != 0 || endptr == cursor || parsed < INT_MIN || parsed > INT_MAX)
        return 0;
    endptr = (char *) ngspice_schwarz_skip_space(endptr);
    if (*endptr != ',' && *endptr != '}' && *endptr != ']')
        return 0;
    *out = (int) parsed;
    return 1;
}

static int
ngspice_schwarz_json_double(const char *json, const char *key, double *out)
{
    const char *cursor = ngspice_schwarz_find_unique_json_value(json, key);
    char *endptr;
    double parsed;
    if (!cursor || !out)
        return 0;
    errno = 0;
    parsed = strtod(cursor, &endptr);
    if (errno != 0 || endptr == cursor || !isfinite(parsed))
        return 0;
    endptr = (char *) ngspice_schwarz_skip_space(endptr);
    if (*endptr != ',' && *endptr != '}' && *endptr != ']')
        return 0;
    *out = parsed;
    return 1;
}

static int
ngspice_schwarz_json_string(
    const char *json,
    const char *key,
    char *out,
    size_t out_size
)
{
    const char *cursor = ngspice_schwarz_find_unique_json_value(json, key);
    size_t length = 0U;
    if (!cursor || !out || out_size == 0U || *cursor != '"')
        return 0;
    cursor++;
    while (cursor[length] && cursor[length] != '"') {
        if (cursor[length] == '\\' || (unsigned char) cursor[length] < 0x20U)
            return 0;
        length++;
    }
    if (cursor[length] != '"' || length + 1U > out_size)
        return 0;
    memcpy(out, cursor, length);
    out[length] = '\0';
    return 1;
}

static int
ngspice_schwarz_json_int_array(
    const char *json,
    const char *key,
    int *out,
    int expected_count
)
{
    const char *cursor = ngspice_schwarz_find_unique_json_value(json, key);
    int index = 0;
    if (!cursor || expected_count < 0 || *cursor != '[')
        return 0;
    cursor = ngspice_schwarz_skip_space(cursor + 1);
    if (expected_count == 0)
        return cursor && *cursor == ']';

    while (cursor && *cursor && *cursor != ']') {
        char *endptr;
        long parsed;
        if (index >= expected_count)
            return 0;
        errno = 0;
        parsed = strtol(cursor, &endptr, 10);
        if (errno != 0 || endptr == cursor || parsed < INT_MIN || parsed > INT_MAX)
            return 0;
        out[index++] = (int) parsed;
        cursor = ngspice_schwarz_skip_space(endptr);
        if (*cursor == ',') {
            cursor = ngspice_schwarz_skip_space(cursor + 1);
            if (*cursor == ']')
                return 0;
        } else if (*cursor != ']') {
            return 0;
        }
    }
    return cursor && *cursor == ']' && index == expected_count;
}

static int
ngspice_schwarz_json_double_array(
    const char *json,
    const char *key,
    double *out,
    int expected_count
)
{
    const char *cursor = ngspice_schwarz_find_unique_json_value(json, key);
    int index = 0;
    if (!cursor || expected_count < 0 || *cursor != '[')
        return 0;
    cursor = ngspice_schwarz_skip_space(cursor + 1);
    if (expected_count == 0)
        return cursor && *cursor == ']';

    while (cursor && *cursor && *cursor != ']') {
        char *endptr;
        double parsed;
        if (index >= expected_count)
            return 0;
        errno = 0;
        parsed = strtod(cursor, &endptr);
        if (errno != 0 || endptr == cursor || !isfinite(parsed))
            return 0;
        out[index++] = parsed;
        cursor = ngspice_schwarz_skip_space(endptr);
        if (*cursor == ',') {
            cursor = ngspice_schwarz_skip_space(cursor + 1);
            if (*cursor == ']')
                return 0;
        } else if (*cursor != ']') {
            return 0;
        }
    }
    return cursor && *cursor == ']' && index == expected_count;
}

static char *
ngspice_schwarz_read_file(
    const char *path,
    size_t *file_bytes,
    char *reason,
    size_t reason_size
)
{
    struct stat st;
    FILE *fp;
    char *buffer;
    size_t bytes;

    if (!path || path[0] == '\0') {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_sidecar_missing");
        return NULL;
    }
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 2) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_sidecar_stat_failed");
        return NULL;
    }
    if ((uintmax_t) st.st_size > (uintmax_t) NGSPICE_SCHWARZ_MAX_SIDECAR_BYTES) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_sidecar_too_large");
        return NULL;
    }
    bytes = (size_t) st.st_size;
    fp = fopen(path, "rb");
    if (!fp) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_sidecar_open_failed");
        return NULL;
    }
    buffer = (char *) malloc(bytes + 1U);
    if (!buffer) {
        fclose(fp);
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_nomem");
        return NULL;
    }
    if (fread(buffer, 1U, bytes, fp) != bytes) {
        free(buffer);
        fclose(fp);
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_sidecar_read_failed");
        return NULL;
    }
    fclose(fp);
    buffer[bytes] = '\0';
    if (file_bytes)
        *file_bytes = bytes;
    return buffer;
}

static void
ngspice_schwarz_sidecar_clear(ngspice_schwarz_sidecar_t *sidecar)
{
    if (!sidecar)
        return;
    free(sidecar->block_offsets);
    free(sidecar->block_rows);
    free(sidecar->block_lambdas);
    free(sidecar->block_row_weights);
    memset(sidecar, 0, sizeof(*sidecar));
}

static int
ngspice_schwarz_sha256_update_int(
    ngspice_gmres_sha256_t *context,
    int value
)
{
    char buffer[32];
    int written = snprintf(buffer, sizeof(buffer), "%d", value);
    if (written <= 0 || (size_t) written >= sizeof(buffer))
        return 0;
    ngspice_gmres_sha256_update(context, buffer, (size_t) written);
    return 1;
}

static int
ngspice_schwarz_sha256_update_int_array(
    ngspice_gmres_sha256_t *context,
    const int *values,
    int count
)
{
    int index;
    if (!context || count < 0 || (count > 0 && !values))
        return 0;
    for (index = 0; index < count; index++) {
        if (index > 0)
            ngspice_gmres_sha256_update(context, ",", 1U);
        if (!ngspice_schwarz_sha256_update_int(context, values[index]))
            return 0;
    }
    ngspice_gmres_sha256_update(context, "\n", 1U);
    return 1;
}

static int
ngspice_schwarz_compute_layout_sha256(
    const ngspice_schwarz_sidecar_t *sidecar,
    char output[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1]
)
{
    static const char schema[] = "schema=learned_schwarz_layout_v1\n";
    static const char matrix_size_key[] = "matrix_size=";
    static const char block_count_key[] = "block_count=";
    static const char total_rows_key[] = "total_block_rows=";
    static const char offsets_key[] = "block_offsets=";
    static const char rows_key[] = "block_rows=";
    ngspice_gmres_sha256_t context;
    unsigned char digest[32];

    if (!sidecar || !output || sidecar->block_count < 0 ||
        sidecar->total_block_rows < 0 || !sidecar->block_offsets ||
        (sidecar->total_block_rows > 0 && !sidecar->block_rows))
        return 0;

    ngspice_gmres_sha256_init(&context);
    ngspice_gmres_sha256_update(&context, schema, sizeof(schema) - 1U);
    ngspice_gmres_sha256_update(
        &context, matrix_size_key, sizeof(matrix_size_key) - 1U);
    if (!ngspice_schwarz_sha256_update_int(
            &context, sidecar->matrix_size))
        return 0;
    ngspice_gmres_sha256_update(&context, "\n", 1U);
    ngspice_gmres_sha256_update(
        &context, block_count_key, sizeof(block_count_key) - 1U);
    if (!ngspice_schwarz_sha256_update_int(
            &context, sidecar->block_count))
        return 0;
    ngspice_gmres_sha256_update(&context, "\n", 1U);
    ngspice_gmres_sha256_update(
        &context, total_rows_key, sizeof(total_rows_key) - 1U);
    if (!ngspice_schwarz_sha256_update_int(
            &context, sidecar->total_block_rows))
        return 0;
    ngspice_gmres_sha256_update(&context, "\n", 1U);
    ngspice_gmres_sha256_update(
        &context, offsets_key, sizeof(offsets_key) - 1U);
    if (!ngspice_schwarz_sha256_update_int_array(
            &context,
            sidecar->block_offsets,
            sidecar->block_count + 1))
        return 0;
    ngspice_gmres_sha256_update(
        &context, rows_key, sizeof(rows_key) - 1U);
    if (!ngspice_schwarz_sha256_update_int_array(
            &context,
            sidecar->block_rows,
            sidecar->total_block_rows))
        return 0;
    ngspice_gmres_sha256_final(&context, digest);
    ngspice_gmres_sha256_hex(digest, output);
    return 1;
}

static int
ngspice_schwarz_initial_guess_mode_valid(const char *value)
{
    return value &&
        (strcmp(value, "rhsold") == 0 || strcmp(value, "zero") == 0);
}

static int
ngspice_schwarz_compute_vector_sha256(
    const double *values,
    int count,
    char output[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1],
    const char *nonfinite_reason,
    char *reason,
    size_t reason_size
)
{
    static const char schema[] = "schema=pals_vector_f64_v1\n";
    ngspice_gmres_sha256_t context;
    unsigned char digest[32];
    char length_line[64];
    int index;
    int written;

    if (!values || count <= 0 || !output || sizeof(double) != 8U) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_vector_sha256_bad_argument");
        return 0;
    }
    written = snprintf(
        length_line,
        sizeof(length_line),
        "length=%d\n",
        count);
    if (written <= 0 || (size_t) written >= sizeof(length_line)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_vector_sha256_header_failed");
        return 0;
    }

    ngspice_gmres_sha256_init(&context);
    ngspice_gmres_sha256_update(&context, schema, sizeof(schema) - 1U);
    ngspice_gmres_sha256_update(
        &context,
        length_line,
        (size_t) written);
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                nonfinite_reason);
            return 0;
        }
        ngspice_gmres_sha256_update_f64_le(&context, values[index]);
    }
    ngspice_gmres_sha256_final(&context, digest);
    ngspice_gmres_sha256_hex(digest, output);
    return 1;
}
static int
ngspice_schwarz_validate_sidecar(
    ngspice_schwarz_sidecar_t *sidecar,
    int expected_matrix_size,
    int expected_newton_iter,
    double expected_time,
    double expected_gmin,
    const char *expected_node_map_hash,
    const char *expected_initial_guess_mode,
    char *reason,
    size_t reason_size
)
{
    char actual_layout_sha256[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    double *weight_sums = NULL;
    unsigned char *covered = NULL;
    int block_id;
    int row;
    if (sidecar->newton_iter < 1) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_newton_iter");
        return 0;
    }
    if (strcmp(
            sidecar->feature_contract,
            NGSPICE_SCHWARZ_FEATURE_CONTRACT) != 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_feature_contract_mismatch");
        return 0;
    }
    if (strcmp(
            sidecar->local_shift_contract,
            NGSPICE_SCHWARZ_LOCAL_SHIFT_CONTRACT) != 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_local_shift_contract_mismatch");
        return 0;
    }
    if (!ngspice_schwarz_double_matches(
            sidecar->local_shift_floor_relative,
            NGSPICE_SCHWARZ_LOCAL_SHIFT_FLOOR_RELATIVE)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_local_shift_floor_relative_mismatch");
        return 0;
    }
    if (!ngspice_schwarz_initial_guess_mode_valid(
            sidecar->initial_guess_mode)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_initial_guess_mode");
        return 0;
    }
    if (!ngspice_schwarz_initial_guess_mode_valid(
            expected_initial_guess_mode)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_invalid_actual_initial_guess_mode");
        return 0;
    }
    if (strcmp(
            sidecar->initial_guess_mode,
            expected_initial_guess_mode) != 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_initial_guess_mode_mismatch");
        return 0;
    }

    if (sidecar->matrix_size != expected_matrix_size) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_matrix_size_mismatch");
        return 0;
    }
    if (sidecar->row_index_base != 1) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_unsupported_row_index_base");
        return 0;
    }
    if (!ngspice_schwarz_is_lower_hex_hash(
            sidecar->linear_rhs_sha256)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_linear_rhs_sha256");
        return 0;
    }
    if (!ngspice_schwarz_is_lower_hex_hash(
            sidecar->initial_guess_sha256)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_initial_guess_sha256");
        return 0;
    }
    if (strcmp(
            sidecar->initial_residual_contract,
            NGSPICE_SCHWARZ_INITIAL_RESIDUAL_CONTRACT) != 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_initial_residual_contract_mismatch");
        return 0;
    }
    if (!isfinite(sidecar->initial_residual_norm_l2) ||
        sidecar->initial_residual_norm_l2 < 0.0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_initial_residual_norm_l2");
        return 0;
    }
    if (!ngspice_schwarz_double_matches(
            sidecar->initial_residual_norm_atol,
            NGSPICE_SCHWARZ_INITIAL_RESIDUAL_NORM_ATOL)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_initial_residual_norm_atol_mismatch");
        return 0;
    }
    if (!ngspice_schwarz_double_matches(
            sidecar->initial_residual_norm_rtol,
            NGSPICE_SCHWARZ_INITIAL_RESIDUAL_NORM_RTOL)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_initial_residual_norm_rtol_mismatch");
        return 0;
    }
    if (sidecar->newton_iter != expected_newton_iter) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_newton_iter_mismatch");
        return 0;
    }
    if (!ngspice_schwarz_double_matches(sidecar->time_value, expected_time)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_time_mismatch");
        return 0;
    }
    if (!ngspice_schwarz_double_matches(sidecar->gmin, expected_gmin)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_gmin_mismatch");
        return 0;
    }
    if (!ngspice_schwarz_hash_equal_ci(
            sidecar->node_map_hash,
            expected_node_map_hash)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_node_map_hash_mismatch");
        return 0;
    }
    if (!ngspice_schwarz_is_lower_hex_hash(sidecar->matrix_fingerprint)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_matrix_fingerprint");
        return 0;
    }
    if (!ngspice_schwarz_is_lower_hex_hash(sidecar->layout_sha256)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_layout_sha256");
        return 0;
    }
    if (!ngspice_schwarz_is_lower_hex_hash(sidecar->checkpoint_sha256)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_checkpoint_sha256");
        return 0;
    }
    if (sidecar->block_count < 0 || sidecar->total_block_rows < 0) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_invalid_counts");
        return 0;
    }
    if (sidecar->block_offsets[0] != 0 ||
        sidecar->block_offsets[sidecar->block_count] !=
            sidecar->total_block_rows) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_invalid_block_offsets");
        return 0;
    }

    sidecar->max_block_size = 0;
    sidecar->lambda_min = 0.0;
    sidecar->lambda_max = 0.0;
    for (block_id = 0; block_id < sidecar->block_count; block_id++) {
        int begin = sidecar->block_offsets[block_id];
        int end = sidecar->block_offsets[block_id + 1];
        int block_size;
        int left;
        int right;

        if (begin < 0 || end < begin || end > sidecar->total_block_rows) {
            ngspice_schwarz_set_block_reason(
                reason,
                reason_size,
                "schwarz_invalid_block_offset",
                block_id);
            return 0;
        }
        block_size = end - begin;
        if (block_size < 2 || block_size > NGSPICE_SCHWARZ_MAX_BLOCK_SIZE) {
            ngspice_schwarz_set_block_reason(
                reason,
                reason_size,
                "schwarz_invalid_block_size",
                block_id);
            return 0;
        }
        if (block_size > sidecar->max_block_size)
            sidecar->max_block_size = block_size;
        if (!isfinite(sidecar->block_lambdas[block_id]) ||
            sidecar->block_lambdas[block_id] < 0.0) {
            ngspice_schwarz_set_block_reason(
                reason,
                reason_size,
                "schwarz_invalid_lambda",
                block_id);
            return 0;
        }
        if (block_id == 0 ||
            sidecar->block_lambdas[block_id] < sidecar->lambda_min)
            sidecar->lambda_min = sidecar->block_lambdas[block_id];
        if (block_id == 0 ||
            sidecar->block_lambdas[block_id] > sidecar->lambda_max)
            sidecar->lambda_max = sidecar->block_lambdas[block_id];

        for (left = begin; left < end; left++) {
            if (sidecar->block_rows[left] < 1 ||
                sidecar->block_rows[left] > expected_matrix_size) {
                ngspice_schwarz_set_block_reason(
                    reason,
                    reason_size,
                    "schwarz_row_out_of_range",
                    block_id);
                return 0;
            }
            if (!isfinite(sidecar->block_row_weights[left]) ||
                sidecar->block_row_weights[left] < 0.0 ||
                sidecar->block_row_weights[left] > 1.0) {
                ngspice_schwarz_set_block_reason(
                    reason,
                    reason_size,
                    "schwarz_invalid_weight",
                    block_id);
                return 0;
            }
            for (right = left + 1; right < end; right++) {
                if (sidecar->block_rows[left] ==
                    sidecar->block_rows[right]) {
                    ngspice_schwarz_set_block_reason(
                        reason,
                        reason_size,
                        "schwarz_duplicate_row",
                        block_id);
                    return 0;
                }
            }
        }
    }

    if (!ngspice_schwarz_compute_layout_sha256(
            sidecar,
            actual_layout_sha256)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_layout_sha256_compute_failed");
        return 0;
    }
    if (strcmp(actual_layout_sha256, sidecar->layout_sha256) != 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_layout_sha256_mismatch");
        return 0;
    }

    weight_sums = (double *) calloc(
        (size_t) expected_matrix_size,
        sizeof(double));
    covered = (unsigned char *) calloc(
        (size_t) expected_matrix_size,
        sizeof(unsigned char));
    if (!weight_sums || !covered) {
        free(weight_sums);
        free(covered);
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_nomem");
        return 0;
    }
    for (row = 0; row < sidecar->total_block_rows; row++) {
        int global_row = sidecar->block_rows[row] - 1;
        covered[global_row] = 1U;
        weight_sums[global_row] += sidecar->block_row_weights[row];
        if (!isfinite(weight_sums[global_row])) {
            free(weight_sums);
            free(covered);
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_weight_sum_nonfinite");
            return 0;
        }
    }
    for (row = 0; row < expected_matrix_size; row++) {
        if (covered[row] &&
            fabs(weight_sums[row] - 1.0) >
                NGSPICE_SCHWARZ_WEIGHT_SUM_TOL) {
            free(weight_sums);
            free(covered);
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_weight_sum_mismatch");
            return 0;
        }
    }
    free(weight_sums);
    free(covered);
    return 1;
}

static int
ngspice_schwarz_load_sidecar(
    const char *path,
    int expected_matrix_size,
    int expected_newton_iter,
    double expected_time,
    double expected_gmin,
    const char *expected_node_map_hash,
    const char *expected_initial_guess_mode,
    ngspice_schwarz_sidecar_t *sidecar,
    char *reason,
    size_t reason_size
)
{
    char *json = NULL;
    char mode[64];
    char uncovered_policy[32];
    int schema_version = 0;
    size_t offsets_bytes = 0U;
    size_t rows_bytes = 0U;
    size_t lambdas_bytes = 0U;
    size_t weights_bytes = 0U;

    memset(sidecar, 0, sizeof(*sidecar));
    json = ngspice_schwarz_read_file(
        path,
        &sidecar->file_bytes,
        reason,
        reason_size);
    if (!json)
        return 0;

    if (!ngspice_schwarz_json_int(
            json,
            "schema_version",
            &schema_version) ||
        schema_version != NGSPICE_SCHWARZ_SCHEMA_VERSION) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_invalid_schema_version");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "preconditioner_mode",
            mode,
            sizeof(mode)) ||
        strcmp(mode, "learned_schwarz_v1_sparse") != 0) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_invalid_mode");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "feature_contract",
            sidecar->feature_contract,
            sizeof(sidecar->feature_contract))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_feature_contract");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "local_shift_contract",
            sidecar->local_shift_contract,
            sizeof(sidecar->local_shift_contract))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_local_shift_contract");
        goto fail;
    }
    if (!ngspice_schwarz_json_double(
            json,
            "local_shift_floor_relative",
            &sidecar->local_shift_floor_relative)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_local_shift_floor_relative");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "linear_rhs_sha256",
            sidecar->linear_rhs_sha256,
            sizeof(sidecar->linear_rhs_sha256))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_linear_rhs_sha256");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "initial_guess_sha256",
            sidecar->initial_guess_sha256,
            sizeof(sidecar->initial_guess_sha256))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_initial_guess_sha256");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "initial_residual_contract",
            sidecar->initial_residual_contract,
            sizeof(sidecar->initial_residual_contract))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_initial_residual_contract");
        goto fail;
    }
    if (!ngspice_schwarz_json_double(
            json,
            "initial_residual_norm_l2",
            &sidecar->initial_residual_norm_l2)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_initial_residual_norm_l2");
        goto fail;
    }
    if (!ngspice_schwarz_json_double(
            json,
            "initial_residual_norm_atol",
            &sidecar->initial_residual_norm_atol)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_initial_residual_norm_atol");
        goto fail;
    }
    if (!ngspice_schwarz_json_double(
            json,
            "initial_residual_norm_rtol",
            &sidecar->initial_residual_norm_rtol)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_initial_residual_norm_rtol");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "initial_guess_mode",
            sidecar->initial_guess_mode,
            sizeof(sidecar->initial_guess_mode))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_initial_guess_mode");
        goto fail;
    }
    if (!ngspice_schwarz_json_int(
            json,
            "row_index_base",
            &sidecar->row_index_base)) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_missing_or_invalid_row_index_base");
        goto fail;
    }
    if (sidecar->row_index_base != 1) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_unsupported_row_index_base");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "uncovered_row_policy",
            uncovered_policy,
            sizeof(uncovered_policy)) ||
        strcmp(uncovered_policy, "row_sum") != 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_uncovered_policy");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "matrix_fingerprint",
            sidecar->matrix_fingerprint,
            sizeof(sidecar->matrix_fingerprint))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_matrix_fingerprint");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "layout_sha256",
            sidecar->layout_sha256,
            sizeof(sidecar->layout_sha256))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_layout_sha256");
        goto fail;
    }
    if (!ngspice_schwarz_json_string(
            json,
            "checkpoint_sha256",
            sidecar->checkpoint_sha256,
            sizeof(sidecar->checkpoint_sha256))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_or_invalid_checkpoint_sha256");
        goto fail;
    }
    if (!ngspice_schwarz_json_int(
            json,
            "matrix_size",
            &sidecar->matrix_size) ||
        !ngspice_schwarz_json_int(
            json,
            "newton_iter",
            &sidecar->newton_iter) ||
        !ngspice_schwarz_json_double(
            json,
            "time",
            &sidecar->time_value) ||
        !ngspice_schwarz_json_double(
            json,
            "gmin",
            &sidecar->gmin) ||
        !ngspice_schwarz_json_int(
            json,
            "block_count",
            &sidecar->block_count) ||
        !ngspice_schwarz_json_int(
            json,
            "total_block_rows",
            &sidecar->total_block_rows) ||
        !ngspice_schwarz_json_string(
            json,
            "node_map_hash",
            sidecar->node_map_hash,
            sizeof(sidecar->node_map_hash))) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_missing_required_field");
        goto fail;
    }
    if (sidecar->newton_iter < 1) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_newton_iter");
        goto fail;
    }
    if (sidecar->block_count < 0 ||
        sidecar->total_block_rows < 0 ||
        sidecar->block_count > sidecar->total_block_rows ||
        (size_t) sidecar->total_block_rows >
            (size_t) sidecar->block_count *
            (size_t) NGSPICE_SCHWARZ_MAX_BLOCK_SIZE) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_invalid_counts");
        goto fail;
    }

    if (!ngspice_schwarz_checked_bytes(
            (size_t) sidecar->block_count + 1U,
            sizeof(int),
            &offsets_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) sidecar->total_block_rows,
            sizeof(int),
            &rows_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) sidecar->block_count,
            sizeof(double),
            &lambdas_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) sidecar->total_block_rows,
            sizeof(double),
            &weights_bytes)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_size_overflow");
        goto fail;
    }

    sidecar->block_offsets = (int *) malloc(offsets_bytes);
    if (rows_bytes > 0U)
        sidecar->block_rows = (int *) malloc(rows_bytes);
    if (lambdas_bytes > 0U)
        sidecar->block_lambdas = (double *) malloc(lambdas_bytes);
    if (weights_bytes > 0U)
        sidecar->block_row_weights = (double *) malloc(weights_bytes);
    if (!sidecar->block_offsets ||
        (rows_bytes > 0U && !sidecar->block_rows) ||
        (lambdas_bytes > 0U && !sidecar->block_lambdas) ||
        (weights_bytes > 0U && !sidecar->block_row_weights)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_nomem");
        goto fail;
    }

    if (!ngspice_schwarz_json_int_array(
            json,
            "block_offsets",
            sidecar->block_offsets,
            sidecar->block_count + 1) ||
        !ngspice_schwarz_json_int_array(
            json,
            "block_rows",
            sidecar->block_rows,
            sidecar->total_block_rows) ||
        !ngspice_schwarz_json_double_array(
            json,
            "block_lambdas",
            sidecar->block_lambdas,
            sidecar->block_count) ||
        !ngspice_schwarz_json_double_array(
            json,
            "block_row_weights",
            sidecar->block_row_weights,
            sidecar->total_block_rows)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_invalid_flat_array");
        goto fail;
    }
    free(json);
    json = NULL;

    if (!ngspice_schwarz_validate_sidecar(
            sidecar,
            expected_matrix_size,
            expected_newton_iter,
            expected_time,
            expected_gmin,
            expected_node_map_hash,
            expected_initial_guess_mode,
            reason,
            reason_size))
        goto fail;
    return 1;

fail:
    free(json);
    ngspice_schwarz_sidecar_clear(sidecar);
    return 0;
}

static int
ngspice_schwarz_dense_lu_factor(
    double *matrix,
    int size,
    int *pivots
)
{
    int column;
    for (column = 0; column < size; column++) {
        int pivot = column;
        int row;
        double pivot_abs = fabs(
            matrix[(size_t) column * (size_t) size + (size_t) column]);
        if (!isfinite(pivot_abs))
            return 0;
        for (row = column + 1; row < size; row++) {
            double candidate = fabs(
                matrix[(size_t) row * (size_t) size + (size_t) column]);
            if (!isfinite(candidate))
                return 0;
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = row;
            }
        }
        if (pivot_abs <= NGSPICE_SCHWARZ_PIVOT_TOL)
            return 0;
        pivots[column] = pivot;
        if (pivot != column) {
            int swap_column;
            for (swap_column = 0; swap_column < size; swap_column++) {
                double temporary =
                    matrix[(size_t) column * (size_t) size +
                           (size_t) swap_column];
                matrix[(size_t) column * (size_t) size +
                       (size_t) swap_column] =
                    matrix[(size_t) pivot * (size_t) size +
                           (size_t) swap_column];
                matrix[(size_t) pivot * (size_t) size +
                       (size_t) swap_column] = temporary;
            }
        }
        {
            double diagonal =
                matrix[(size_t) column * (size_t) size + (size_t) column];
            if (!isfinite(diagonal) ||
                fabs(diagonal) <= NGSPICE_SCHWARZ_PIVOT_TOL)
                return 0;
            for (row = column + 1; row < size; row++) {
                int trailing_column;
                double multiplier;
                size_t lower_index =
                    (size_t) row * (size_t) size + (size_t) column;
                matrix[lower_index] /= diagonal;
                multiplier = matrix[lower_index];
                if (!isfinite(multiplier))
                    return 0;
                for (trailing_column = column + 1;
                     trailing_column < size;
                     trailing_column++) {
                    size_t target =
                        (size_t) row * (size_t) size +
                        (size_t) trailing_column;
                    matrix[target] -=
                        multiplier *
                        matrix[(size_t) column * (size_t) size +
                               (size_t) trailing_column];
                    if (!isfinite(matrix[target]))
                        return 0;
                }
            }
        }
    }
    for (column = 0; column < size; column++) {
        double diagonal =
            matrix[(size_t) column * (size_t) size + (size_t) column];
        if (!isfinite(diagonal) ||
            fabs(diagonal) <= NGSPICE_SCHWARZ_PIVOT_TOL)
            return 0;
    }
    return 1;
}

static int
ngspice_schwarz_dense_lu_solve(
    const double *lu,
    const int *pivots,
    double *values,
    int size
)
{
    int row;
    for (row = 0; row < size; row++) {
        int pivot = pivots[row];
        if (pivot < row || pivot >= size)
            return 0;
        if (pivot != row) {
            double temporary = values[row];
            values[row] = values[pivot];
            values[pivot] = temporary;
        }
    }
    for (row = 1; row < size; row++) {
        int column;
        double value = values[row];
        for (column = 0; column < row; column++)
            value -=
                lu[(size_t) row * (size_t) size + (size_t) column] *
                values[column];
        if (!isfinite(value))
            return 0;
        values[row] = value;
    }
    for (row = size - 1; row >= 0; row--) {
        int column;
        double value = values[row];
        double diagonal =
            lu[(size_t) row * (size_t) size + (size_t) row];
        for (column = row + 1; column < size; column++)
            value -=
                lu[(size_t) row * (size_t) size + (size_t) column] *
                values[column];
        if (!isfinite(value) || !isfinite(diagonal) ||
            fabs(diagonal) <= NGSPICE_SCHWARZ_PIVOT_TOL)
            return 0;
        values[row] = value / diagonal;
        if (!isfinite(values[row]))
            return 0;
    }
    return 1;
}

static int
ngspice_schwarz_find_local_row(
    const int *rows,
    int size,
    int external_row
)
{
    int index;
    for (index = 0; index < size; index++) {
        if (rows[index] == external_row)
            return index;
    }
    return -1;
}

static int
ngspice_schwarz_prepare_matrix_rows(
    CKTcircuit *ckt,
    int matrix_size,
    char *reason,
    size_t reason_size
)
{
    MatrixFrame *matrix;
    if (!ckt || !ckt->CKTmatrix || !ckt->CKTmatrix->SPmatrix) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_bad_matrix");
        return 0;
    }
    if (SMPmatSize(ckt->CKTmatrix) != matrix_size) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_live_matrix_size_mismatch");
        return 0;
    }
    matrix = ckt->CKTmatrix->SPmatrix;
    if (matrix->Factored) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_matrix_already_factored");
        return 0;
    }
    if (matrix->Complex) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_complex_matrix_unsupported");
        return 0;
    }
    if (!matrix->ExtToIntRowMap || !matrix->IntToExtColMap ||
        !matrix->FirstInRow) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_maps_missing");
        return 0;
    }
    if (!matrix->RowsLinked)
        spcLinkRows(matrix);
    if (!matrix->RowsLinked) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_rows_unlinked");
        return 0;
    }
    return 1;
}

static int
ngspice_schwarz_compare_row_entries(
    const void *left_value,
    const void *right_value
)
{
    const ngspice_schwarz_row_entry_t *left =
        (const ngspice_schwarz_row_entry_t *) left_value;
    const ngspice_schwarz_row_entry_t *right =
        (const ngspice_schwarz_row_entry_t *) right_value;
    if (left->column < right->column)
        return -1;
    if (left->column > right->column)
        return 1;
    return 0;
}

static int
ngspice_schwarz_max_live_row_entries(
    MatrixFrame *matrix,
    int matrix_size,
    size_t *capacity,
    char *reason,
    size_t reason_size
)
{
    size_t maximum = 0U;
    int external_row;

    if (!matrix || !capacity || matrix_size <= 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_matrix_fingerprint_bad_argument");
        return 0;
    }
    for (external_row = 1; external_row <= matrix_size; external_row++) {
        int internal_row = matrix->ExtToIntRowMap[external_row];
        ElementPtr element;
        size_t count = 0U;
        if (internal_row <= 0 || internal_row > matrix->Size) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_invalid_row_map");
            return 0;
        }
        for (element = matrix->FirstInRow[internal_row];
             element;
             element = element->NextInRow) {
            if (count == SIZE_MAX) {
                ngspice_schwarz_set_reason(
                    reason,
                    reason_size,
                    "schwarz_matrix_fingerprint_size_overflow");
                return 0;
            }
            count++;
        }
        if (count > maximum)
            maximum = count;
    }
    if (maximum == SIZE_MAX) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_fingerprint_size_overflow");
        return 0;
    }
    *capacity = maximum + 1U;
    return 1;
}

static int
ngspice_schwarz_canonicalize_live_row(
    MatrixFrame *matrix,
    int matrix_size,
    int external_row,
    double gmin,
    ngspice_schwarz_row_entry_t *entries,
    size_t capacity,
    size_t *canonical_count,
    char *reason,
    size_t reason_size
)
{
    int internal_row;
    ElementPtr element;
    size_t raw_count = 0U;
    size_t read_index = 0U;
    size_t write_index = 0U;
    int diagonal_found = 0;

    if (!matrix || !entries || !canonical_count ||
        external_row < 1 || external_row > matrix_size) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_matrix_fingerprint_bad_argument");
        return 0;
    }
    internal_row = matrix->ExtToIntRowMap[external_row];
    if (internal_row <= 0 || internal_row > matrix->Size) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_fingerprint_invalid_row_map");
        return 0;
    }

    for (element = matrix->FirstInRow[internal_row];
         element;
         element = element->NextInRow) {
        int external_column;
        if (raw_count >= capacity) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_row_overflow");
            return 0;
        }
        external_column = matrix->IntToExtColMap[element->Col];
        if (external_column < 1 || external_column > matrix_size) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_invalid_column_map");
            return 0;
        }
        if (!isfinite(element->Real)) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_nonfinite_value");
            return 0;
        }
        entries[raw_count].column = external_column;
        entries[raw_count].value = element->Real;
        raw_count++;
    }
    qsort(
        entries,
        raw_count,
        sizeof(ngspice_schwarz_row_entry_t),
        ngspice_schwarz_compare_row_entries);

    while (read_index < raw_count) {
        int column = entries[read_index].column;
        double combined = entries[read_index].value;
        read_index++;
        while (read_index < raw_count &&
               entries[read_index].column == column) {
            combined += entries[read_index].value;
            read_index++;
            if (!isfinite(combined)) {
                ngspice_schwarz_set_reason(
                    reason,
                    reason_size,
                    "schwarz_matrix_fingerprint_nonfinite_sum");
                return 0;
            }
        }
        if (column == external_row) {
            diagonal_found = 1;
            combined += gmin;
            if (!isfinite(combined)) {
                ngspice_schwarz_set_reason(
                    reason,
                    reason_size,
                    "schwarz_matrix_fingerprint_nonfinite_diagonal");
                return 0;
            }
        }
        if (combined != 0.0) {
            entries[write_index].column = column;
            entries[write_index].value = combined;
            write_index++;
        }
    }

    if (!diagonal_found && gmin != 0.0) {
        size_t insertion = 0U;
        if (write_index >= capacity) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_row_overflow");
            return 0;
        }
        while (insertion < write_index &&
               entries[insertion].column < external_row)
            insertion++;
        memmove(
            entries + insertion + 1U,
            entries + insertion,
            (write_index - insertion) * sizeof(*entries));
        entries[insertion].column = external_row;
        entries[insertion].value = gmin;
        write_index++;
    }

    *canonical_count = write_index;
    return 1;
}

static int
ngspice_schwarz_compute_live_matrix_fingerprint(
    CKTcircuit *ckt,
    int matrix_size,
    char output[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1],
    size_t *temporary_bytes,
    char *reason,
    size_t reason_size
)
{
    static const char header_format[] =
        "schema=pals_csr_f64_v1\n"
        "rows=%d\n"
        "cols=%d\n"
        "nnz=%llu\n";
    MatrixFrame *matrix;
    ngspice_schwarz_row_entry_t *entries = NULL;
    uint64_t *row_counts = NULL;
    ngspice_gmres_sha256_t context;
    unsigned char digest[32];
    char header[160];
    size_t entry_capacity = 0U;
    size_t entry_bytes = 0U;
    size_t row_counts_bytes = 0U;
    size_t fingerprint_temporary_bytes = 0U;
    uint64_t total_nnz = UINT64_C(0);
    uint64_t cumulative = UINT64_C(0);
    int external_row;
    int written;
    int ok = 0;

    if (temporary_bytes)
        *temporary_bytes = 0U;
    if (!ckt || !ckt->CKTmatrix || !ckt->CKTmatrix->SPmatrix ||
        !output || !temporary_bytes || matrix_size <= 0 ||
        sizeof(double) != 8U) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_matrix_fingerprint_bad_argument");
        return 0;
    }
    matrix = ckt->CKTmatrix->SPmatrix;
    if (matrix->Factored) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_fingerprint_matrix_factored");
        return 0;
    }
    if (!ngspice_schwarz_max_live_row_entries(
            matrix,
            matrix_size,
            &entry_capacity,
            reason,
            reason_size))
        return 0;
    if (!ngspice_schwarz_checked_bytes(
            entry_capacity,
            sizeof(*entries),
            &entry_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) matrix_size,
            sizeof(*row_counts),
            &row_counts_bytes) ||
        !ngspice_schwarz_checked_add(
            entry_bytes,
            row_counts_bytes,
            &fingerprint_temporary_bytes)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_fingerprint_size_overflow");
        return 0;
    }
    *temporary_bytes = fingerprint_temporary_bytes;

    entries = (ngspice_schwarz_row_entry_t *) malloc(entry_bytes);
    row_counts = (uint64_t *) calloc(
        (size_t) matrix_size,
        sizeof(*row_counts));
    if (!entries || !row_counts) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_matrix_fingerprint_nomem");
        goto cleanup;
    }

    for (external_row = 1; external_row <= matrix_size; external_row++) {
        size_t count = 0U;
        if (!ngspice_schwarz_canonicalize_live_row(
                matrix,
                matrix_size,
                external_row,
                ckt->CKTdiagGmin,
                entries,
                entry_capacity,
                &count,
                reason,
                reason_size))
            goto cleanup;
        if ((uint64_t) count > UINT64_MAX - total_nnz) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_size_overflow");
            goto cleanup;
        }
        row_counts[external_row - 1] = (uint64_t) count;
        total_nnz += (uint64_t) count;
    }
    if (total_nnz > (uint64_t) INT64_MAX) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_fingerprint_size_overflow");
        goto cleanup;
    }

    written = snprintf(
        header,
        sizeof(header),
        header_format,
        matrix_size,
        matrix_size,
        (unsigned long long) total_nnz);
    if (written <= 0 || (size_t) written >= sizeof(header)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_matrix_fingerprint_header_failed");
        goto cleanup;
    }

    ngspice_gmres_sha256_init(&context);
    ngspice_gmres_sha256_update(&context, header, (size_t) written);
    ngspice_gmres_sha256_update_i64_le(&context, INT64_C(0));
    for (external_row = 1; external_row <= matrix_size; external_row++) {
        cumulative += row_counts[external_row - 1];
        ngspice_gmres_sha256_update_i64_le(
            &context,
            (int64_t) cumulative);
    }

    for (external_row = 1; external_row <= matrix_size; external_row++) {
        size_t count = 0U;
        size_t index;
        if (!ngspice_schwarz_canonicalize_live_row(
                matrix,
                matrix_size,
                external_row,
                ckt->CKTdiagGmin,
                entries,
                entry_capacity,
                &count,
                reason,
                reason_size))
            goto cleanup;
        if ((uint64_t) count != row_counts[external_row - 1]) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_live_matrix_changed");
            goto cleanup;
        }
        for (index = 0U; index < count; index++) {
            ngspice_gmres_sha256_update_i64_le(
                &context,
                (int64_t) entries[index].column - INT64_C(1));
        }
    }

    for (external_row = 1; external_row <= matrix_size; external_row++) {
        size_t count = 0U;
        size_t index;
        if (!ngspice_schwarz_canonicalize_live_row(
                matrix,
                matrix_size,
                external_row,
                ckt->CKTdiagGmin,
                entries,
                entry_capacity,
                &count,
                reason,
                reason_size))
            goto cleanup;
        if ((uint64_t) count != row_counts[external_row - 1]) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_matrix_fingerprint_live_matrix_changed");
            goto cleanup;
        }
        for (index = 0U; index < count; index++)
            ngspice_gmres_sha256_update_f64_le(
                &context,
                entries[index].value);
    }

    ngspice_gmres_sha256_final(&context, digest);
    ngspice_gmres_sha256_hex(digest, output);
    ok = 1;

cleanup:
    free(entries);
    free(row_counts);
    return ok;
}

static int
ngspice_schwarz_compute_fallback(
    CKTcircuit *ckt,
    int matrix_size,
    double *fallback_scales,
    char *reason,
    size_t reason_size
)
{
    MatrixFrame *matrix = ckt->CKTmatrix->SPmatrix;
    int external_row;

    for (external_row = 1; external_row <= matrix_size; external_row++) {
        int internal_row = matrix->ExtToIntRowMap[external_row];
        ElementPtr element;
        double row_sum = 0.0;
        double raw_diagonal = 0.0;
        int diagonal_found = 0;

        if (internal_row <= 0 || internal_row > matrix->Size) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_invalid_row_map");
            return 0;
        }
        for (element = matrix->FirstInRow[internal_row];
             element;
             element = element->NextInRow) {
            int external_column = matrix->IntToExtColMap[element->Col];
            if (!isfinite(element->Real)) {
                ngspice_schwarz_set_reason(
                    reason,
                    reason_size,
                    "schwarz_nonfinite_matrix");
                return 0;
            }
            if (external_column == external_row) {
                diagonal_found = 1;
                raw_diagonal = element->Real;
            } else {
                row_sum += fabs(element->Real);
                if (!isfinite(row_sum)) {
                    ngspice_schwarz_set_reason(
                        reason,
                        reason_size,
                        "schwarz_invalid_row_sum");
                    return 0;
                }
            }
        }
        if (diagonal_found)
            row_sum += fabs(raw_diagonal + ckt->CKTdiagGmin);
        else
            row_sum += fabs(ckt->CKTdiagGmin);
        if (!isfinite(row_sum) || row_sum < 0.0) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_invalid_row_sum");
            return 0;
        }
        fallback_scales[external_row - 1] =
            1.0 / fmax(row_sum, 1e-30);
    }
    return 1;
}

static int
ngspice_schwarz_extract_and_factor(
    CKTcircuit *ckt,
    const ngspice_schwarz_sidecar_t *sidecar,
    ngspice_gmres_schwarz_state_t *state,
    char *reason,
    size_t reason_size
)
{
    MatrixFrame *matrix = ckt->CKTmatrix->SPmatrix;
    int block_id;

    for (block_id = 0; block_id < sidecar->block_count; block_id++) {
        int begin = sidecar->block_offsets[block_id];
        int end = sidecar->block_offsets[block_id + 1];
        int block_size = end - begin;
        const int *rows = sidecar->block_rows + begin;
        double *local = state->factors + state->factor_offsets[block_id];
        int local_row;

        memset(
            local,
            0,
            (size_t) block_size * (size_t) block_size * sizeof(double));
        for (local_row = 0; local_row < block_size; local_row++) {
            int external_row = rows[local_row];
            int internal_row = matrix->ExtToIntRowMap[external_row];
            ElementPtr element;

            if (internal_row <= 0 || internal_row > matrix->Size) {
                ngspice_schwarz_set_block_reason(
                    reason,
                    reason_size,
                    "schwarz_invalid_row_map_block",
                    block_id);
                return 0;
            }
            for (element = matrix->FirstInRow[internal_row];
                 element;
                 element = element->NextInRow) {
                int external_column = matrix->IntToExtColMap[element->Col];
                int local_column = ngspice_schwarz_find_local_row(
                    rows,
                    block_size,
                    external_column);
                if (local_column >= 0) {
                    if (!isfinite(element->Real)) {
                        ngspice_schwarz_set_block_reason(
                            reason,
                            reason_size,
                            "schwarz_nonfinite_block",
                            block_id);
                        return 0;
                    }
                    local[(size_t) local_row * (size_t) block_size +
                          (size_t) local_column] = element->Real;
                }
            }
        }
        for (local_row = 0; local_row < block_size; local_row++) {
            size_t diagonal =
                (size_t) local_row * (size_t) block_size +
                (size_t) local_row;
            local[diagonal] +=
                ckt->CKTdiagGmin + sidecar->block_lambdas[block_id];
            if (!isfinite(local[diagonal])) {
                ngspice_schwarz_set_block_reason(
                    reason,
                    reason_size,
                    "schwarz_nonfinite_shifted_block",
                    block_id);
                return 0;
            }
        }
        if (!ngspice_schwarz_dense_lu_factor(
                local,
                block_size,
                state->pivots + begin)) {
            ngspice_schwarz_set_block_reason(
                reason,
                reason_size,
                "schwarz_lu_failed_block",
                block_id);
            return 0;
        }
    }
    return 1;
}

static int
ngspice_schwarz_allocate_state(
    const ngspice_schwarz_sidecar_t *sidecar,
    ngspice_gmres_schwarz_state_t **out,
    char *reason,
    size_t reason_size
)
{
    ngspice_gmres_schwarz_state_t *state = NULL;
    size_t factor_value_count = 0U;
    size_t factor_bytes = 0U;
    size_t offsets_bytes = 0U;
    size_t rows_bytes = 0U;
    size_t weights_bytes = 0U;
    size_t factor_offsets_bytes = 0U;
    size_t pivots_bytes = 0U;
    size_t covered_bytes = 0U;
    size_t fallback_bytes = 0U;
    size_t scratch_bytes = 0U;
    int block_id;

    state = (ngspice_gmres_schwarz_state_t *) calloc(1U, sizeof(*state));
    if (!state) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_nomem");
        return 0;
    }
    state->matrix_size = sidecar->matrix_size;
    state->block_count = sidecar->block_count;
    state->total_block_rows = sidecar->total_block_rows;
    state->max_block_size = sidecar->max_block_size;

    if (!ngspice_schwarz_checked_bytes(
            (size_t) state->block_count + 1U,
            sizeof(int),
            &offsets_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) state->total_block_rows,
            sizeof(int),
            &rows_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) state->total_block_rows,
            sizeof(double),
            &weights_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) state->block_count + 1U,
            sizeof(size_t),
            &factor_offsets_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) state->total_block_rows,
            sizeof(int),
            &pivots_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) state->matrix_size,
            sizeof(unsigned char),
            &covered_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) state->matrix_size,
            sizeof(double),
            &fallback_bytes) ||
        !ngspice_schwarz_checked_bytes(
            (size_t) state->max_block_size,
            sizeof(double),
            &scratch_bytes)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_size_overflow");
        goto fail;
    }

    state->factor_offsets = (size_t *) malloc(factor_offsets_bytes);
    if (!state->factor_offsets) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_nomem");
        goto fail;
    }
    state->factor_offsets[0] = 0U;
    for (block_id = 0; block_id < state->block_count; block_id++) {
        size_t block_size = (size_t) (
            sidecar->block_offsets[block_id + 1] -
            sidecar->block_offsets[block_id]);
        size_t block_values;

        if (block_size > 0U && block_size > SIZE_MAX / block_size) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_factor_size_overflow");
            goto fail;
        }
        block_values = block_size * block_size;
        if (!ngspice_schwarz_checked_add(
                factor_value_count,
                block_values,
                &factor_value_count)) {
            ngspice_schwarz_set_reason(
                reason,
                reason_size,
                "schwarz_factor_size_overflow");
            goto fail;
        }
        state->factor_offsets[block_id + 1] = factor_value_count;
    }
    if (!ngspice_schwarz_checked_bytes(
            factor_value_count,
            sizeof(double),
            &factor_bytes)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_factor_size_overflow");
        goto fail;
    }

    state->block_offsets = (int *) malloc(offsets_bytes);
    if (rows_bytes > 0U)
        state->block_rows = (int *) malloc(rows_bytes);
    if (weights_bytes > 0U)
        state->block_row_weights = (double *) malloc(weights_bytes);
    if (factor_bytes > 0U)
        state->factors = (double *) malloc(factor_bytes);
    if (pivots_bytes > 0U)
        state->pivots = (int *) malloc(pivots_bytes);
    state->covered = (unsigned char *) calloc(
        (size_t) state->matrix_size,
        sizeof(unsigned char));
    state->fallback_scales = (double *) malloc(fallback_bytes);
    if (scratch_bytes > 0U)
        state->scratch = (double *) malloc(scratch_bytes);
    if (!state->block_offsets ||
        (rows_bytes > 0U && !state->block_rows) ||
        (weights_bytes > 0U && !state->block_row_weights) ||
        (factor_bytes > 0U && !state->factors) ||
        (pivots_bytes > 0U && !state->pivots) ||
        !state->covered ||
        !state->fallback_scales ||
        (scratch_bytes > 0U && !state->scratch)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_nomem");
        goto fail;
    }

    memcpy(state->block_offsets, sidecar->block_offsets, offsets_bytes);
    if (rows_bytes > 0U)
        memcpy(state->block_rows, sidecar->block_rows, rows_bytes);
    if (weights_bytes > 0U) {
        memcpy(
            state->block_row_weights,
            sidecar->block_row_weights,
            weights_bytes);
    }
    for (block_id = 0; block_id < state->total_block_rows; block_id++)
        state->covered[state->block_rows[block_id] - 1] = 1U;

    state->metrics.layout_bytes =
        offsets_bytes + rows_bytes + factor_offsets_bytes + covered_bytes;
    state->metrics.parameter_bytes = weights_bytes;
    state->metrics.factor_bytes = factor_bytes + pivots_bytes;
    state->metrics.fallback_bytes = fallback_bytes;
    state->metrics.workspace_bytes = scratch_bytes;
    state->metrics.retained_bytes =
        sizeof(*state) +
        state->metrics.layout_bytes +
        state->metrics.parameter_bytes +
        state->metrics.factor_bytes +
        state->metrics.fallback_bytes +
        state->metrics.workspace_bytes;
    *out = state;
    return 1;

fail:
    ngspice_gmres_schwarz_destroy(state);
    return 0;
}

int
ngspice_gmres_schwarz_create(
    CKTcircuit *ckt,
    const char *sidecar_path,
    int matrix_size,
    int newton_iter,
    const char *node_map_hash,
    const double *linear_rhs,
    const double *initial_guess,
    const double *initial_residual,
    const char *initial_guess_mode,
    ngspice_gmres_schwarz_state_t **out,
    char *reason,
    size_t reason_size
)
{
    char actual_matrix_fingerprint[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char actual_linear_rhs_sha256[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    char actual_initial_guess_sha256[NGSPICE_SCHWARZ_HASH_HEX_LENGTH + 1];
    double actual_initial_residual_norm_l2;
    ngspice_schwarz_sidecar_t sidecar;
    ngspice_gmres_schwarz_state_t *state = NULL;
    double setup_started = ngspice_schwarz_now();
    double load_started;
    double factor_started;
    size_t temporary_bytes = 0U;
    size_t fingerprint_temporary_bytes = 0U;
    size_t item_bytes = 0U;
    int covered_rows = 0;
    int row;

    memset(&sidecar, 0, sizeof(sidecar));
    if (out)
        *out = NULL;
    ngspice_schwarz_set_reason(reason, reason_size, "");
    if (newton_iter < 1) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_invalid_newton_iter");
        return 0;
    }
    if (!ngspice_schwarz_initial_guess_mode_valid(initial_guess_mode)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_invalid_actual_initial_guess_mode");
        return 0;
    }
    if (!ckt || !out || matrix_size <= 0 || !node_map_hash ||
        !ngspice_schwarz_is_hex_hash(node_map_hash) || !linear_rhs ||
        !initial_guess || !initial_residual ||
        !isfinite(ckt->CKTdiagGmin) || !isfinite(ckt->CKTtime)) {
        ngspice_schwarz_set_reason(reason, reason_size, "schwarz_bad_argument");
        return 0;
    }
    if (!ngspice_schwarz_prepare_matrix_rows(
            ckt,
            matrix_size,
            reason,
            reason_size))
        return 0;

    load_started = ngspice_schwarz_now();
    if (!ngspice_schwarz_load_sidecar(
            sidecar_path,
            matrix_size,
            newton_iter,
            ckt->CKTtime,
            ckt->CKTdiagGmin,
            node_map_hash,
            initial_guess_mode,
            &sidecar,
            reason,
            reason_size))
        return 0;

    if (!ngspice_schwarz_compute_vector_sha256(
            linear_rhs,
            matrix_size,
            actual_linear_rhs_sha256,
            "schwarz_linear_rhs_nonfinite",
            reason,
            reason_size))
        goto fail;
    if (strcmp(
            actual_linear_rhs_sha256,
            sidecar.linear_rhs_sha256) != 0) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_linear_rhs_sha256_mismatch");
        goto fail;
    }
    if (!ngspice_schwarz_compute_vector_sha256(
            initial_guess,
            matrix_size,
            actual_initial_guess_sha256,
            "schwarz_initial_guess_nonfinite",
            reason,
            reason_size))
        goto fail;
    if (strcmp(
            actual_initial_guess_sha256,
            sidecar.initial_guess_sha256) != 0) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_initial_guess_sha256_mismatch");
        goto fail;
    }
    if (!ngspice_schwarz_l2_norm(
            initial_residual,
            matrix_size,
            &actual_initial_residual_norm_l2)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_initial_residual_nonfinite");
        goto fail;
    }
    if (!ngspice_schwarz_initial_residual_norm_matches(
            actual_initial_residual_norm_l2,
            sidecar.initial_residual_norm_l2,
            sidecar.initial_residual_norm_atol,
            sidecar.initial_residual_norm_rtol)) {
        ngspice_schwarz_set_reason(
            reason,
            reason_size,
            "schwarz_initial_residual_norm_mismatch");
        goto fail;
    }
    if (!ngspice_schwarz_compute_live_matrix_fingerprint(
            ckt,
            matrix_size,
            actual_matrix_fingerprint,
            &fingerprint_temporary_bytes,
            reason,
            reason_size))
        goto fail;
    if (strcmp(
            actual_matrix_fingerprint,
            sidecar.matrix_fingerprint) != 0) {
        ngspice_schwarz_set_reason(
            reason, reason_size, "schwarz_matrix_fingerprint_mismatch");
        goto fail;
    }

    if (!ngspice_schwarz_allocate_state(
            &sidecar,
            &state,
            reason,
            reason_size))
        goto fail;
    state->metrics.matrix_size = matrix_size;
    state->metrics.block_count = sidecar.block_count;
    state->metrics.total_block_rows = sidecar.total_block_rows;
    state->metrics.max_block_size = sidecar.max_block_size;
    state->metrics.gmin = ckt->CKTdiagGmin;
    state->metrics.lambda_min = sidecar.lambda_min;
    state->metrics.lambda_max = sidecar.lambda_max;
    state->metrics.sidecar_load_time =
        ngspice_schwarz_now() - load_started;
    state->metrics.sidecar_file_bytes = sidecar.file_bytes;

    for (row = 0; row < matrix_size; row++) {
        if (state->covered[row])
            covered_rows++;
    }
    state->metrics.covered_rows = covered_rows;
    state->metrics.uncovered_rows = matrix_size - covered_rows;

    factor_started = ngspice_schwarz_now();
    if (!ngspice_schwarz_compute_fallback(
            ckt,
            matrix_size,
            state->fallback_scales,
            reason,
            reason_size) ||
        !ngspice_schwarz_extract_and_factor(
            ckt,
            &sidecar,
            state,
            reason,
            reason_size))
        goto fail;
    state->metrics.factor_time = ngspice_schwarz_now() - factor_started;
    state->metrics.setup_time = ngspice_schwarz_now() - setup_started;

    temporary_bytes = sidecar.file_bytes;
    if (ngspice_schwarz_checked_bytes(
            (size_t) sidecar.block_count + 1U,
            sizeof(int),
            &item_bytes) &&
        ngspice_schwarz_checked_add(
            temporary_bytes,
            item_bytes,
            &temporary_bytes) &&
        ngspice_schwarz_checked_bytes(
            (size_t) sidecar.total_block_rows,
            sizeof(int),
            &item_bytes) &&
        ngspice_schwarz_checked_add(
            temporary_bytes,
            item_bytes,
            &temporary_bytes) &&
        ngspice_schwarz_checked_bytes(
            (size_t) sidecar.block_count,
            sizeof(double),
            &item_bytes) &&
        ngspice_schwarz_checked_add(
            temporary_bytes,
            item_bytes,
            &temporary_bytes) &&
        ngspice_schwarz_checked_bytes(
            (size_t) sidecar.total_block_rows,
            sizeof(double),
            &item_bytes) &&
        ngspice_schwarz_checked_add(
            temporary_bytes,
            item_bytes,
            &temporary_bytes) &&
        ngspice_schwarz_checked_bytes(
            (size_t) matrix_size,
            sizeof(double) + sizeof(unsigned char),
            &item_bytes) &&
        ngspice_schwarz_checked_add(
            temporary_bytes,
            item_bytes,
            &temporary_bytes) &&
        ngspice_schwarz_checked_add(
            temporary_bytes,
            fingerprint_temporary_bytes,
            &temporary_bytes) &&
        ngspice_schwarz_checked_add(
            state->metrics.retained_bytes,
            temporary_bytes,
            &state->metrics.peak_estimated_bytes)) {
        /* Conservative setup peak estimate completed. */
    } else {
        state->metrics.peak_estimated_bytes = SIZE_MAX;
    }

    ngspice_schwarz_sidecar_clear(&sidecar);
    *out = state;
    ngspice_schwarz_set_reason(reason, reason_size, "");
    return 1;

fail:
    ngspice_schwarz_sidecar_clear(&sidecar);
    ngspice_gmres_schwarz_destroy(state);
    return 0;
}

int
ngspice_gmres_schwarz_apply(
    ngspice_gmres_schwarz_state_t *state,
    const double *rhs,
    double *out,
    int count
)
{
    double started = ngspice_schwarz_now();
    int block_id;
    int row;
    int ok = 1;

    if (!state || !rhs || !out || count != state->matrix_size)
        return 0;
    memset(out, 0, (size_t) count * sizeof(double));
    state->metrics.apply_count++;

    for (block_id = 0; block_id < state->block_count; block_id++) {
        int begin = state->block_offsets[block_id];
        int end = state->block_offsets[block_id + 1];
        int block_size = end - begin;
        const double *factor =
            state->factors + state->factor_offsets[block_id];
        int local_row;

        for (local_row = 0; local_row < block_size; local_row++) {
            double value = rhs[state->block_rows[begin + local_row] - 1];
            if (!isfinite(value)) {
                ok = 0;
                break;
            }
            state->scratch[local_row] = value;
        }
        if (!ok ||
            !ngspice_schwarz_dense_lu_solve(
                factor,
                state->pivots + begin,
                state->scratch,
                block_size)) {
            ok = 0;
            break;
        }
        for (local_row = 0; local_row < block_size; local_row++) {
            int global_row = state->block_rows[begin + local_row] - 1;
            double contribution =
                state->block_row_weights[begin + local_row] *
                state->scratch[local_row];
            if (!isfinite(contribution)) {
                ok = 0;
                break;
            }
            out[global_row] += contribution;
            if (!isfinite(out[global_row])) {
                ok = 0;
                break;
            }
        }
        if (!ok)
            break;
    }

    if (ok) {
        for (row = 0; row < count; row++) {
            if (!state->covered[row]) {
                if (!isfinite(rhs[row])) {
                    ok = 0;
                    break;
                }
                out[row] = state->fallback_scales[row] * rhs[row];
                if (!isfinite(out[row])) {
                    ok = 0;
                    break;
                }
            }
        }
    }
    if (!ok) {
        memset(out, 0, (size_t) count * sizeof(double));
        state->metrics.failed_apply_count++;
    }
    state->metrics.apply_time_total += ngspice_schwarz_now() - started;
    return ok;
}

void
ngspice_gmres_schwarz_get_metrics(
    const ngspice_gmres_schwarz_state_t *state,
    ngspice_gmres_schwarz_metrics_t *out
)
{
    if (!out)
        return;
    if (!state) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = state->metrics;
}

void
ngspice_gmres_schwarz_destroy(
    ngspice_gmres_schwarz_state_t *state
)
{
    if (!state)
        return;
    free(state->block_offsets);
    free(state->block_rows);
    free(state->block_row_weights);
    free(state->factor_offsets);
    free(state->factors);
    free(state->pivots);
    free(state->covered);
    free(state->fallback_scales);
    free(state->scratch);
    free(state);
}
