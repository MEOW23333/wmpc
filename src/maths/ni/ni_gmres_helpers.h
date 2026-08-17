#ifndef NGSPICE_NI_GMRES_HELPERS_H
#define NGSPICE_NI_GMRES_HELPERS_H

#include "ngspice/cktdefs.h"
#include "ngspice/memory.h"
#include "ngspice/smpdefs.h"
#include "ngspice/spmatrix.h"
#include "../sparse/spdefs.h"
#include "ni_gmres_schwarz.h"

#ifndef FREE
#define FREE(x) do { if (x) { txfree(x); (x) = NULL; } } while (0)
#endif

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <sys/stat.h>

#define NGSPICE_GMRES_PATH_MAX 1024
#define NGSPICE_GMRES_REASON_MAX 128
#define NGSPICE_GMRES_NAME_MAX 32
#define NGSPICE_GMRES_SCOPE_MAX 64

typedef enum {
    NGSPICE_GMRES_PRECOND_IDENTITY = 0,
    NGSPICE_GMRES_PRECOND_JACOBI = 1,
    NGSPICE_GMRES_PRECOND_ROW_SUM = 2,
    NGSPICE_GMRES_PRECOND_LEARNED_DIAGONAL = 3,
    NGSPICE_GMRES_PRECOND_LEARNED_SCHWARZ = 4
} ngspice_gmres_precond_mode_t;

typedef struct {
    int enabled;
    int restart;
    int max_iters;
    double rtol;
    double atol;
    int use_rhsold_x0;
    ngspice_gmres_precond_mode_t requested_precond;
    ngspice_gmres_precond_mode_t fallback_precond;
    char sidecar_scope[NGSPICE_GMRES_SCOPE_MAX];
    char sidecar_path[NGSPICE_GMRES_PATH_MAX];
} ngspice_gmres_config_t;

typedef struct {
    int success;
    int converged;
    int iterations;
    int restart_count;
    double rhs_norm;
    double initial_raw_residual;
    double final_raw_residual;
    double initial_true_relative_residual;
    double final_true_relative_residual;
    double initial_precond_residual;
    double final_precond_residual;
    double sidecar_load_time;
    double preconditioner_setup_time;
    double preconditioner_factor_time;
    double preconditioner_apply_time;
    int preconditioner_apply_count;
    int preconditioner_failed_apply_count;
    int preconditioner_block_count;
    int preconditioner_total_block_rows;
    int preconditioner_covered_rows;
    int preconditioner_uncovered_rows;
    int preconditioner_max_block_size;
    size_t preconditioner_sidecar_file_bytes;
    size_t preconditioner_layout_bytes;
    size_t preconditioner_parameter_bytes;
    size_t preconditioner_retained_bytes;
    size_t preconditioner_peak_estimated_bytes;
    size_t preconditioner_factor_bytes;
    size_t preconditioner_fallback_bytes;
    size_t preconditioner_workspace_bytes;
    size_t gmres_workspace_bytes;
    char executed_precond[NGSPICE_GMRES_NAME_MAX];
    char fallback_reason[NGSPICE_GMRES_REASON_MAX];
    char resolved_sidecar_path[NGSPICE_GMRES_PATH_MAX];
    int online_sidecar_enabled;
    int online_sidecar_success;
    int online_sidecar_exit_code;
    int online_sidecar_timed_out;
    double online_sidecar_snapshot_seconds;
    double online_sidecar_generation_seconds;
    size_t online_sidecar_bytes;
    char online_sidecar_failure_reason[NGSPICE_GMRES_REASON_MAX];
    char online_sidecar_input_path[NGSPICE_GMRES_PATH_MAX];
    char online_sidecar_jacobian_path[NGSPICE_GMRES_PATH_MAX];
    char online_sidecar_output_path[NGSPICE_GMRES_PATH_MAX];
    char online_sidecar_status_path[NGSPICE_GMRES_PATH_MAX];
} ngspice_gmres_result_t;

typedef struct {
    int valid;
    int matrix_size;
    off_t file_size;
    time_t file_mtime;
    char sidecar_path[NGSPICE_GMRES_PATH_MAX];
    char node_map_hash[65];
    double *scales;
} ngspice_gmres_sidecar_cache_t;

typedef struct {
    CKTcircuit *ckt;
    int size;
    double diag_gmin;
    double *in_vec;
    double *out_vec;
} ngspice_gmres_matvec_ctx_t;

typedef struct {
    ngspice_gmres_precond_mode_t mode;
    double *scales;
    int count;
    ngspice_gmres_schwarz_state_t *schwarz;
    double apply_time;
    int apply_count;
} ngspice_gmres_preconditioner_t;

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    unsigned int datalen;
    unsigned char data[64];
} ngspice_sha256_ctx_t;

static const uint32_t ngspice_sha256_k[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static ngspice_gmres_sidecar_cache_t ngspice_gmres_sidecar_cache = {0};

static uint32_t
ngspice_sha256_rotr(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32U - bits));
}

static void
ngspice_sha256_transform(ngspice_sha256_ctx_t *ctx, const unsigned char data[64])
{
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t m[64];
    unsigned int i;

    for (i = 0; i < 16; i++) {
        m[i] = ((uint32_t) data[i * 4] << 24) |
               ((uint32_t) data[i * 4 + 1] << 16) |
               ((uint32_t) data[i * 4 + 2] << 8) |
               ((uint32_t) data[i * 4 + 3]);
    }
    for (i = 16; i < 64; i++) {
        uint32_t s0 = ngspice_sha256_rotr(m[i - 15], 7) ^ ngspice_sha256_rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
        uint32_t s1 = ngspice_sha256_rotr(m[i - 2], 17) ^ ngspice_sha256_rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 64; i++) {
        uint32_t S1 = ngspice_sha256_rotr(e, 6) ^ ngspice_sha256_rotr(e, 11) ^ ngspice_sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + ngspice_sha256_k[i] + m[i];
        uint32_t S0 = ngspice_sha256_rotr(a, 2) ^ ngspice_sha256_rotr(a, 13) ^ ngspice_sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void
ngspice_sha256_init(ngspice_sha256_ctx_t *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667U;
    ctx->state[1] = 0xbb67ae85U;
    ctx->state[2] = 0x3c6ef372U;
    ctx->state[3] = 0xa54ff53aU;
    ctx->state[4] = 0x510e527fU;
    ctx->state[5] = 0x9b05688cU;
    ctx->state[6] = 0x1f83d9abU;
    ctx->state[7] = 0x5be0cd19U;
}

static void
ngspice_sha256_update(ngspice_sha256_ctx_t *ctx, const unsigned char *data, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64U) {
            ngspice_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512U;
            ctx->datalen = 0;
        }
    }
}

static void
ngspice_sha256_final(ngspice_sha256_ctx_t *ctx, unsigned char hash[32])
{
    unsigned int i = ctx->datalen;

    if (ctx->datalen < 56U) {
        ctx->data[i++] = 0x80U;
        while (i < 56U)
            ctx->data[i++] = 0x00U;
    } else {
        ctx->data[i++] = 0x80U;
        while (i < 64U)
            ctx->data[i++] = 0x00U;
        ngspice_sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56U);
    }

    ctx->bitlen += (uint64_t) ctx->datalen * 8U;
    ctx->data[63] = (unsigned char) (ctx->bitlen);
    ctx->data[62] = (unsigned char) (ctx->bitlen >> 8);
    ctx->data[61] = (unsigned char) (ctx->bitlen >> 16);
    ctx->data[60] = (unsigned char) (ctx->bitlen >> 24);
    ctx->data[59] = (unsigned char) (ctx->bitlen >> 32);
    ctx->data[58] = (unsigned char) (ctx->bitlen >> 40);
    ctx->data[57] = (unsigned char) (ctx->bitlen >> 48);
    ctx->data[56] = (unsigned char) (ctx->bitlen >> 56);
    ngspice_sha256_transform(ctx, ctx->data);

    for (i = 0; i < 4; i++) {
        hash[i]      = (unsigned char) ((ctx->state[0] >> (24U - i * 8U)) & 0xffU);
        hash[i + 4]  = (unsigned char) ((ctx->state[1] >> (24U - i * 8U)) & 0xffU);
        hash[i + 8]  = (unsigned char) ((ctx->state[2] >> (24U - i * 8U)) & 0xffU);
        hash[i + 12] = (unsigned char) ((ctx->state[3] >> (24U - i * 8U)) & 0xffU);
        hash[i + 16] = (unsigned char) ((ctx->state[4] >> (24U - i * 8U)) & 0xffU);
        hash[i + 20] = (unsigned char) ((ctx->state[5] >> (24U - i * 8U)) & 0xffU);
        hash[i + 24] = (unsigned char) ((ctx->state[6] >> (24U - i * 8U)) & 0xffU);
        hash[i + 28] = (unsigned char) ((ctx->state[7] >> (24U - i * 8U)) & 0xffU);
    }
}

static void
ngspice_sha256_hex(const unsigned char hash[32], char out_hex[65])
{
    static const char digits[] = "0123456789abcdef";
    unsigned int i;
    for (i = 0; i < 32; i++) {
        out_hex[i * 2] = digits[(hash[i] >> 4) & 0x0f];
        out_hex[i * 2 + 1] = digits[hash[i] & 0x0f];
    }
    out_hex[64] = '\0';
}

static int
ngspice_gmres_string_equal_ci(const char *lhs, const char *rhs)
{
    if (!lhs || !rhs)
        return 0;
    while (*lhs && *rhs) {
        if (tolower((unsigned char) *lhs) != tolower((unsigned char) *rhs))
            return 0;
        lhs++;
        rhs++;
    }
    return (*lhs == '\0' && *rhs == '\0');
}

static void
ngspice_gmres_copy_string(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0U)
        return;
    if (!src)
        src = "";
    strncpy(dst, src, dst_size - 1U);
    dst[dst_size - 1U] = '\0';
}

static void
ngspice_gmres_append_lookup_log(
    CKTcircuit *ckt,
    int newton_iter,
    const char *sidecar_path,
    int open_ok,
    const char *reason
)
{
    const char *log_path = getenv("NGSPICE_GMRES_SIDECAR_LOOKUP_LOG");
    FILE *fp;
    char gmin_buf[64];
    char time_buf[64];

    if (!log_path || log_path[0] == '\0')
        return;

    snprintf(time_buf, sizeof(time_buf), "%.17e", ckt ? ckt->CKTtime : 0.0);
    snprintf(gmin_buf, sizeof(gmin_buf), "%.17e", ckt ? ckt->CKTdiagGmin : 0.0);
    fp = fopen(log_path, "a");
    if (!fp)
        return;
    fprintf(
        fp,
        "{\"time\":%.17e,\"gmin\":%.17e,\"newton_iter\":%d,\"gmin_key\":\"%s\",\"sidecar_path\":\"%s\",\"open_ok\":%s,\"reason\":\"%s\"}\n",
        ckt ? ckt->CKTtime : 0.0,
        ckt ? ckt->CKTdiagGmin : 0.0,
        newton_iter,
        gmin_buf,
        sidecar_path ? sidecar_path : "",
        open_ok ? "true" : "false",
        reason ? reason : ""
    );
    fclose(fp);
}

static void
ngspice_gmres_resolve_sidecar_path(
    const ngspice_gmres_config_t *config,
    CKTcircuit *ckt,
    int newton_iter,
    char resolved_path[NGSPICE_GMRES_PATH_MAX]
)
{
    const char *src = config->sidecar_path;
    const char *scope = config->sidecar_scope;
    const char *iter_token = "{iter}";
    const char *time_token = "{time}";
    const char *gmin_token = "{gmin}";
    size_t iter_len = strlen(iter_token);
    size_t time_len = strlen(time_token);
    size_t gmin_len = strlen(gmin_token);
    char *dst = resolved_path;
    size_t remaining = NGSPICE_GMRES_PATH_MAX;
    char iter_buf[64];
    char time_buf[64];
    char gmin_buf[64];

    if (!resolved_path)
        return;
    resolved_path[0] = '\0';
    if (!src || src[0] == '\0')
        return;

    if (!ngspice_gmres_string_equal_ci(scope, "per_step")) {
        ngspice_gmres_copy_string(resolved_path, NGSPICE_GMRES_PATH_MAX, src);
        return;
    }

    snprintf(iter_buf, sizeof(iter_buf), "%d", newton_iter);
    snprintf(time_buf, sizeof(time_buf), "%.17e", ckt ? ckt->CKTtime : 0.0);
    snprintf(gmin_buf, sizeof(gmin_buf), "%.17e", ckt ? ckt->CKTdiagGmin : 0.0);

    while (*src != '\0' && remaining > 1U) {
        const char *replacement = NULL;
        size_t replacement_len = 0U;
        if (strncmp(src, iter_token, iter_len) == 0) {
            replacement = iter_buf;
            replacement_len = strlen(iter_buf);
            src += iter_len;
        } else if (strncmp(src, time_token, time_len) == 0) {
            replacement = time_buf;
            replacement_len = strlen(time_buf);
            src += time_len;
        } else if (strncmp(src, gmin_token, gmin_len) == 0) {
            replacement = gmin_buf;
            replacement_len = strlen(gmin_buf);
            src += gmin_len;
        } else {
            *dst++ = *src++;
            remaining--;
            continue;
        }

        while (replacement_len > 0U && remaining > 1U) {
            *dst++ = *replacement++;
            replacement_len--;
            remaining--;
        }
    }
    *dst = '\0';
}

static const char *
ngspice_gmres_precond_name(ngspice_gmres_precond_mode_t mode)
{
    switch (mode) {
    case NGSPICE_GMRES_PRECOND_JACOBI:
        return "jacobi";
    case NGSPICE_GMRES_PRECOND_ROW_SUM:
        return "row_sum";
    case NGSPICE_GMRES_PRECOND_LEARNED_DIAGONAL:
        return "learned_diagonal";
    case NGSPICE_GMRES_PRECOND_LEARNED_SCHWARZ:
        return "learned_schwarz_v1_sparse";
    case NGSPICE_GMRES_PRECOND_IDENTITY:
    default:
        return "identity";
    }
}

static int
ngspice_gmres_json_has_key(const char *json_text, const char *key)
{
    char pattern[128];
    if (!json_text || !key)
        return 0;
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    return strstr(json_text, pattern) != NULL;
}

static int
ngspice_gmres_supported_sidecar_scale_mode(const char *scale_mode)
{
    return ngspice_gmres_string_equal_ci(scale_mode, "log10_scale") ||
           ngspice_gmres_string_equal_ci(scale_mode, "log_scale_dense") ||
           ngspice_gmres_string_equal_ci(scale_mode, "log10_dense_scale");
}

static ngspice_gmres_precond_mode_t
ngspice_gmres_precond_from_string(const char *value, ngspice_gmres_precond_mode_t fallback)
{
    if (!value || value[0] == '\0')
        return fallback;
    if (ngspice_gmres_string_equal_ci(value, "identity"))
        return NGSPICE_GMRES_PRECOND_IDENTITY;
    if (ngspice_gmres_string_equal_ci(value, "jacobi"))
        return NGSPICE_GMRES_PRECOND_JACOBI;
    if (ngspice_gmres_string_equal_ci(value, "row_sum"))
        return NGSPICE_GMRES_PRECOND_ROW_SUM;
    if (ngspice_gmres_string_equal_ci(value, "learned_diagonal"))
        return NGSPICE_GMRES_PRECOND_LEARNED_DIAGONAL;
    if (ngspice_gmres_string_equal_ci(value, "learned_schwarz") ||
        ngspice_gmres_string_equal_ci(value, "learned_schwarz_v1_sparse"))
        return NGSPICE_GMRES_PRECOND_LEARNED_SCHWARZ;
    return fallback;
}

static int
ngspice_gmres_parse_int_env(const char *env_name, int default_value)
{
    const char *value = getenv(env_name);
    long parsed;
    char *endptr;
    if (!value || value[0] == '\0')
        return default_value;
    errno = 0;
    parsed = strtol(value, &endptr, 10);
    if (errno != 0 || endptr == value)
        return default_value;
    return (int) parsed;
}

static double
ngspice_gmres_parse_double_env(const char *env_name, double default_value)
{
    const char *value = getenv(env_name);
    double parsed;
    char *endptr;
    if (!value || value[0] == '\0')
        return default_value;
    errno = 0;
    parsed = strtod(value, &endptr);
    if (errno != 0 || endptr == value)
        return default_value;
    return parsed;
}

static void
ngspice_gmres_parse_config(ngspice_gmres_config_t *config)
{
    const char *enabled = getenv("NGSPICE_GMRES_MODE");
    const char *precond = getenv("NGSPICE_GMRES_PRECOND");
    const char *fallback = getenv("NGSPICE_GMRES_PRECOND_FALLBACK");
    const char *sidecar_scope = getenv("NGSPICE_GMRES_PRECOND_SIDECAR_SCOPE");
    const char *sidecar_path = getenv("NGSPICE_GMRES_PRECOND_SIDECAR_PATH");

    memset(config, 0, sizeof(*config));
    config->enabled = (enabled && strcmp(enabled, "1") == 0);
    config->restart = ngspice_gmres_parse_int_env("NGSPICE_GMRES_RESTART", 30);
    config->max_iters = ngspice_gmres_parse_int_env("NGSPICE_GMRES_MAX_ITERS", 120);
    config->rtol = ngspice_gmres_parse_double_env("NGSPICE_GMRES_RTOL", 1e-8);
    config->atol = ngspice_gmres_parse_double_env("NGSPICE_GMRES_ATOL", 1e-10);
    config->use_rhsold_x0 = ngspice_gmres_parse_int_env("NGSPICE_GMRES_USE_RHSOLD_X0", 1) != 0;
    config->requested_precond = ngspice_gmres_precond_from_string(precond, NGSPICE_GMRES_PRECOND_IDENTITY);
    config->fallback_precond = ngspice_gmres_precond_from_string(fallback, NGSPICE_GMRES_PRECOND_ROW_SUM);
    if (sidecar_scope && sidecar_scope[0] != '\0') {
        ngspice_gmres_copy_string(config->sidecar_scope, sizeof(config->sidecar_scope), sidecar_scope);
    } else {
        ngspice_gmres_copy_string(config->sidecar_scope, sizeof(config->sidecar_scope), "transient_static");
    }
    if (sidecar_path && sidecar_path[0] != '\0') {
        ngspice_gmres_copy_string(config->sidecar_path, sizeof(config->sidecar_path), sidecar_path);
    }
    if (config->restart < 1)
        config->restart = 30;
    if (config->max_iters < config->restart)
        config->max_iters = config->restart;
    if (config->rtol <= 0.0)
        config->rtol = 1e-8;
    if (config->atol < 0.0)
        config->atol = 1e-10;
}

static double
ngspice_gmres_dot(const double *lhs, const double *rhs, int count)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < count; i++)
        sum += lhs[i] * rhs[i];
    return sum;
}

static double
ngspice_gmres_norm2(const double *values, int count)
{
    return sqrt(ngspice_gmres_dot(values, values, count));
}

static void
ngspice_gmres_copy(double *dst, const double *src, int count)
{
    if (count > 0)
        memcpy(dst, src, (size_t) count * sizeof(double));
}

static void
ngspice_gmres_scale(double *values, double scale, int count)
{
    int i;
    for (i = 0; i < count; i++)
        values[i] *= scale;
}

static void
ngspice_gmres_axpy(double *dst, double alpha, const double *src, int count)
{
    int i;
    for (i = 0; i < count; i++)
        dst[i] += alpha * src[i];
}

static int
ngspice_gmres_row_is_branch(CKTcircuit *ckt, int row_index)
{
    CKTnode *node;
    const char *name;
    size_t len;
    for (node = ckt->CKTnodes; node; node = node->next) {
        if (node->number != row_index)
            continue;
        name = node->name ? (const char *) node->name : "";
        len = strlen(name);
        if (len >= 7U && ngspice_gmres_string_equal_ci(name + len - 7U, "#branch"))
            return 1;
        return 0;
    }
    return 0;
}

static int
ngspice_gmres_compute_sparse_scales(
    CKTcircuit *ckt,
    ngspice_gmres_precond_mode_t mode,
    double *scales
)
{
    MatrixFrame *matrix = ckt->CKTmatrix ? ckt->CKTmatrix->SPmatrix : NULL;
    int external_row;
    if (!matrix)
        return 0;
    if (!matrix->RowsLinked)
        spcLinkRows(matrix);

    for (external_row = 1; external_row <= matrix->Size; external_row++) {
        int internal_row = matrix->ExtToIntRowMap
            ? matrix->ExtToIntRowMap[external_row]
            : external_row;
        ElementPtr diag = (ElementPtr) SMPfindElt(
            ckt->CKTmatrix,
            external_row,
            external_row,
            0
        );
        double diag_value = diag ? diag->Real : 0.0;
        double scale = 1.0;

        if (internal_row < 1 || internal_row > matrix->Size)
            return 0;
        if (mode == NGSPICE_GMRES_PRECOND_IDENTITY) {
            scale = 1.0;
        } else if (mode == NGSPICE_GMRES_PRECOND_JACOBI) {
            double diagonal_abs = fabs(diag_value + ckt->CKTdiagGmin);
            scale = 1.0 / (
                diagonal_abs > 1e-30 ? diagonal_abs : 1e-30
            );
        } else if (mode == NGSPICE_GMRES_PRECOND_ROW_SUM) {
            ElementPtr element = matrix->FirstInRow
                ? matrix->FirstInRow[internal_row]
                : NULL;
            double row_sum = 0.0;
            for (; element; element = element->NextInRow) {
                if (diag && element == diag)
                    row_sum += fabs(element->Real + ckt->CKTdiagGmin);
                else
                    row_sum += fabs(element->Real);
            }
            if (!diag)
                row_sum += fabs(ckt->CKTdiagGmin);
            scale = 1.0 / (row_sum > 1e-30 ? row_sum : 1e-30);
        } else {
            return 0;
        }
        if (!isfinite(scale) || scale <= 0.0)
            scale = 1.0;
        scales[external_row - 1] = scale;
    }
    return 1;
}

static char *
ngspice_gmres_read_text_file(const char *path)
{
    FILE *fp;
    long size;
    char *buffer;
    if (!path || path[0] == '\0')
        return NULL;
    fp = fopen(path, "r");
    if (!fp)
        return NULL;
    if (fseek(fp, 0L, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    size = ftell(fp);
    if (size < 0L) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0L, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    buffer = (char *) malloc((size_t) size + 1U);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }
    if (size > 0L && fread(buffer, 1, (size_t) size, fp) != (size_t) size) {
        free(buffer);
        fclose(fp);
        return NULL;
    }
    buffer[size] = '\0';
    fclose(fp);
    return buffer;
}

static const char *
ngspice_gmres_find_json_key(const char *json_text, const char *key)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    return strstr(json_text, pattern);
}

static int
ngspice_gmres_parse_json_number(const char *json_text, const char *key, double *out_value)
{
    const char *cursor = ngspice_gmres_find_json_key(json_text, key);
    char *endptr;
    if (!cursor)
        return 0;
    cursor = strchr(cursor, ':');
    if (!cursor)
        return 0;
    cursor++;
    while (*cursor && isspace((unsigned char) *cursor))
        cursor++;
    errno = 0;
    *out_value = strtod(cursor, &endptr);
    if (errno != 0 || endptr == cursor)
        return 0;
    return 1;
}

static int
ngspice_gmres_parse_json_string(const char *json_text, const char *key, char *out_value, size_t out_size)
{
    const char *cursor = ngspice_gmres_find_json_key(json_text, key);
    size_t len = 0;
    if (!cursor)
        return 0;
    cursor = strchr(cursor, ':');
    if (!cursor)
        return 0;
    cursor++;
    while (*cursor && isspace((unsigned char) *cursor))
        cursor++;
    if (*cursor != '"')
        return 0;
    cursor++;
    while (cursor[len] && cursor[len] != '"')
        len++;
    if (cursor[len] != '"' || len + 1U > out_size)
        return 0;
    memcpy(out_value, cursor, len);
    out_value[len] = '\0';
    return 1;
}

static int
ngspice_gmres_parse_json_double_array(const char *json_text, const char *key, double *out_values, int expected_count)
{
    const char *cursor = ngspice_gmres_find_json_key(json_text, key);
    char *endptr;
    int idx = 0;
    if (!cursor)
        return 0;
    cursor = strchr(cursor, '[');
    if (!cursor)
        return 0;
    cursor++;
    while (*cursor && *cursor != ']') {
        while (*cursor && (isspace((unsigned char) *cursor) || *cursor == ','))
            cursor++;
        if (*cursor == ']')
            break;
        if (idx >= expected_count)
            return 0;
        errno = 0;
        out_values[idx] = strtod(cursor, &endptr);
        if (errno != 0 || endptr == cursor)
            return 0;
        cursor = endptr;
        idx++;
    }
    return idx == expected_count;
}

static int
ngspice_gmres_validate_finite_array(const double *values, int count)
{
    int idx;
    for (idx = 0; idx < count; idx++) {
        if (!isfinite(values[idx]))
            return 0;
    }
    return 1;
}

static int
ngspice_gmres_parse_json_bool_array(const char *json_text, const char *key, int *out_values, int expected_count)
{
    const char *cursor = ngspice_gmres_find_json_key(json_text, key);
    int idx = 0;
    if (!cursor)
        return 0;
    cursor = strchr(cursor, '[');
    if (!cursor)
        return 0;
    cursor++;
    while (*cursor && *cursor != ']') {
        while (*cursor && (isspace((unsigned char) *cursor) || *cursor == ','))
            cursor++;
        if (*cursor == ']')
            break;
        if (idx >= expected_count)
            return 0;
        if (strncmp(cursor, "true", 4) == 0) {
            out_values[idx++] = 1;
            cursor += 4;
        } else if (strncmp(cursor, "false", 5) == 0) {
            out_values[idx++] = 0;
            cursor += 5;
        } else if (*cursor == '1' || *cursor == '0') {
            out_values[idx++] = (*cursor == '1') ? 1 : 0;
            cursor++;
        } else {
            return 0;
        }
    }
    return idx == expected_count;
}

typedef struct {
    int number;
    char *name;
} ngspice_gmres_node_hash_entry_t;

static int
ngspice_gmres_compare_node_hash_entries(const void *lhs, const void *rhs)
{
    const ngspice_gmres_node_hash_entry_t *left =
        (const ngspice_gmres_node_hash_entry_t *) lhs;
    const ngspice_gmres_node_hash_entry_t *right =
        (const ngspice_gmres_node_hash_entry_t *) rhs;
    if (left->number < right->number)
        return -1;
    if (left->number > right->number)
        return 1;
    return strcmp(left->name, right->name);
}

static int
ngspice_gmres_compute_node_map_hash(CKTcircuit *ckt, int matrix_size, char out_hash[65])
{
    static const unsigned char newline = '\n';
    ngspice_gmres_node_hash_entry_t *entries = NULL;
    ngspice_sha256_ctx_t sha_ctx;
    unsigned char hash[32];
    CKTnode *node;
    size_t count = 0U;
    size_t index = 0U;
    int success = 1;

    if (!ckt || matrix_size <= 0 || !out_hash)
        return 0;
    for (node = ckt->CKTnodes; node; node = node->next) {
        if (node->number >= 0 && node->number <= matrix_size)
            count++;
    }
    if (count > 0U) {
        entries = (ngspice_gmres_node_hash_entry_t *) calloc(
            count,
            sizeof(*entries));
        if (!entries)
            return 0;
    }

    for (node = ckt->CKTnodes; node; node = node->next) {
        const char *src_name;
        size_t len;
        size_t char_index;
        if (node->number < 0 || node->number > matrix_size)
            continue;
        src_name = node->name ? (const char *) node->name : "";
        len = strlen(src_name);
        entries[index].number = node->number;
        entries[index].name = (char *) malloc(len + 1U);
        if (!entries[index].name) {
            success = 0;
            break;
        }
        for (char_index = 0U; char_index < len; char_index++) {
            entries[index].name[char_index] = (char) tolower(
                (unsigned char) src_name[char_index]);
        }
        entries[index].name[len] = '\0';
        index++;
    }

    if (success) {
        qsort(entries, count, sizeof(*entries),
            ngspice_gmres_compare_node_hash_entries);
        ngspice_sha256_init(&sha_ctx);
        for (index = 0U; index < count; index++) {
            char prefix[64];
            int written = snprintf(
                prefix,
                sizeof(prefix),
                "%d:",
                entries[index].number);
            if (written < 0 || (size_t) written >= sizeof(prefix)) {
                success = 0;
                break;
            }
            ngspice_sha256_update(
                &sha_ctx,
                (const unsigned char *) prefix,
                (size_t) written);
            ngspice_sha256_update(
                &sha_ctx,
                (const unsigned char *) entries[index].name,
                strlen(entries[index].name));
            ngspice_sha256_update(
                &sha_ctx,
                &newline,
                sizeof(newline));
        }
        if (success) {
            ngspice_sha256_final(&sha_ctx, hash);
            ngspice_sha256_hex(hash, out_hash);
        }
    }

    for (index = 0U; index < count; index++)
        free(entries[index].name);
    free(entries);
    return success;
}

static int
ngspice_gmres_load_learned_sidecar(
    CKTcircuit *ckt,
    const ngspice_gmres_config_t *config,
    int newton_iter,
    double *scales,
    char resolved_sidecar_path[NGSPICE_GMRES_PATH_MAX],
    char fallback_reason[NGSPICE_GMRES_REASON_MAX]
)
{
    char *json_text = NULL;
    char local_resolved_sidecar_path[NGSPICE_GMRES_PATH_MAX];
    double schema_version_value = 0.0;
    double matrix_size_value = 0.0;
    double scale_clip = 12.0;
    double default_scale = 1.0;
    double *log_scales = NULL;
    int *valid_mask = NULL;
    char scale_mode[64];
    char expected_hash[65];
    char actual_hash[65];
    int size = SMPmatSize(ckt->CKTmatrix);
    int row;
    int ok = 0;
    struct stat st;
    int stat_ok = 0;

    fallback_reason[0] = '\0';
    ngspice_gmres_resolve_sidecar_path(config, ckt, newton_iter, local_resolved_sidecar_path);
    if (resolved_sidecar_path)
        ngspice_gmres_copy_string(
            resolved_sidecar_path,
            NGSPICE_GMRES_PATH_MAX,
            local_resolved_sidecar_path
        );
    if (local_resolved_sidecar_path[0] == '\0') {
        strncpy(fallback_reason, "sidecar_missing", NGSPICE_GMRES_REASON_MAX - 1);
        ngspice_gmres_append_lookup_log(
            ckt,
            newton_iter,
            local_resolved_sidecar_path,
            0,
            fallback_reason
        );
        return 0;
    }

    if (!ngspice_gmres_compute_node_map_hash(ckt, size, actual_hash)) {
        strncpy(fallback_reason, "sidecar_invalid_node_map_hash", NGSPICE_GMRES_REASON_MAX - 1);
        return 0;
    }
    stat_ok = (stat(local_resolved_sidecar_path, &st) == 0);

    if (stat_ok &&
        ngspice_gmres_sidecar_cache.valid &&
        ngspice_gmres_sidecar_cache.scales != NULL &&
        ngspice_gmres_sidecar_cache.matrix_size == size &&
        ngspice_gmres_sidecar_cache.file_size == st.st_size &&
        ngspice_gmres_sidecar_cache.file_mtime == st.st_mtime &&
        strcmp(ngspice_gmres_sidecar_cache.sidecar_path, local_resolved_sidecar_path) == 0 &&
        strcmp(ngspice_gmres_sidecar_cache.node_map_hash, actual_hash) == 0) {
        for (row = 0; row < size; row++)
            scales[row] = ngspice_gmres_sidecar_cache.scales[row];
        return 1;
    }

    json_text = ngspice_gmres_read_text_file(local_resolved_sidecar_path);
    if (!json_text) {
        strncpy(fallback_reason, "sidecar_invalid_open_failed", NGSPICE_GMRES_REASON_MAX - 1);
        ngspice_gmres_append_lookup_log(
            ckt,
            newton_iter,
            local_resolved_sidecar_path,
            0,
            fallback_reason
        );
        return 0;
    }
    ngspice_gmres_append_lookup_log(
        ckt,
        newton_iter,
        local_resolved_sidecar_path,
        1,
        "open_ok"
    );

    if (!ngspice_gmres_parse_json_number(json_text, "schema_version", &schema_version_value) ||
        !isfinite(schema_version_value) ||
        ((int) schema_version_value != 1)) {
        strncpy(fallback_reason, "sidecar_invalid_schema_version", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }
    if (!ngspice_gmres_parse_json_string(json_text, "scale_mode", scale_mode, sizeof(scale_mode)) ||
        !ngspice_gmres_supported_sidecar_scale_mode(scale_mode)) {
        strncpy(fallback_reason, "sidecar_invalid_scale_mode", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }
    if (!ngspice_gmres_parse_json_number(json_text, "matrix_size", &matrix_size_value) ||
        !isfinite(matrix_size_value) ||
        ((int) matrix_size_value != size)) {
        strncpy(fallback_reason, "sidecar_invalid_matrix_size", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }
    if (ngspice_gmres_parse_json_string(json_text, "node_map_hash", expected_hash, sizeof(expected_hash))) {
        if (strcmp(expected_hash, actual_hash) != 0) {
            strncpy(fallback_reason, "sidecar_invalid_node_map_hash", NGSPICE_GMRES_REASON_MAX - 1);
            goto cleanup;
        }
    }
    (void) ngspice_gmres_parse_json_number(json_text, "scale_clip", &scale_clip);
    (void) ngspice_gmres_parse_json_number(json_text, "default_scale", &default_scale);
    if (!isfinite(scale_clip) || scale_clip <= 0.0) {
        strncpy(fallback_reason, "sidecar_invalid_scale_clip", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }
    if (!isfinite(default_scale) || default_scale <= 0.0) {
        strncpy(fallback_reason, "sidecar_invalid_default_scale", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }

    log_scales = (double *) malloc((size_t) size * sizeof(double));
    valid_mask = (int *) malloc((size_t) size * sizeof(int));
    if (!log_scales || !valid_mask) {
        strncpy(fallback_reason, "sidecar_invalid_nomem", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }
    for (row = 0; row < size; row++)
        valid_mask[row] = 1;

    if (!ngspice_gmres_parse_json_double_array(json_text, "log_scale_dense", log_scales, size)) {
        strncpy(fallback_reason, "sidecar_invalid_scale_length", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }
    if (!ngspice_gmres_validate_finite_array(log_scales, size)) {
        strncpy(fallback_reason, "sidecar_invalid_nan_inf", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }
    if (ngspice_gmres_json_has_key(json_text, "valid_mask_dense") &&
        !ngspice_gmres_parse_json_bool_array(json_text, "valid_mask_dense", valid_mask, size)) {
        strncpy(fallback_reason, "sidecar_invalid_valid_mask_length", NGSPICE_GMRES_REASON_MAX - 1);
        goto cleanup;
    }

    for (row = 1; row <= size; row++) {
        double clipped = log_scales[row - 1];
        if (clipped > scale_clip)
            clipped = scale_clip;
        else if (clipped < -scale_clip)
            clipped = -scale_clip;
        scales[row - 1] = pow(10.0, clipped);
        if (ngspice_gmres_row_is_branch(ckt, row))
            scales[row - 1] = 1.0;
        else if (!valid_mask[row - 1] || !isfinite(scales[row - 1]) || scales[row - 1] <= 0.0)
            scales[row - 1] = default_scale;
    }
    ok = 1;

cleanup:
    if (ok && stat_ok) {
        double *cached_scales = (double *) malloc((size_t) size * sizeof(double));
        if (cached_scales) {
            for (row = 0; row < size; row++)
                cached_scales[row] = scales[row];
            free(ngspice_gmres_sidecar_cache.scales);
            ngspice_gmres_sidecar_cache.valid = 1;
            ngspice_gmres_sidecar_cache.matrix_size = size;
            ngspice_gmres_sidecar_cache.file_size = st.st_size;
            ngspice_gmres_sidecar_cache.file_mtime = st.st_mtime;
            ngspice_gmres_sidecar_cache.scales = cached_scales;
            strncpy(
                ngspice_gmres_sidecar_cache.sidecar_path,
                local_resolved_sidecar_path,
                NGSPICE_GMRES_PATH_MAX - 1
            );
            ngspice_gmres_sidecar_cache.sidecar_path[NGSPICE_GMRES_PATH_MAX - 1] = '\0';
            strncpy(
                ngspice_gmres_sidecar_cache.node_map_hash,
                actual_hash,
                sizeof(ngspice_gmres_sidecar_cache.node_map_hash) - 1
            );
            ngspice_gmres_sidecar_cache.node_map_hash[
                sizeof(ngspice_gmres_sidecar_cache.node_map_hash) - 1
            ] = '\0';
        }
    }
    free(log_scales);
    free(valid_mask);
    free(json_text);
    return ok;
}

static int
ngspice_gmres_prepare_preconditioner(
    CKTcircuit *ckt,
    const ngspice_gmres_config_t *config,
    int newton_iter,
    const double *linear_rhs,
    const double *initial_guess,
    const double *initial_residual,
    const char *initial_guess_mode,
    ngspice_gmres_preconditioner_t *preconditioner,
    char resolved_sidecar_path[NGSPICE_GMRES_PATH_MAX],
    ngspice_gmres_precond_mode_t *executed_mode,
    char fallback_reason[NGSPICE_GMRES_REASON_MAX],
    double *sidecar_load_time
)
{
    double load_start;

    fallback_reason[0] = '\0';
    if (sidecar_load_time)
        *sidecar_load_time = 0.0;
    preconditioner->mode = config->requested_precond;
    *executed_mode = config->requested_precond;

    if (config->requested_precond == NGSPICE_GMRES_PRECOND_LEARNED_DIAGONAL) {
        load_start = SPfrontEnd->IFseconds();
        if (ngspice_gmres_load_learned_sidecar(
                ckt,
                config,
                newton_iter,
                preconditioner->scales,
                resolved_sidecar_path,
                fallback_reason))
        {
            if (sidecar_load_time)
                *sidecar_load_time = SPfrontEnd->IFseconds() - load_start;
            return 1;
        }
        if (sidecar_load_time)
            *sidecar_load_time = SPfrontEnd->IFseconds() - load_start;
        *executed_mode = config->fallback_precond;
        preconditioner->mode = *executed_mode;
    } else if (
        config->requested_precond ==
        NGSPICE_GMRES_PRECOND_LEARNED_SCHWARZ) {
        char actual_hash[65];
        char schwarz_reason[NGSPICE_GMRES_REASON_MAX];

        load_start = SPfrontEnd->IFseconds();
        schwarz_reason[0] = '\0';
        resolved_sidecar_path[0] = '\0';
        if (!ngspice_gmres_string_equal_ci(
                config->sidecar_scope,
                "per_step")) {
            ngspice_gmres_copy_string(
                schwarz_reason,
                sizeof(schwarz_reason),
                "schwarz_requires_per_step_sidecar");
        } else if (
            !strstr(config->sidecar_path, "{iter}") ||
            !strstr(config->sidecar_path, "{time}") ||
            !strstr(config->sidecar_path, "{gmin}")) {
            ngspice_gmres_copy_string(
                schwarz_reason,
                sizeof(schwarz_reason),
                "schwarz_sidecar_template_incomplete");
        } else {
            ngspice_gmres_resolve_sidecar_path(
                config,
                ckt,
                newton_iter,
                resolved_sidecar_path);
        }
        if (schwarz_reason[0] != '\0') {
            /* Fail closed and use the configured non-learning fallback. */
        } else if (!resolved_sidecar_path[0]) {
            ngspice_gmres_copy_string(
                schwarz_reason,
                sizeof(schwarz_reason),
                "schwarz_sidecar_missing");
        } else if (!ngspice_gmres_compute_node_map_hash(
                       ckt,
                       preconditioner->count,
                       actual_hash)) {
            ngspice_gmres_copy_string(
                schwarz_reason,
                sizeof(schwarz_reason),
                "schwarz_node_map_hash_failed");
        } else if (ngspice_gmres_schwarz_create(
                       ckt,
                       resolved_sidecar_path,
                       preconditioner->count,
                       newton_iter,
                       actual_hash,
                       linear_rhs,
                       initial_guess,
                       initial_residual,
                       initial_guess_mode,
                       &preconditioner->schwarz,
                       schwarz_reason,
                       sizeof(schwarz_reason))) {
            ngspice_gmres_schwarz_metrics_t metrics;
            ngspice_gmres_schwarz_get_metrics(
                preconditioner->schwarz,
                &metrics);
            if (sidecar_load_time)
                *sidecar_load_time = metrics.sidecar_load_time;
            ngspice_gmres_append_lookup_log(
                ckt,
                newton_iter,
                resolved_sidecar_path,
                1,
                "ok");
            return 1;
        }

        if (sidecar_load_time)
            *sidecar_load_time = SPfrontEnd->IFseconds() - load_start;
        ngspice_gmres_copy_string(
            fallback_reason,
            NGSPICE_GMRES_REASON_MAX,
            schwarz_reason[0] ? schwarz_reason : "schwarz_setup_failed");
        ngspice_gmres_append_lookup_log(
            ckt,
            newton_iter,
            resolved_sidecar_path,
            0,
            fallback_reason);
        *executed_mode = config->fallback_precond;
        preconditioner->mode = *executed_mode;
    }

    if (*executed_mode == NGSPICE_GMRES_PRECOND_LEARNED_DIAGONAL ||
        *executed_mode == NGSPICE_GMRES_PRECOND_LEARNED_SCHWARZ) {
        *executed_mode = NGSPICE_GMRES_PRECOND_ROW_SUM;
        preconditioner->mode = *executed_mode;
        if (!fallback_reason[0]) {
            ngspice_gmres_copy_string(
                fallback_reason,
                NGSPICE_GMRES_REASON_MAX,
                "invalid_learned_fallback");
        }
    }
    return ngspice_gmres_compute_sparse_scales(
        ckt,
        *executed_mode,
        preconditioner->scales);
}

static int
ngspice_gmres_apply_preconditioner(
    ngspice_gmres_preconditioner_t *preconditioner,
    const double *rhs,
    double *out,
    int count
)
{
    double started = SPfrontEnd->IFseconds();
    int ok = 1;
    int i;

    if (!preconditioner || !rhs || !out ||
        count != preconditioner->count)
        return 0;

    if (preconditioner->mode ==
        NGSPICE_GMRES_PRECOND_LEARNED_SCHWARZ) {
        if (!preconditioner->schwarz)
            ok = 0;
        else
            ok = ngspice_gmres_schwarz_apply(
                preconditioner->schwarz,
                rhs,
                out,
                count);
    } else {
        for (i = 0; i < count; i++)
            out[i] = preconditioner->scales[i] * rhs[i];
    }
    preconditioner->apply_count++;
    preconditioner->apply_time += SPfrontEnd->IFseconds() - started;
    return ok;
}

static void
ngspice_gmres_collect_preconditioner_metrics(
    ngspice_gmres_preconditioner_t *preconditioner,
    ngspice_gmres_result_t *result
)
{
    result->preconditioner_apply_count = preconditioner->apply_count;
    result->preconditioner_apply_time = preconditioner->apply_time;
    result->preconditioner_retained_bytes =
        (size_t) preconditioner->count * sizeof(double);
    if (preconditioner->schwarz) {
        ngspice_gmres_schwarz_metrics_t metrics;
        ngspice_gmres_schwarz_get_metrics(
            preconditioner->schwarz,
            &metrics);
        result->sidecar_load_time = metrics.sidecar_load_time;
        if (result->preconditioner_setup_time <= 0.0)
            result->preconditioner_setup_time = metrics.setup_time;
        result->preconditioner_factor_time = metrics.factor_time;
        result->preconditioner_apply_count = metrics.apply_count;
        result->preconditioner_failed_apply_count =
            metrics.failed_apply_count;
        result->preconditioner_apply_time = metrics.apply_time_total;
        result->preconditioner_block_count = metrics.block_count;
        result->preconditioner_total_block_rows =
            metrics.total_block_rows;
        result->preconditioner_covered_rows = metrics.covered_rows;
        result->preconditioner_uncovered_rows = metrics.uncovered_rows;
        result->preconditioner_max_block_size =
            metrics.max_block_size;
        result->preconditioner_sidecar_file_bytes =
            metrics.sidecar_file_bytes;
        result->preconditioner_layout_bytes = metrics.layout_bytes;
        result->preconditioner_parameter_bytes =
            metrics.parameter_bytes;
        result->preconditioner_retained_bytes += metrics.retained_bytes;
        result->preconditioner_peak_estimated_bytes =
            metrics.peak_estimated_bytes +
            (size_t) preconditioner->count * sizeof(double);
        result->preconditioner_factor_bytes = metrics.factor_bytes;
        result->preconditioner_fallback_bytes =
            metrics.fallback_bytes;
        result->preconditioner_workspace_bytes =
            metrics.workspace_bytes;
    }
}

static void
ngspice_gmres_destroy_preconditioner(
    ngspice_gmres_preconditioner_t *preconditioner
)
{
    if (!preconditioner)
        return;
    if (preconditioner->schwarz) {
        ngspice_gmres_schwarz_destroy(preconditioner->schwarz);
        preconditioner->schwarz = NULL;
    }
    preconditioner->scales = NULL;
    preconditioner->count = 0;
    preconditioner->mode = NGSPICE_GMRES_PRECOND_IDENTITY;
}

static void
ngspice_gmres_apply_matrix(ngspice_gmres_matvec_ctx_t *ctx, const double *x, double *y)
{
    int i;
    ctx->in_vec[0] = 0.0;
    ctx->out_vec[0] = 0.0;
    for (i = 0; i < ctx->size; i++)
        ctx->in_vec[i + 1] = x[i];
    SMPmultiply(ctx->ckt->CKTmatrix, ctx->out_vec, ctx->in_vec, NULL, NULL);
    for (i = 0; i < ctx->size; i++)
        y[i] = ctx->out_vec[i + 1] + (ctx->diag_gmin * x[i]);
}

static int
ngspice_gmres_solve(
    CKTcircuit *ckt,
    const ngspice_gmres_config_t *config,
    int newton_iter,
    ngspice_gmres_result_t *result
)
{
    int n = SMPmatSize(ckt->CKTmatrix);
    int restart = config->restart;
    int max_iters = config->max_iters;
    int max_cycles = (max_iters + restart - 1) / restart;
    int cycle;
    int total_iters = 0;
    int converged = 0;
    double *x = NULL, *b = NULL, *r = NULL, *z = NULL, *w = NULL, *scales = NULL;
    double *V = NULL, *H = NULL, *cs = NULL, *sn = NULL, *s = NULL, *y = NULL;
    double raw_tol, rhs_scale, initial_precond_beta, current_beta;
    ngspice_gmres_precond_mode_t executed_mode = NGSPICE_GMRES_PRECOND_IDENTITY;
    ngspice_gmres_matvec_ctx_t matvec_ctx;
    ngspice_gmres_preconditioner_t preconditioner;
    double preconditioner_setup_started;
    char precond_fallback_reason[NGSPICE_GMRES_REASON_MAX];

    memset(result, 0, sizeof(*result));
    memset(&preconditioner, 0, sizeof(preconditioner));
    strncpy(result->executed_precond, "identity", sizeof(result->executed_precond) - 1);
    precond_fallback_reason[0] = '\0';
    if (!ckt->CKTmatrix || !ckt->CKTmatrix->SPmatrix || n <= 0) {
        strncpy(result->fallback_reason, "gmres_bad_matrix", sizeof(result->fallback_reason) - 1);
        return 0;
    }

    spSetReal(ckt->CKTmatrix->SPmatrix);

    x = (double *) calloc((size_t) n, sizeof(double));
    b = (double *) calloc((size_t) n, sizeof(double));
    r = (double *) calloc((size_t) n, sizeof(double));
    z = (double *) calloc((size_t) n, sizeof(double));
    w = (double *) calloc((size_t) n, sizeof(double));
    scales = (double *) calloc((size_t) n, sizeof(double));
    V = (double *) calloc((size_t) (restart + 1) * (size_t) n, sizeof(double));
    H = (double *) calloc((size_t) (restart + 1) * (size_t) restart, sizeof(double));
    cs = (double *) calloc((size_t) restart, sizeof(double));
    sn = (double *) calloc((size_t) restart, sizeof(double));
    s = (double *) calloc((size_t) (restart + 1), sizeof(double));
    y = (double *) calloc((size_t) restart, sizeof(double));
    matvec_ctx.in_vec = (double *) calloc((size_t) n + 1U, sizeof(double));
    matvec_ctx.out_vec = (double *) calloc((size_t) n + 1U, sizeof(double));
    if (!x || !b || !r || !z || !w || !scales || !V || !H || !cs || !sn || !s || !y || !matvec_ctx.in_vec || !matvec_ctx.out_vec) {
        strncpy(result->fallback_reason, "gmres_nomem", sizeof(result->fallback_reason) - 1);
        goto cleanup;
    }
    preconditioner.mode = NGSPICE_GMRES_PRECOND_IDENTITY;
    preconditioner.scales = scales;
    preconditioner.count = n;
    result->gmres_workspace_bytes = sizeof(double) * (
        ((size_t) restart + 9U) * (size_t) n +
        ((size_t) restart + 1U) * (size_t) restart +
        4U * (size_t) restart +
        3U
    );

    {
        const char *initial_guess_mode =
            (config->use_rhsold_x0 && ckt->CKTrhsOld) ? "rhsold" : "zero";
        for (cycle = 0; cycle < n; cycle++) {
            b[cycle] = ckt->CKTrhs[cycle + 1];
            x[cycle] = strcmp(initial_guess_mode, "rhsold") == 0
                ? ckt->CKTrhsOld[cycle + 1]
                : 0.0;
        }
        result->rhs_norm = ngspice_gmres_norm2(b, n);
        rhs_scale = result->rhs_norm > DBL_EPSILON
            ? result->rhs_norm : DBL_EPSILON;
        raw_tol = config->atol + config->rtol * rhs_scale;

        matvec_ctx.ckt = ckt;
        matvec_ctx.size = n;
        matvec_ctx.diag_gmin = ckt->CKTdiagGmin;
        ngspice_gmres_apply_matrix(&matvec_ctx, x, r);
        for (cycle = 0; cycle < n; cycle++)
            r[cycle] = b[cycle] - r[cycle];
        result->initial_raw_residual = ngspice_gmres_norm2(r, n);
        result->initial_true_relative_residual =
            result->initial_raw_residual / rhs_scale;
        result->final_raw_residual = result->initial_raw_residual;
        result->final_true_relative_residual =
            result->initial_true_relative_residual;

        preconditioner_setup_started = SPfrontEnd->IFseconds();
        if (!ngspice_gmres_prepare_preconditioner(
                ckt,
                config,
                newton_iter,
                b,
                x,
                r,
                initial_guess_mode,
                &preconditioner,
                result->resolved_sidecar_path,
                &executed_mode,
                precond_fallback_reason,
                &result->sidecar_load_time)) {
            result->preconditioner_setup_time =
                SPfrontEnd->IFseconds() - preconditioner_setup_started;
            strncpy(result->fallback_reason, "gmres_precond_prepare_failed",
                sizeof(result->fallback_reason) - 1);
            goto cleanup;
        }
    }
    result->preconditioner_setup_time =
        SPfrontEnd->IFseconds() - preconditioner_setup_started;
    strncpy(result->executed_precond, ngspice_gmres_precond_name(executed_mode), sizeof(result->executed_precond) - 1);
    if (precond_fallback_reason[0] != '\0')
        strncpy(result->fallback_reason, precond_fallback_reason, sizeof(result->fallback_reason) - 1);
    if (!ngspice_gmres_apply_preconditioner(
            &preconditioner, r, z, n)) {
        strncpy(result->fallback_reason, "gmres_precond_apply_failed", sizeof(result->fallback_reason) - 1);
        goto cleanup;
    }
    result->initial_precond_residual = ngspice_gmres_norm2(z, n);
    result->final_precond_residual = result->initial_precond_residual;
    initial_precond_beta = result->initial_precond_residual;

    if (result->initial_raw_residual <= raw_tol) {
        converged = 1;
        goto finalize_success;
    }

    for (cycle = 0; cycle < max_cycles && total_iters < max_iters; cycle++) {
        int j;
        int inner_iters = restart;
        memset(H, 0, (size_t) (restart + 1) * (size_t) restart * sizeof(double));
        memset(cs, 0, (size_t) restart * sizeof(double));
        memset(sn, 0, (size_t) restart * sizeof(double));
        memset(s, 0, (size_t) (restart + 1) * sizeof(double));

        current_beta = ngspice_gmres_norm2(z, n);
        if (!isfinite(current_beta) || current_beta <= DBL_EPSILON) {
            strncpy(result->fallback_reason, "gmres_breakdown",
                sizeof(result->fallback_reason) - 1);
            goto cleanup;
        }
        ngspice_gmres_copy(V, z, n);
        ngspice_gmres_scale(V, 1.0 / current_beta, n);
        s[0] = current_beta;

        for (j = 0; j < restart && total_iters < max_iters; j++) {
            int i;
            double h_ij;
            double h_diag, h_sub, denom, resid;
            double *Vj = V + (size_t) j * (size_t) n;
            double *Vjp1 = V + (size_t) (j + 1) * (size_t) n;

            ngspice_gmres_apply_matrix(&matvec_ctx, Vj, w);
            if (!ngspice_gmres_apply_preconditioner(
                    &preconditioner, w, z, n)) {
                strncpy(result->fallback_reason, "gmres_precond_apply_failed", sizeof(result->fallback_reason) - 1);
                goto cleanup;
            }
            ngspice_gmres_copy(w, z, n);

            for (i = 0; i <= j; i++) {
                double *Vi = V + (size_t) i * (size_t) n;
                h_ij = ngspice_gmres_dot(w, Vi, n);
                H[(size_t) i * (size_t) restart + (size_t) j] = h_ij;
                ngspice_gmres_axpy(w, -h_ij, Vi, n);
            }

            H[(size_t) (j + 1) * (size_t) restart + (size_t) j] = ngspice_gmres_norm2(w, n);
            if (H[(size_t) (j + 1) * (size_t) restart + (size_t) j] > DBL_EPSILON) {
                ngspice_gmres_copy(Vjp1, w, n);
                ngspice_gmres_scale(Vjp1, 1.0 / H[(size_t) (j + 1) * (size_t) restart + (size_t) j], n);
            } else {
                memset(Vjp1, 0, (size_t) n * sizeof(double));
            }

            for (i = 0; i < j; i++) {
                double temp = cs[i] * H[(size_t) i * (size_t) restart + (size_t) j] +
                              sn[i] * H[(size_t) (i + 1) * (size_t) restart + (size_t) j];
                H[(size_t) (i + 1) * (size_t) restart + (size_t) j] =
                    -sn[i] * H[(size_t) i * (size_t) restart + (size_t) j] +
                    cs[i] * H[(size_t) (i + 1) * (size_t) restart + (size_t) j];
                H[(size_t) i * (size_t) restart + (size_t) j] = temp;
            }

            h_diag = H[(size_t) j * (size_t) restart + (size_t) j];
            h_sub = H[(size_t) (j + 1) * (size_t) restart + (size_t) j];
            denom = sqrt(h_diag * h_diag + h_sub * h_sub);
            if (denom <= DBL_EPSILON) {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = h_diag / denom;
                sn[j] = h_sub / denom;
            }
            H[(size_t) j * (size_t) restart + (size_t) j] =
                cs[j] * h_diag + sn[j] * h_sub;
            H[(size_t) (j + 1) * (size_t) restart + (size_t) j] = 0.0;

            {
                double temp = cs[j] * s[j];
                s[j + 1] = -sn[j] * s[j];
                s[j] = temp;
            }

            total_iters++;
            resid = fabs(s[j + 1]);
            inner_iters = j + 1;
            if (resid <= config->atol + config->rtol * ((initial_precond_beta > 1.0) ? initial_precond_beta : 1.0))
                break;
        }

        for (j = inner_iters - 1; j >= 0; j--) {
            int i;
            double sum = s[j];
            for (i = j + 1; i < inner_iters; i++)
                sum -= H[(size_t) j * (size_t) restart + (size_t) i] * y[i];
            if (fabs(H[(size_t) j * (size_t) restart + (size_t) j]) <= DBL_EPSILON) {
                strncpy(result->fallback_reason, "gmres_breakdown", sizeof(result->fallback_reason) - 1);
                goto cleanup;
            }
            y[j] = sum / H[(size_t) j * (size_t) restart + (size_t) j];
        }

        for (j = 0; j < inner_iters; j++)
            ngspice_gmres_axpy(x, y[j], V + (size_t) j * (size_t) n, n);

        ngspice_gmres_apply_matrix(&matvec_ctx, x, r);
        for (j = 0; j < n; j++)
            r[j] = b[j] - r[j];
        result->final_raw_residual = ngspice_gmres_norm2(r, n);
        result->final_true_relative_residual =
            result->final_raw_residual / rhs_scale;
        if (!ngspice_gmres_apply_preconditioner(
                &preconditioner, r, z, n)) {
            strncpy(result->fallback_reason, "gmres_precond_apply_failed", sizeof(result->fallback_reason) - 1);
            goto cleanup;
        }
        result->final_precond_residual = ngspice_gmres_norm2(z, n);

        if (result->final_raw_residual <= raw_tol) {
            converged = 1;
            break;
        }
    }

    result->iterations = total_iters;
    result->restart_count = (total_iters > 0) ? ((total_iters - 1) / restart) : 0;
    result->converged = converged;
    if (!converged && result->fallback_reason[0] == '\0')
        strncpy(result->fallback_reason, "gmres_no_convergence", sizeof(result->fallback_reason) - 1);

    if (!converged)
        goto cleanup;

finalize_success:
    for (cycle = 0; cycle < n; cycle++)
        ckt->CKTrhs[cycle + 1] = x[cycle];
    result->success = 1;
    result->converged = 1;
    if (result->iterations == 0) {
        result->final_raw_residual = result->initial_raw_residual;
        result->final_precond_residual = result->initial_precond_residual;
    }

cleanup:
    ngspice_gmres_collect_preconditioner_metrics(&preconditioner, result);
    ngspice_gmres_destroy_preconditioner(&preconditioner);
    free(x);
    free(b);
    free(r);
    free(z);
    free(w);
    free(scales);
    free(V);
    free(H);
    free(cs);
    free(sn);
    free(s);
    free(y);
    free(matvec_ctx.in_vec);
    free(matvec_ctx.out_vec);
    return result->success;
}

#endif
