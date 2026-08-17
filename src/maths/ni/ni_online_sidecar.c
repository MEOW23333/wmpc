#include "ni_online_sidecar.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define NGSPICE_ONLINE_SIDECAR_MODE "oneshot_v1"
#define NGSPICE_ONLINE_PYTHON_PATH "/home/ZhangLexin/miniconda3/envs/PALS_env/bin/python3"
#define NGSPICE_ONLINE_DEFAULT_TIMEOUT_MS 60000
#define NGSPICE_ONLINE_MAX_TIMEOUT_MS 300000

static void
online_copy(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0U)
        return;
    if (!src)
        src = "";
    strncpy(dst, src, dst_size - 1U);
    dst[dst_size - 1U] = '\0';
}

static void
online_set_reason(char *dst, size_t dst_size, const char *reason)
{
    online_copy(dst, dst_size, reason && reason[0] ? reason : "online_unspecified");
}

static double
online_seconds_now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec * 1.0e-6;
}

static int
online_is_directory(const char *path)
{
    struct stat st;
    return path && path[0] != '\0' && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int
online_is_regular_file(const char *path, size_t *bytes)
{
    struct stat st;
    if (!path || path[0] == '\0' || stat(path, &st) != 0 || !S_ISREG(st.st_mode))
        return 0;
    if (bytes)
        *bytes = st.st_size > 0 ? (size_t) st.st_size : 0U;
    return 1;
}

static int
online_path_within(const char *root, const char *path)
{
    size_t root_len;
    if (!root || !path || root[0] == '\0' || path[0] == '\0')
        return 0;
    root_len = strlen(root);
    if (strncmp(root, path, root_len) != 0)
        return 0;
    return path[root_len] == '\0' || path[root_len] == '/';
}

static int
online_canonical_directory(const char *input, char output[NGSPICE_ONLINE_SIDECAR_PATH_MAX])
{
    char *resolved;
    if (!input || input[0] == '\0')
        return 0;
    resolved = realpath(input, output);
    return resolved != NULL && online_is_directory(output);
}

static int
online_canonical_regular_file(const char *input, char output[NGSPICE_ONLINE_SIDECAR_PATH_MAX])
{
    char *resolved;
    if (!input || input[0] == '\0')
        return 0;
    resolved = realpath(input, output);
    return resolved != NULL && online_is_regular_file(output, NULL);
}

static int
online_parent_directory(
    const char *path,
    char output[NGSPICE_ONLINE_SIDECAR_PATH_MAX]
)
{
    char copy[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char *slash;
    if (!path || path[0] == '\0' || strlen(path) >= sizeof(copy))
        return 0;
    online_copy(copy, sizeof(copy), path);
    slash = strrchr(copy, '/');
    if (!slash || slash == copy)
        return 0;
    *slash = '\0';
    return online_canonical_directory(copy, output);
}

static int
online_make_canonical_child(
    const char *raw_path,
    char output[NGSPICE_ONLINE_SIDECAR_PATH_MAX]
)
{
    char parent[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    const char *name;
    int written;
    if (!raw_path || raw_path[0] == '\0')
        return 0;
    name = strrchr(raw_path, '/');
    if (!name || name[1] == '\0' || strcmp(name + 1, ".") == 0 || strcmp(name + 1, "..") == 0)
        return 0;
    if (!online_parent_directory(raw_path, parent))
        return 0;
    written = snprintf(output, NGSPICE_ONLINE_SIDECAR_PATH_MAX, "%s/%s", parent, name + 1);
    return written > 0 && (size_t) written < NGSPICE_ONLINE_SIDECAR_PATH_MAX;
}

static int
online_parse_bounded_int(
    const char *name,
    int default_value,
    int minimum,
    int maximum,
    int *out
)
{
    const char *raw = getenv(name);
    char *end = NULL;
    long parsed;
    if (!out)
        return 0;
    if (!raw || raw[0] == '\0') {
        *out = default_value;
        return 1;
    }
    errno = 0;
    parsed = strtol(raw, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed < minimum || parsed > maximum)
        return 0;
    *out = (int) parsed;
    return 1;
}

static int
online_set_required_file(
    const char *name,
    const char *repo_root,
    char output[NGSPICE_ONLINE_SIDECAR_PATH_MAX]
)
{
    const char *raw = getenv(name);
    if (!online_canonical_regular_file(raw, output))
        return 0;
    return online_path_within(repo_root, output);
}

static int
online_set_required_directory(
    const char *name,
    const char *repo_root,
    char output[NGSPICE_ONLINE_SIDECAR_PATH_MAX]
)
{
    const char *raw = getenv(name);
    if (!online_canonical_directory(raw, output))
        return 0;
    return online_path_within(repo_root, output);
}

static int
online_template_is_per_step(const char *template_path)
{
    return template_path && strstr(template_path, "{iter}") &&
           strstr(template_path, "{time}") && strstr(template_path, "{gmin}");
}

void
ngspice_online_sidecar_result_clear(ngspice_online_sidecar_result_t *result)
{
    if (!result)
        return;
    memset(result, 0, sizeof(*result));
    result->exit_code = -1;
}

void
ngspice_online_sidecar_parse_config(
    int gmres_enabled,
    int learned_schwarz_requested,
    const char *sidecar_scope,
    const char *sidecar_template,
    ngspice_online_sidecar_config_t *config
)
{
    const char *mode;
    const char *event_log;
    const char *reuse_raw;
    if (!config)
        return;
    memset(config, 0, sizeof(*config));
    mode = getenv("PALS_ONLINE_SCHWARZ_MODE");
    if (!mode || mode[0] == '\0')
        return;
    config->enabled = 1;
    if (strcmp(mode, NGSPICE_ONLINE_SIDECAR_MODE) != 0) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_invalid_mode");
        return;
    }
    if (!gmres_enabled || !learned_schwarz_requested) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_requires_learned_schwarz");
        return;
    }
    if (!sidecar_scope || strcmp(sidecar_scope, "per_step") != 0 ||
        !online_template_is_per_step(sidecar_template)) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_invalid_sidecar_template");
        return;
    }
    reuse_raw = getenv("PALS_ONLINE_SCHWARZ_REUSE_EXISTING");
    if (reuse_raw && strcmp(reuse_raw, "0") != 0 && strcmp(reuse_raw, "1") != 0) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_invalid_reuse_flag");
        return;
    }
    config->reuse_existing = reuse_raw && strcmp(reuse_raw, "1") == 0;
    if (!online_canonical_directory(getenv("PALS_ONLINE_SCHWARZ_REPO_ROOT"), config->repo_root)) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_invalid_repo_root");
        return;
    }
    if (!online_set_required_file("PALS_ONLINE_SCHWARZ_GENERATOR", config->repo_root, config->generator_path) ||
        !online_set_required_file("PALS_ONLINE_SCHWARZ_CHECKPOINT", config->repo_root, config->checkpoint_path) ||
        !online_set_required_file("PALS_ONLINE_SCHWARZ_NETLIST", config->repo_root, config->netlist_path) ||
        !online_set_required_directory("PALS_ONLINE_SCHWARZ_INPUT_DIR", config->repo_root, config->input_dir) ||
        !online_set_required_directory("PALS_ONLINE_SCHWARZ_SIDECAR_DIR", config->repo_root, config->sidecar_dir) ||
        !online_set_required_directory("PALS_ONLINE_SCHWARZ_STATUS_DIR", config->repo_root, config->status_dir)) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_invalid_path");
        return;
    }
    event_log = getenv("PALS_ONLINE_SCHWARZ_EVENT_LOG");
    if (!online_make_canonical_child(event_log, config->event_log) ||
        !online_path_within(config->repo_root, config->event_log)) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_invalid_event_log");
        return;
    }
    if (!online_parse_bounded_int("PALS_ONLINE_SCHWARZ_TIMEOUT_MS", NGSPICE_ONLINE_DEFAULT_TIMEOUT_MS, 1000, NGSPICE_ONLINE_MAX_TIMEOUT_MS, &config->timeout_ms) ||
        !online_parse_bounded_int("PALS_ONLINE_SCHWARZ_MIN_BLOCK_SIZE", 2, 2, 32, &config->min_block_size) ||
        !online_parse_bounded_int("PALS_ONLINE_SCHWARZ_MAX_BLOCK_SIZE", 32, 2, 32, &config->max_block_size) ||
        !online_parse_bounded_int("PALS_ONLINE_SCHWARZ_MAX_BLOCKS", 0, 0, 1000000, &config->max_blocks) ||
        config->min_block_size > config->max_block_size) {
        online_set_reason(config->failure_reason, sizeof(config->failure_reason), "online_config_invalid_limits");
        return;
    }
    config->valid = 1;
}

static void
online_json_string(FILE *fp, const char *value)
{
    const unsigned char *cursor = (const unsigned char *) (value ? value : "");
    fputc('"', fp);
    while (*cursor) {
        unsigned char ch = *cursor++;
        if (ch == '"')
            fputs("\\\"", fp);
        else if (ch == '\\')
            fputs("\\\\", fp);
        else if (ch == '\n')
            fputs("\\n", fp);
        else if (ch == '\r')
            fputs("\\r", fp);
        else if (ch == '\t')
            fputs("\\t", fp);
        else if (ch < 0x20U)
            fprintf(fp, "\\u%04x", (unsigned int) ch);
        else
            fputc((int) ch, fp);
    }
    fputc('"', fp);
}

static void
online_append_event(
    const ngspice_online_sidecar_config_t *config,
    const ngspice_online_sidecar_result_t *result,
    int newton_iter,
    double time_value,
    double gmin
)
{
    int saved_errno = errno;
    FILE *fp;
    if (!config || !result || config->event_log[0] == '\0')
        return;
    fp = fopen(config->event_log, "a");
    if (!fp) {
        errno = saved_errno;
        return;
    }
    fprintf(fp, "{\"event\":\"online_sidecar_generation\",\"newton_iter\":%d,\"time\":%.17e,\"gmin\":%.17e,\"enabled\":%s,\"attempted\":%s,\"success\":%s,\"exit_code\":%d,\"timed_out\":%s,\"snapshot_seconds\":%.17e,\"generation_seconds\":%.17e,\"sidecar_bytes\":%llu,\"failure_reason\":", newton_iter, time_value, gmin, result->enabled ? "true" : "false", result->attempted ? "true" : "false", result->success ? "true" : "false", result->exit_code, result->timed_out ? "true" : "false", result->snapshot_seconds, result->generation_seconds, (unsigned long long) result->sidecar_bytes);
    online_json_string(fp, result->failure_reason);
    fputs(",\"input_path\":", fp);
    online_json_string(fp, result->input_path);
    fputs(",\"jacobian_path\":", fp);
    online_json_string(fp, result->jacobian_path);
    fputs(",\"output_path\":", fp);
    online_json_string(fp, result->output_path);
    fputs(",\"status_path\":", fp);
    online_json_string(fp, result->status_path);
    fputs("}\n", fp);
    fclose(fp);
    errno = saved_errno;
}

static int
online_parent_equals(const char *path, const char *expected_parent)
{
    char actual_parent[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    return online_parent_directory(path, actual_parent) &&
           strcmp(actual_parent, expected_parent) == 0;
}

static int
online_status_reports_success(const char *path)
{
    char buffer[4097];
    size_t count;
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;
    count = fread(buffer, 1U, sizeof(buffer) - 1U, fp);
    fclose(fp);
    buffer[count] = '\0';
    return strstr(buffer, "\"success\": true") != NULL ||
           strstr(buffer, "\"success\":true") != NULL;
}

int
ngspice_online_sidecar_generate(
    const ngspice_online_sidecar_config_t *config,
    const char *system_path,
    const char *jacobian_path,
    const char *output_path,
    int newton_iter,
    double time_value,
    double gmin,
    const char *initial_guess_mode,
    ngspice_online_sidecar_result_t *result
)
{
    char time_buffer[64];
    char gmin_buffer[64];
    char iter_buffer[32];
    char min_block_buffer[32];
    char max_block_buffer[32];
    char max_blocks_buffer[32];
    char *argv[29];
    pid_t pid;
    int wait_status = 0;
    double started;
    int written;
    size_t bytes = 0U;

    if (!result)
        return 0;
    result->enabled = config && config->enabled;
    if (!config || !config->enabled || !config->valid) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), config ? config->failure_reason : "online_config_missing");
        return 0;
    }
    if (!system_path || !jacobian_path || !output_path || newton_iter < 1 ||
        !initial_guess_mode || initial_guess_mode[0] == '\0') {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_input_invalid");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    if (!online_is_regular_file(system_path, NULL) ||
        !online_is_regular_file(jacobian_path, NULL) ||
        !online_parent_equals(system_path, config->input_dir) ||
        !online_parent_equals(jacobian_path, config->input_dir) ||
        !online_parent_equals(output_path, config->sidecar_dir)) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_snapshot_or_output_path_invalid");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    online_copy(result->input_path, sizeof(result->input_path), system_path);
    online_copy(result->jacobian_path, sizeof(result->jacobian_path), jacobian_path);
    online_copy(result->output_path, sizeof(result->output_path), output_path);
    written = snprintf(result->status_path, sizeof(result->status_path), "%s/sidecar_iter_%d_time_%.17e_gmin_%.17e.json", config->status_dir, newton_iter, time_value, gmin);
    if (written <= 0 || (size_t) written >= sizeof(result->status_path) ||
        !online_parent_equals(result->status_path, config->status_dir)) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_status_path_invalid");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    if (config->reuse_existing) {
        if (online_is_regular_file(output_path, &bytes) && bytes >= 2U) {
            result->attempted = 0;
            result->success = 1;
            result->exit_code = 0;
            result->generation_seconds = 0.0;
            result->sidecar_bytes = bytes;
            result->failure_reason[0] = '\0';
            online_append_event(config, result, newton_iter, time_value, gmin);
            return 1;
        }
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "precomputed_sidecar_missing");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    if ((unlink(output_path) != 0 && errno != ENOENT) ||
        (unlink(result->status_path) != 0 && errno != ENOENT)) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_stale_output_remove_failed");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }

    snprintf(time_buffer, sizeof(time_buffer), "%.17e", time_value);
    snprintf(gmin_buffer, sizeof(gmin_buffer), "%.17e", gmin);
    snprintf(iter_buffer, sizeof(iter_buffer), "%d", newton_iter);
    snprintf(min_block_buffer, sizeof(min_block_buffer), "%d", config->min_block_size);
    snprintf(max_block_buffer, sizeof(max_block_buffer), "%d", config->max_block_size);
    snprintf(max_blocks_buffer, sizeof(max_blocks_buffer), "%d", config->max_blocks);
    argv[0] = (char *) NGSPICE_ONLINE_PYTHON_PATH;
    argv[1] = (char *) config->generator_path;
    argv[2] = "--system-path";
    argv[3] = (char *) system_path;
    argv[4] = "--jacobian-path";
    argv[5] = (char *) jacobian_path;
    argv[6] = "--netlist-path";
    argv[7] = (char *) config->netlist_path;
    argv[8] = "--checkpoint";
    argv[9] = (char *) config->checkpoint_path;
    argv[10] = "--output-path";
    argv[11] = (char *) output_path;
    argv[12] = "--status-path";
    argv[13] = result->status_path;
    argv[14] = "--time";
    argv[15] = time_buffer;
    argv[16] = "--gmin";
    argv[17] = gmin_buffer;
    argv[18] = "--newton-iter";
    argv[19] = iter_buffer;
    argv[20] = "--initial-guess-mode";
    argv[21] = (char *) initial_guess_mode;
    argv[22] = "--min-block-size";
    argv[23] = min_block_buffer;
    argv[24] = "--max-block-size";
    argv[25] = max_block_buffer;
    argv[26] = "--max-blocks";
    argv[27] = max_blocks_buffer;
    argv[28] = NULL;

    result->attempted = 1;
    started = online_seconds_now();
    pid = fork();
    if (pid < 0) {
        result->generation_seconds = online_seconds_now() - started;
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_spawn_failed");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    if (pid == 0) {
        setenv("OMP_NUM_THREADS", "1", 1);
        setenv("MKL_NUM_THREADS", "1", 1);
        setenv("OPENBLAS_NUM_THREADS", "1", 1);
        setenv("PYTHONUNBUFFERED", "1", 1);
        execv(NGSPICE_ONLINE_PYTHON_PATH, argv);
        _exit(127);
    }
    for (;;) {
        pid_t waited = waitpid(pid, &wait_status, WNOHANG);
        double elapsed = online_seconds_now() - started;
        if (waited == pid)
            break;
        if (waited < 0) {
            online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_wait_failed");
            break;
        }
        if (elapsed * 1000.0 > (double) config->timeout_ms) {
            kill(pid, SIGKILL);
            waitpid(pid, &wait_status, 0);
            result->timed_out = 1;
            online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_timeout");
            break;
        }
        usleep(10000U);
    }
    result->generation_seconds = online_seconds_now() - started;
    if (result->timed_out) {
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    if (!WIFEXITED(wait_status)) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_generator_signaled");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    result->exit_code = WEXITSTATUS(wait_status);
    if (result->exit_code != 0) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_generator_exit_nonzero");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    if (!online_is_regular_file(output_path, &bytes) || bytes < 2U) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_output_missing");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    if (!online_is_regular_file(result->status_path, NULL) ||
        !online_status_reports_success(result->status_path)) {
        online_set_reason(result->failure_reason, sizeof(result->failure_reason), "online_status_invalid");
        online_append_event(config, result, newton_iter, time_value, gmin);
        return 0;
    }
    result->sidecar_bytes = bytes;
    result->success = 1;
    result->failure_reason[0] = '\0';
    online_append_event(config, result, newton_iter, time_value, gmin);
    return 1;
}
