#ifndef NGSPICE_NI_ONLINE_SIDECAR_H
#define NGSPICE_NI_ONLINE_SIDECAR_H

#include <stddef.h>

#define NGSPICE_ONLINE_SIDECAR_PATH_MAX 1024
#define NGSPICE_ONLINE_SIDECAR_REASON_MAX 128

typedef struct {
    int enabled;
    int valid;
    int timeout_ms;
    int min_block_size;
    int max_block_size;
    int max_blocks;
    int reuse_existing;
    char repo_root[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char generator_path[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char checkpoint_path[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char netlist_path[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char input_dir[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char sidecar_dir[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char status_dir[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char event_log[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char failure_reason[NGSPICE_ONLINE_SIDECAR_REASON_MAX];
} ngspice_online_sidecar_config_t;

typedef struct {
    int enabled;
    int attempted;
    int success;
    int exit_code;
    int timed_out;
    double snapshot_seconds;
    double generation_seconds;
    size_t sidecar_bytes;
    char failure_reason[NGSPICE_ONLINE_SIDECAR_REASON_MAX];
    char input_path[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char jacobian_path[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char output_path[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
    char status_path[NGSPICE_ONLINE_SIDECAR_PATH_MAX];
} ngspice_online_sidecar_result_t;

void ngspice_online_sidecar_result_clear(
    ngspice_online_sidecar_result_t *result
);

void ngspice_online_sidecar_parse_config(
    int gmres_enabled,
    int learned_schwarz_requested,
    const char *sidecar_scope,
    const char *sidecar_template,
    ngspice_online_sidecar_config_t *config
);

int ngspice_online_sidecar_generate(
    const ngspice_online_sidecar_config_t *config,
    const char *system_path,
    const char *jacobian_path,
    const char *output_path,
    int newton_iter,
    double time_value,
    double gmin,
    const char *initial_guess_mode,
    ngspice_online_sidecar_result_t *result
);

#endif
