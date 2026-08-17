#ifndef NGSPICE_NI_GMRES_SCHWARZ_H
#define NGSPICE_NI_GMRES_SCHWARZ_H

#include <stddef.h>

#include "ngspice/cktdefs.h"

typedef struct ngspice_gmres_schwarz_state ngspice_gmres_schwarz_state_t;

typedef struct {
    int matrix_size;
    int block_count;
    int total_block_rows;
    int covered_rows;
    int uncovered_rows;
    int max_block_size;
    int apply_count;
    int failed_apply_count;
    double gmin;
    double lambda_min;
    double lambda_max;
    double sidecar_load_time;
    double factor_time;
    double setup_time;
    double apply_time_total;
    size_t sidecar_file_bytes;
    size_t layout_bytes;
    size_t parameter_bytes;
    size_t factor_bytes;
    size_t fallback_bytes;
    size_t workspace_bytes;
    size_t retained_bytes;
    size_t peak_estimated_bytes;
} ngspice_gmres_schwarz_metrics_t;

int ngspice_gmres_schwarz_create(
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
);

int ngspice_gmres_schwarz_apply(
    ngspice_gmres_schwarz_state_t *state,
    const double *rhs,
    double *out,
    int count
);

void ngspice_gmres_schwarz_get_metrics(
    const ngspice_gmres_schwarz_state_t *state,
    ngspice_gmres_schwarz_metrics_t *out
);

void ngspice_gmres_schwarz_destroy(
    ngspice_gmres_schwarz_state_t *state
);

#endif
