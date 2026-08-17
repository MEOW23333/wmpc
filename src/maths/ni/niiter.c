/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2001 AlansFixes
**********/

/*
 * NIiter(ckt,maxIter)
 *
 *  This subroutine performs the actual numerical iteration.
 *  It uses the sparse matrix stored in the circuit struct
 *  along with the matrix loading program, the load data, the
 *  convergence test function, and the convergence parameters
 */

#include "ngspice/ngspice.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"
#include "ngspice/spmatrix.h"  // 为了使用 spSetReal
#include "ni_gmres_helpers.h"
#include "ni_online_sidecar.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> // <--- 新增，为了 gettimeofday

/* External flag to indicate if currently in gmin stepping */

/* Limit the number of 'singular matrix' warnings */
static int msgcount = 0;
static int continuation_consumed = 0;
static int continuation_active = 0;
static int continuation_step_counter = 0;
static int linear_system_corpus_klu_warned = 0;

static int
continuation_load_inputs(CKTcircuit *ckt, const char *wp_in_str, const char *state0_path_str)
{
    int i;
    FILE *fp_in = wp_in_str ? fopen(wp_in_str, "r") : NULL;
    if (!fp_in)
        return 0;

    for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
        if (fscanf(fp_in, "%le", &ckt->CKTrhsOld[i]) != 1) {
            ckt->CKTrhsOld[i] = 0.0;
        }
    }
    fclose(fp_in);

    if (state0_path_str && ckt->CKTstate0) {
        FILE *state_fp_in = fopen(state0_path_str, "r");
        if (state_fp_in) {
            for (i = 0; i < ckt->CKTnumStates; i++) {
                if (fscanf(state_fp_in, "%le", &ckt->CKTstate0[i]) != 1) {
                    ckt->CKTstate0[i] = 0.0;
                }
            }
            fclose(state_fp_in);
        }
    }
    return 1;
}

static void
linear_system_corpus_write_vector(FILE *fp, const char *section, const double *values, int count)
{
    int idx;
    if (!fp || !section)
        return;

    fprintf(fp, "*************%s*************\n", section);
    if (!values || count <= 0)
        return;
    for (idx = 0; idx < count; idx++)
        fprintf(fp, "%.17e\n", values[idx]);
}

static void
linear_system_corpus_write_metadata(
    FILE *fp,
    CKTcircuit *ckt,
    const char *ckt_id_str,
    int matrix_size,
    int iterno
)
{
    if (!fp)
        return;

    fprintf(fp, "*************META*************\n");
    fprintf(fp, "circuit_id %s\n", ckt_id_str ? ckt_id_str : "default");
    fprintf(fp, "time %.17e\n", ckt->CKTtime);
    fprintf(fp, "gmin %.17e\n", ckt->CKTdiagGmin);
    fprintf(fp, "iteration %d\n", iterno);
    fprintf(fp, "matrix_size %d\n", matrix_size);
    fprintf(fp, "solver sparse\n");
}
static void
gmres_metrics_json_write_string(FILE *fp, const char *value)
{
    const unsigned char *cursor =
        (const unsigned char *) (value ? value : "");

    fputc('"', fp);
    while (*cursor) {
        unsigned char ch = *cursor++;
        switch (ch) {
        case '"':
            fputs("\\\"", fp);
            break;
        case '\\':
            fputs("\\\\", fp);
            break;
        case '\b':
            fputs("\\b", fp);
            break;
        case '\f':
            fputs("\\f", fp);
            break;
        case '\n':
            fputs("\\n", fp);
            break;
        case '\r':
            fputs("\\r", fp);
            break;
        case '\t':
            fputs("\\t", fp);
            break;
        default:
            if (ch < 0x20U)
                fprintf(fp, "\\u%04x", (unsigned int) ch);
            else
                fputc((int) ch, fp);
            break;
        }
    }
    fputc('"', fp);
}

static void
gmres_metrics_json_write_double(FILE *fp, double value)
{
    if (isfinite(value))
        fprintf(fp, "%.17e", value);
    else
        fputs("null", fp);
}

static void
gmres_metrics_append_jsonl(
    CKTcircuit *ckt,
    const ngspice_gmres_config_t *config,
    int newton_iter,
    int matrix_size,
    int gmres_success,
    double gmres_solve_time,
    const ngspice_gmres_result_t *result
)
{
    const char *metrics_path = getenv("NGSPICE_GMRES_METRICS_PATH");
    const char *fallback_reason;
    const char *direct_fallback_reason;
    double coverage_ratio = 0.0;
    double matrix_density = 0.0;
    int coverage_available = 0;
    int matrix_structural_nnz = 0;
    int strict_true_relative_residual_pass = 0;
    int saved_errno;
    FILE *metrics_fp;

    if (!metrics_path || metrics_path[0] == '\0' || !ckt || !config || !result)
        return;

    saved_errno = errno;
    metrics_fp = fopen(metrics_path, "a");
    if (!metrics_fp) {
        errno = saved_errno;
        return;
    }

    fallback_reason = result->fallback_reason[0]
        ? result->fallback_reason
        : "";
    direct_fallback_reason = gmres_success
        ? ""
        : (fallback_reason[0] ? fallback_reason : "unspecified");
    matrix_structural_nnz =
        (ckt->CKTmatrix && ckt->CKTmatrix->SPmatrix)
            ? spElementCount(ckt->CKTmatrix->SPmatrix)
            : 0;
    if (matrix_size > 0 && matrix_structural_nnz >= 0)
        matrix_density =
            (double) matrix_structural_nnz /
            ((double) matrix_size * (double) matrix_size);
    strict_true_relative_residual_pass =
        gmres_success &&
        isfinite(result->final_true_relative_residual) &&
        result->final_true_relative_residual <= config->rtol;
    if (result->preconditioner_covered_rows >= 0 &&
        result->preconditioner_uncovered_rows >= 0 &&
        result->preconditioner_covered_rows +
            result->preconditioner_uncovered_rows > 0) {
        coverage_ratio =
            (double) result->preconditioner_covered_rows /
            (double) (result->preconditioner_covered_rows +
                      result->preconditioner_uncovered_rows);
        coverage_available = 1;
    }

    fprintf(
        metrics_fp,
        "{\"schema_version\":1,\"event\":\"%s\","
        "\"newton_iteration\":%d,\"matrix_size\":%d,"
        "\"matrix_structural_nnz\":%d,\"sparse_solver_active\":true,"
        "\"gmres_success\":%s,\"gmres_converged\":%s,"
        "\"strict_true_relative_residual_pass\":%s,"
        "\"gmres_iterations\":%d,\"gmres_restart_count\":%d,"
        "\"direct_fallback\":%s",
        gmres_success ? "gmres_converged" : "fallback_to_direct",
        newton_iter,
        matrix_size,
        matrix_structural_nnz,
        gmres_success ? "true" : "false",
        result->converged ? "true" : "false",
        strict_true_relative_residual_pass ? "true" : "false",
        result->iterations,
        result->restart_count,
        gmres_success ? "false" : "true"
    );
    fputs(",\"matrix_density\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, matrix_density);

    fputs(",\"requested_preconditioner_mode\":", metrics_fp);
    gmres_metrics_json_write_string(
        metrics_fp,
        ngspice_gmres_precond_name(config->requested_precond));
    fputs(",\"executed_preconditioner_mode\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, result->executed_precond);
    fputs(",\"preconditioner_fallback_reason\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, fallback_reason);
    fputs(",\"direct_fallback_reason\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, direct_fallback_reason);
    fputs(",\"resolved_sidecar_path\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, result->resolved_sidecar_path);
    fprintf(metrics_fp, ",\"online_sidecar_enabled\":%s,\"online_sidecar_success\":%s,\"online_sidecar_exit_code\":%d,\"online_sidecar_timed_out\":%s,\"online_sidecar_bytes\":%llu", result->online_sidecar_enabled ? "true" : "false", result->online_sidecar_success ? "true" : "false", result->online_sidecar_exit_code, result->online_sidecar_timed_out ? "true" : "false", (unsigned long long) result->online_sidecar_bytes);
    fputs(",\"online_sidecar_failure_reason\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, result->online_sidecar_failure_reason);
    fputs(",\"online_sidecar_input_path\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, result->online_sidecar_input_path);
    fputs(",\"online_sidecar_jacobian_path\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, result->online_sidecar_jacobian_path);
    fputs(",\"online_sidecar_output_path\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, result->online_sidecar_output_path);
    fputs(",\"online_sidecar_status_path\":", metrics_fp);
    gmres_metrics_json_write_string(metrics_fp, result->online_sidecar_status_path);
    fputs(",\"online_sidecar_snapshot_seconds\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->online_sidecar_snapshot_seconds);
    fputs(",\"online_sidecar_generation_seconds\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->online_sidecar_generation_seconds);

    fputs(",\"circuit_time\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, ckt->CKTtime);
    fputs(",\"gmin\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, ckt->CKTdiagGmin);
    fputs(",\"gmres_solve_time_seconds\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, gmres_solve_time);
    fputs(",\"initial_raw_residual\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->initial_raw_residual);
    fputs(",\"final_raw_residual\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->final_raw_residual);
    fputs(",\"rhs_norm\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->rhs_norm);
    fputs(",\"initial_true_relative_residual\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->initial_true_relative_residual);
    fputs(",\"final_true_relative_residual\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->final_true_relative_residual);
    fputs(",\"initial_preconditioned_residual\":", metrics_fp);
    gmres_metrics_json_write_double(
        metrics_fp, result->initial_precond_residual);
    fputs(",\"final_preconditioned_residual\":", metrics_fp);
    gmres_metrics_json_write_double(
        metrics_fp, result->final_precond_residual);

    fputs(",\"schwarz_read_time_seconds\":", metrics_fp);
    gmres_metrics_json_write_double(metrics_fp, result->sidecar_load_time);
    fputs(",\"schwarz_setup_time_seconds\":", metrics_fp);
    gmres_metrics_json_write_double(
        metrics_fp, result->preconditioner_setup_time);
    fputs(",\"schwarz_factor_time_seconds\":", metrics_fp);
    gmres_metrics_json_write_double(
        metrics_fp, result->preconditioner_factor_time);
    fputs(",\"schwarz_apply_time_seconds\":", metrics_fp);
    gmres_metrics_json_write_double(
        metrics_fp, result->preconditioner_apply_time);

    fprintf(
        metrics_fp,
        ",\"preconditioner_apply_count\":%d,"
        "\"schwarz_failed_apply_count\":%d,"
        "\"schwarz_block_count\":%d,"
        "\"schwarz_total_block_rows\":%d,"
        "\"schwarz_max_block_size\":%d,"
        "\"schwarz_failed_block_count\":null,"
        "\"schwarz_covered_rows\":%d,"
        "\"schwarz_uncovered_rows\":%d,"
        "\"preconditioner_sidecar_file_bytes\":%llu,"
        "\"preconditioner_layout_bytes\":%llu,"
        "\"preconditioner_parameter_bytes\":%llu,"
        "\"preconditioner_retained_bytes\":%llu,"
        "\"preconditioner_peak_estimated_bytes\":%llu,"
        "\"preconditioner_factor_bytes\":%llu,"
        "\"preconditioner_fallback_bytes\":%llu,"
        "\"preconditioner_workspace_bytes\":%llu,"
        "\"gmres_workspace_bytes\":%llu",
        result->preconditioner_apply_count,
        result->preconditioner_failed_apply_count,
        result->preconditioner_block_count,
        result->preconditioner_total_block_rows,
        result->preconditioner_max_block_size,
        result->preconditioner_covered_rows,
        result->preconditioner_uncovered_rows,
        (unsigned long long) result->preconditioner_sidecar_file_bytes,
        (unsigned long long) result->preconditioner_layout_bytes,
        (unsigned long long) result->preconditioner_parameter_bytes,
        (unsigned long long) result->preconditioner_retained_bytes,
        (unsigned long long) result->preconditioner_peak_estimated_bytes,
        (unsigned long long) result->preconditioner_factor_bytes,
        (unsigned long long) result->preconditioner_fallback_bytes,
        (unsigned long long) result->preconditioner_workspace_bytes,
        (unsigned long long) result->gmres_workspace_bytes
    );
    fputs(",\"schwarz_coverage_ratio\":", metrics_fp);
    if (coverage_available)
        gmres_metrics_json_write_double(metrics_fp, coverage_ratio);
    else
        fputs("null", metrics_fp);
    fputs("}\n", metrics_fp);

    (void) fclose(metrics_fp);
    errno = saved_errno;
}

/* NIiter() - return value is non-zero for convergence failure */

int
NIiter(CKTcircuit *ckt, int maxIter)
{
    double startTime, *OldCKTstate0 = NULL;
    int error, i, j;

    int iterno = 0;
    int ipass = 0;

    const char *if_traj = getenv("TRAJ");
    const char *if_value = getenv("VALUE");
    const char *if_close_loop_train = getenv("CLOSE_LOOP_TRAIN");
    const char *if_continuation = getenv("CONTINUATION_MODE");
    const char *trace_niiter = getenv("PALS_NIITER_TRACE");
    const char *traj_start_iter_str = getenv("PALS_TRAJ_START_ITER");
    const char *continuation_start_iter_str = getenv("CONTINUATION_START_ITER");
    const char *continuation_max_steps_str = getenv("CONTINUATION_MAX_STEPS");
    const char *continuation_dir_str = getenv("CONTINUATION_DIR");
    const char *continuation_gmin_str = getenv("CONTINUATION_GMIN");
    const char *continuation_state0_path_str = getenv("CONTINUATION_STATE0_PATH");
    const char *continuation_reapply_wp_steps_str = getenv("CONTINUATION_REAPPLY_WP_STEPS");
    const char *segment_warmup_residual_path = getenv("SEGMENT_WARMUP_RESIDUAL_PATH");
    const char *linear_system_corpus_mode = getenv("LINEAR_SYSTEM_CORPUS_MODE");
    const char *linear_system_corpus_dir = getenv("LINEAR_SYSTEM_CORPUS_DIR");
    char filename[512] = "";
    char jac_filename[512] = "";
    char linear_system_filename[512] = "";
    char linear_system_jac_filename[512] = "";
    FILE *fp = NULL;
    FILE *linear_system_fp = NULL;
    int continuation_enabled = 0;
    int continuation_start_iter = 0;
    int continuation_max_steps = 0;
    int continuation_stop_iter = -1;
    int continuation_applied = 0;
    int continuation_capture = 0;
    int continuation_step_index = -1;
    int continuation_has_gmin = 0;
    int continuation_reapply_wp_steps = 0;
    int linear_system_corpus_enabled = 0;
    int linear_system_corpus_active = 0;
    int trace_enabled = trace_niiter && strcmp(trace_niiter, "1") == 0;
    int traj_start_iter = 0;
    ngspice_gmres_config_t gmres_config;
    ngspice_online_sidecar_config_t online_sidecar_config;
    double continuation_gmin = 0.0;

    if (traj_start_iter_str) {
        traj_start_iter = atoi(traj_start_iter_str);
        if (traj_start_iter < 0)
            traj_start_iter = 0;
    }

    ngspice_gmres_parse_config(&gmres_config);

    ngspice_online_sidecar_parse_config(
        gmres_config.enabled,
        gmres_config.requested_precond == NGSPICE_GMRES_PRECOND_LEARNED_SCHWARZ,
        gmres_config.sidecar_scope,
        gmres_config.sidecar_path,
        &online_sidecar_config
    );

    if (
        linear_system_corpus_mode &&
        strcmp(linear_system_corpus_mode, "1") == 0 &&
        linear_system_corpus_dir &&
        linear_system_corpus_dir[0] != '\0'
    ) {
        linear_system_corpus_enabled = 1;
    }
    if (online_sidecar_config.enabled && online_sidecar_config.valid) {
        if (linear_system_corpus_enabled &&
            strcmp(linear_system_corpus_dir, online_sidecar_config.input_dir) != 0) {
            online_sidecar_config.valid = 0;
            snprintf(
                online_sidecar_config.failure_reason,
                sizeof(online_sidecar_config.failure_reason),
                "%s",
                "online_config_corpus_dir_conflict"
            );
        } else {
            linear_system_corpus_enabled = 1;
            linear_system_corpus_dir = online_sidecar_config.input_dir;
        }
    }

    if (if_continuation && strcmp(if_continuation, "1") == 0) {
        continuation_enabled = 1;
        if (continuation_start_iter_str)
            continuation_start_iter = atoi(continuation_start_iter_str);
        if (continuation_max_steps_str)
            continuation_max_steps = atoi(continuation_max_steps_str);
        if (continuation_max_steps > 0)
            continuation_stop_iter = continuation_start_iter + continuation_max_steps - 1;
        if (continuation_gmin_str) {
            continuation_gmin = atof(continuation_gmin_str);
            continuation_has_gmin = 1;
        }
        if (continuation_reapply_wp_steps_str)
            continuation_reapply_wp_steps = atoi(continuation_reapply_wp_steps_str);
    }

    // printf(">>>> NIiter CALLED at time = %g <<<<\n", ckt->CKTtime);
    fflush(stdout);


    /* some convergence issues that get resolved by increasing max iter */
    if (maxIter < 100)
        maxIter = 100;

    if ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
        if (trace_enabled) {
            fprintf(
                stderr,
                "PALS_NIITER_TRACE event=uic_tranop_fast_return time=%.17e mode=0x%lx\n",
                ckt->CKTtime,
                ckt->CKTmode
            );
        }
        SWAP(double *, ckt->CKTrhs, ckt->CKTrhsOld);
        error = CKTload(ckt);
        if (error)
            return(error);
        return(OK);
    }

#ifdef WANT_SENSE2
    if (ckt->CKTsenInfo) {
        error = NIsenReinit(ckt);
        if (error)
            return(error);
    }
#endif

    if (ckt->CKTniState & NIUNINITIALIZED) {
        error = NIreinit(ckt); /* always returns 0 */
        if (error) {
#ifdef STEPDEBUG
            printf("re-init returned error \n");
#endif
            return(error);
        }
    }

    /* OldCKTstate0 = TMALLOC(double, ckt->CKTnumStates + 1); */

    for (;;) {
        ckt->CKTnoncon = 0;

#ifdef NEWPRED
        if (!(ckt->CKTmode & MODEINITPRED))
#endif
        {
            int direct_op = 0;
            int continuation_capture_pending = 0;
            double online_snapshot_started = 0.0;
            ngspice_online_sidecar_result_t online_sidecar_result;
            ngspice_online_sidecar_result_clear(&online_sidecar_result);
            online_sidecar_result.enabled = online_sidecar_config.enabled;
            if (online_sidecar_config.enabled && !online_sidecar_config.valid) {
                snprintf(
                    online_sidecar_result.failure_reason,
                    sizeof(online_sidecar_result.failure_reason),
                    "%s",
                    online_sidecar_config.failure_reason[0]
                        ? online_sidecar_config.failure_reason
                        : "online_config_invalid"
                );
            }

            if (
                continuation_enabled &&
                !continuation_consumed &&
                !continuation_applied &&
                continuation_dir_str &&
                iterno == continuation_start_iter
            ) {
                const char *wp_in_str = getenv("WP_IN_PATH");
                if (continuation_load_inputs(ckt, wp_in_str, continuation_state0_path_str)) {
                    continuation_applied = 1;
                    continuation_consumed = 1;
                    continuation_active = 1;
                    continuation_step_counter = 0;
                }
            } else if (
                continuation_enabled &&
                continuation_active &&
                continuation_reapply_wp_steps > 0 &&
                continuation_step_counter < continuation_reapply_wp_steps
            ) {
                const char *wp_in_str = getenv("WP_IN_PATH");
                continuation_load_inputs(ckt, wp_in_str, continuation_state0_path_str);
            } else if (if_value && strcmp(if_value, "1") == 0 && iterno > 4 && iterno < 100) {

                // ---  从文件读入工作点 V(k-1) ---
                const char *wp_in_str = getenv("WP_IN_PATH");
                FILE *fp_in = fopen(wp_in_str, "r");
                if (fp_in) {
                    for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                        if (fscanf(fp_in, "%le", &ckt->CKTrhsOld[i]) != 1) {
                            ckt->CKTrhsOld[i] = 0.0;
                        }
                    }
                    fclose(fp_in);
                    
                    // // 将 V(k-1) 同时放入 CKTrhs，因为第一个时间点的 DEVload 会用到
                    // for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                    //     ckt->CKTrhs[i] = ckt->CKTrhsOld[i];
                    // }
                }
            }

            if (continuation_enabled && continuation_has_gmin) {
                ckt->CKTdiagGmin = continuation_gmin;
            }

            direct_op = if_traj && strcmp(if_traj, "1") == 0 && iterno >= traj_start_iter && iterno < 100;
            continuation_capture_pending = continuation_enabled &&
                continuation_dir_str &&
                continuation_active &&
                ((continuation_max_steps <= 0) || (continuation_step_counter < continuation_max_steps));
            continuation_step_index = continuation_capture_pending ? continuation_step_counter : -1;

            if (direct_op || continuation_capture_pending) {
                const char *ckt_id_str = getenv("CKT_ID");
                const char *traj_dir_str = getenv("TRAJ_DIR");
                const char *output_dir_str = continuation_capture_pending ? continuation_dir_str : traj_dir_str;
                if (!ckt_id_str) {
                    ckt_id_str = "default";
                }
                if (continuation_capture_pending) {
                    snprintf(
                        filename,
                        sizeof(filename),
                        "%s/continuation_circuit_%s_time_%.17e_gmin_%.17e_iter_%03d.txt",
                        output_dir_str,
                        ckt_id_str,
                        ckt->CKTtime,
                        ckt->CKTdiagGmin,
                        continuation_step_index
                    );
                } else {
                    snprintf(
                        filename,
                        sizeof(filename),
                        "%s/circuit_%s_time_%.17e_gmin_%.17e_iter_%03d.txt",
                        output_dir_str,
                        ckt_id_str,
                        ckt->CKTtime,
                        ckt->CKTdiagGmin,
                        iterno
                    );
                }
                fp = fopen(filename, "w");
                if (fp) {
                    fprintf(fp, "*************OLD*************\n");
                    for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                        fprintf(fp, "%.17e\n", ckt->CKTrhsOld[i]);
                    }
                    fprintf(fp, "*************STATE0_IN*************\n");
                    if (ckt->CKTstate0) {
                        for (i = 0; i < ckt->CKTnumStates; i++) {
                            fprintf(fp, "%.17e\n", ckt->CKTstate0[i]);
                        }
                    }
                    fclose(fp);
                    fp = NULL;
                }
            }

            error = CKTload(ckt);

            continuation_capture = continuation_enabled &&
                continuation_dir_str &&
                continuation_active &&
                ((continuation_max_steps <= 0) || (continuation_step_counter < continuation_max_steps));
            continuation_step_index = continuation_capture ? continuation_step_counter : -1;
            linear_system_corpus_active = 0;
            linear_system_filename[0] = '\0';
            linear_system_jac_filename[0] = '\0';

            if (linear_system_corpus_enabled) {
                if (online_sidecar_config.enabled && online_sidecar_config.valid)
                    online_snapshot_started = SPfrontEnd->IFseconds();
                int linear_system_export_allowed = 1;
                int matrix_size = SMPmatSize(ckt->CKTmatrix);
                const char *ckt_id_str = getenv("CKT_ID");

#ifdef KLU
                if (ckt->CKTkluMODE)
                    linear_system_export_allowed = 0;
#endif

                if (!linear_system_export_allowed) {
                    if (!linear_system_corpus_klu_warned) {
                        fprintf(
                            stderr,
                            "Warning: LINEAR_SYSTEM_CORPUS export currently supports Sparse only; skipping export while KLU is active.\n"
                        );
                        linear_system_corpus_klu_warned = 1;
                    }
                } else {
                    linear_system_corpus_active = 1;
                    snprintf(
                        linear_system_filename,
                        sizeof(linear_system_filename),
                        "%s/linear_system_circuit_%s_time_%.17e_gmin_%.17e_iter_%03d.txt",
                        linear_system_corpus_dir,
                        ckt_id_str ? ckt_id_str : "default",
                        ckt->CKTtime,
                        ckt->CKTdiagGmin,
                        iterno
                    );
                    linear_system_fp = fopen(linear_system_filename, "w");
                    if (linear_system_fp) {
                        linear_system_corpus_write_metadata(
                            linear_system_fp,
                            ckt,
                            ckt_id_str,
                            matrix_size,
                            iterno
                        );
                        linear_system_corpus_write_vector(
                            linear_system_fp,
                            "RHSOLD",
                            ckt->CKTrhsOld ? (ckt->CKTrhsOld + 1) : NULL,
                            matrix_size
                        );
                        linear_system_corpus_write_vector(
                            linear_system_fp,
                            "RHS",
                            ckt->CKTrhs ? (ckt->CKTrhs + 1) : NULL,
                            matrix_size
                        );
                        linear_system_corpus_write_vector(
                            linear_system_fp,
                            "STATE0",
                            ckt->CKTstate0,
                            ckt->CKTstate0 ? ckt->CKTnumStates : 0
                        );
                        fprintf(linear_system_fp, "*************NODE_MAP*************\n");
                        if (ckt->CKTnodes) {
                            CKTnode *node;
                            for (node = ckt->CKTnodes; node; node = node->next)
                                fprintf(linear_system_fp, "%s %d\n", node->name, node->number);
                        }
                        fclose(linear_system_fp);
                        linear_system_fp = NULL;
                    }

                    snprintf(
                        linear_system_jac_filename,
                        sizeof(linear_system_jac_filename),
                        "%s/linear_system_circuit_%s_time_%.17e_gmin_%.17e_iter_%03d_jac.txt",
                        linear_system_corpus_dir,
                        ckt_id_str ? ckt_id_str : "default",
                        ckt->CKTtime,
                        ckt->CKTdiagGmin,
                        iterno
                    );
                    SMPprint(ckt->CKTmatrix, linear_system_jac_filename);
                    if (online_sidecar_config.enabled && online_sidecar_config.valid &&
                        online_snapshot_started > 0.0) {
                        online_sidecar_result.snapshot_seconds =
                            SPfrontEnd->IFseconds() - online_snapshot_started;
                    }
                }
            }

            if (direct_op || continuation_capture) {
                // TRAJ 模式：输出到轨迹文件
                const char *ckt_id_str = getenv("CKT_ID");
                const char *traj_dir_str = getenv("TRAJ_DIR");
                const char *output_dir_str = continuation_capture ? continuation_dir_str : traj_dir_str;
                if (!ckt_id_str) {
                    ckt_id_str = "default";
                }
                if (continuation_capture) {
                    snprintf(
                        filename,
                        sizeof(filename),
                        "%s/continuation_circuit_%s_time_%.17e_gmin_%.17e_iter_%03d.txt",
                        output_dir_str,
                        ckt_id_str,
                        ckt->CKTtime,
                        ckt->CKTdiagGmin,
                        continuation_step_index
                    );
                } else {
                    snprintf(filename, sizeof(filename),
                            "%s/circuit_%s_time_%.17e_gmin_%.17e_iter_%03d.txt",
                            output_dir_str, ckt_id_str, ckt->CKTtime, ckt->CKTdiagGmin, iterno);
                }
                
                fp = fopen(filename, "a");
                if (fp) {
                    fprintf(fp,"*************NEW*************\n");
                    for (int i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                        fprintf(fp, "%.17e\n", ckt->CKTrhs[i]);
                    }
                    fprintf(fp,"*************STATE0_OUT*************\n");
                    if (ckt->CKTstate0) {
                        for (i = 0; i < ckt->CKTnumStates; i++) {
                            fprintf(fp, "%.17e\n", ckt->CKTstate0[i]);
                        }
                    }
                    fprintf(fp, "*************NODE_MAP*************\n");
                    CKTnode *node;
                    for (node = ckt->CKTnodes; node; node = node->next) {
                        fprintf(fp, "%s %d\n", node->name, node->number);
                    }
                    // printf("node map written\n");
                    printf("node info file saved\n");
                }
                
                if (continuation_capture) {
                    snprintf(
                        jac_filename,
                        sizeof(jac_filename),
                        "%s/continuation_circuit_%s_time_%.17e_gmin_%.17e_iter_%03d_jac.txt",
                        output_dir_str,
                        ckt_id_str,
                        ckt->CKTtime,
                        ckt->CKTdiagGmin,
                        continuation_step_index
                    );
                } else {
                    snprintf(jac_filename, sizeof(jac_filename),
                            "%s/circuit_%s_time_%.17e_gmin_%.17e_iter_%03d_jac.txt",
                            output_dir_str, ckt_id_str, ckt->CKTtime, ckt->CKTdiagGmin, iterno);
                }
                
                        SMPprint(ckt->CKTmatrix, jac_filename);
            }

            int should_compute_residual = 0;
            if (if_value && strcmp(if_value, "1") == 0 && iterno > 4 && iterno < 100)
                should_compute_residual = 1;
            else if (direct_op)
                should_compute_residual = 1;
            else if (continuation_capture)
                should_compute_residual = 1;
            else if (linear_system_corpus_active)
                should_compute_residual = 1;
            else if (segment_warmup_residual_path && ckt->CKTdiagGmin > 0.0)
                should_compute_residual = 1;
            
            if (should_compute_residual) {
                /* 确保矩阵被设置为实数模式，因为我们只使用实数部分 */
                /* 矩阵默认创建时是复数，但在调用 SMPreorder/SMPluFac 之前可能还是复数状态 */
                spSetReal(ckt->CKTmatrix->SPmatrix);
                
                double *J_times_xk = TMALLOC(double, SMPmatSize(ckt->CKTmatrix) + 1); 
                SMPmultiply(
                    ckt->CKTmatrix,      // A: 雅可比矩阵 J
                    J_times_xk,          // y: (输出) 存储结果 J*x^k 的向量
                    ckt->CKTrhsOld,      // x: (输入) 源向量 x^k (即 V(k-1))
                    NULL,                // iy: (输出) 结果的虚部，我们不需要，传 NULL
                    NULL                 // ix: (输入) 源向量的虚部，我们不需要，传 NULL
                );

                if (segment_warmup_residual_path && ckt->CKTdiagGmin > 0.0) {
                    const char *ckt_id_str = getenv("CKT_ID");
                    FILE *res_fp = fopen(segment_warmup_residual_path, "a");
                    double residual_l2_sq = 0.0;
                    for (int i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                        double residual = J_times_xk[i] - ckt->CKTrhs[i];
                        residual_l2_sq += residual * residual;
                    }
                    if (res_fp) {
                        fprintf(
                            res_fp,
                            "%s\t%.17e\t%.17e\t%d\t%.17e\n",
                            ckt_id_str ? ckt_id_str : "default",
                            ckt->CKTtime,
                            ckt->CKTdiagGmin,
                            iterno,
                            sqrt(residual_l2_sq)
                        );
                        fclose(res_fp);
                    }
                }
                
                // VALUE 模式：输出残差到 F_PATH 文件
                if (if_value && strcmp(if_value, "1") == 0 && iterno > 4 && iterno < 100) {
                    const char *f_path_str = getenv("F_PATH");
                    if (f_path_str) {
                        FILE *fp_out_f = fopen(f_path_str, "w");
                        if (fp_out_f) {
                            fprintf(fp_out_f, "************RES************\n");
                            for (int i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                                fprintf(fp_out_f, "%.17e\n", J_times_xk[i]-ckt->CKTrhs[i]);
                            }
                            fclose(fp_out_f);
                        }
                    }
                    
                    const char *j_path_str = getenv("J_PATH");
                    if (j_path_str) {
                        SMPprint(ckt->CKTmatrix, j_path_str);
                    }
                    
                    if (!(if_close_loop_train && strcmp(if_close_loop_train, "1")==0)) {
                        FREE(J_times_xk);
                        exit(0);
                    }
                } 
                else if (direct_op || continuation_capture) {
                    if (fp) {
                        fprintf(fp, "************RES************\n");
                        for (int i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                            fprintf(fp, "%.17e\n", J_times_xk[i]-ckt->CKTrhs[i]);
                        }
                        fclose(fp);
                    }
                    FREE(J_times_xk);
                    if (continuation_capture)
                        continuation_step_counter++;
                }
                else if (linear_system_corpus_active && linear_system_filename[0] != '\0') {
                    linear_system_fp = fopen(linear_system_filename, "a");
                    if (linear_system_fp) {
                        double residual_l2_sq = 0.0;
                        fprintf(linear_system_fp, "*************RAW_RESIDUAL*************\n");
                        for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                            double residual_value = J_times_xk[i] - ckt->CKTrhs[i];
                            residual_l2_sq += residual_value * residual_value;
                            fprintf(linear_system_fp, "%.17e\n", residual_value);
                        }
                        fprintf(linear_system_fp, "*************RAW_RESIDUAL_NORM*************\n");
                        fprintf(linear_system_fp, "%.17e\n", sqrt(residual_l2_sq));
                        fclose(linear_system_fp);
                        linear_system_fp = NULL;
                    }
                    FREE(J_times_xk);
                }
                else {
                    FREE(J_times_xk);
                }
            }

            if (online_sidecar_config.enabled) {
                if (!online_sidecar_config.valid) {
                    /* The configuration error is retained in the metrics record. */
                } else if (error) {
                    snprintf(
                        online_sidecar_result.failure_reason,
                        sizeof(online_sidecar_result.failure_reason),
                        "%s",
                        "online_snapshot_load_failed"
                    );
                } else if (!linear_system_corpus_active ||
                           linear_system_filename[0] == '\0' ||
                           linear_system_jac_filename[0] == '\0') {
                    snprintf(
                        online_sidecar_result.failure_reason,
                        sizeof(online_sidecar_result.failure_reason),
                        "%s",
                        "online_snapshot_missing"
                    );
                } else {
                    char online_output_path[NGSPICE_GMRES_PATH_MAX];
                    ngspice_gmres_resolve_sidecar_path(
                        &gmres_config,
                        ckt,
                        iterno + 1,
                        online_output_path
                    );
                    if (online_output_path[0] == '\0') {
                        snprintf(
                            online_sidecar_result.failure_reason,
                            sizeof(online_sidecar_result.failure_reason),
                            "%s",
                            "online_sidecar_path_missing"
                        );
                    } else {
                        const char *online_initial_guess_mode =
                            (gmres_config.use_rhsold_x0 && ckt->CKTrhsOld)
                                ? "rhsold"
                                : "zero";
                        ngspice_online_sidecar_generate(
                            &online_sidecar_config,
                            linear_system_filename,
                            linear_system_jac_filename,
                            online_output_path,
                            iterno + 1,
                            ckt->CKTtime,
                            ckt->CKTdiagGmin,
                            online_initial_guess_mode,
                            &online_sidecar_result
                        );
                    }
                }
            }

            /* printf("loaded, noncon is %d\n", ckt->CKTnoncon); */
            /* fflush(stdout); */
            if (trace_enabled) {
                fprintf(
                    stderr,
                    "PALS_NIITER_TRACE event=after_load time=%.17e mode=0x%lx iterno=%d noncon=%d direct_op=%d continuation_capture=%d\n",
                    ckt->CKTtime,
                    ckt->CKTmode,
                    iterno,
                    ckt->CKTnoncon,
                    direct_op,
                    continuation_capture
                );
            }
            iterno++;
            if (error) {
                ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                printf("load returned error \n");
#endif
                FREE(OldCKTstate0);
                return (error);
            }

            /* printf("after loading, before solving\n"); */
            /* CKTdump(ckt); */

            if (!(ckt->CKTniState & NIDIDPREORDER)) {
                error = SMPpreOrder(ckt->CKTmatrix);
                if (error) {
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("pre-order returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* badly formed matrix */
                }
                ckt->CKTniState |= NIDIDPREORDER;
            }

            if ((ckt->CKTmode & MODEINITJCT) ||
                ((ckt->CKTmode & MODEINITTRAN) && (iterno == 1)))
            {
                ckt->CKTniState |= NISHOULDREORDER;
            }

            /* moved here so both direct solve and native GMRES can reuse it */
            if (!OldCKTstate0)
                OldCKTstate0 = TMALLOC(double, ckt->CKTnumStates + 1);
            if (ckt->CKTstate0)
                memcpy(OldCKTstate0, ckt->CKTstate0,
                       (size_t) ckt->CKTnumStates * sizeof(double));

            if (!gmres_config.enabled
#ifdef KLU
                || ckt->CKTkluMODE
#endif
            ) {
            if (ckt->CKTniState & NISHOULDREORDER) {
                startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                if (ckt->CKTkluMODE) {
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                }
#endif

                error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                   ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                ckt->CKTstat->STATreorderTime +=
                    SPfrontEnd->IFseconds() - startTime;
                if (error) {
                    /* new feature - we can now find out something about what is
                     * wrong - so we ask for the troublesome entry
                     * Limit the number of messages to 6, if not 'set ngdebug'.
                     */
                    if (ft_ngdebug || msgcount < 6) {
                        SMPgetError(ckt->CKTmatrix, &i, &j);
                        if(eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                            SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                        else
                            SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                        msgcount += 1;
                    }
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("reorder returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error); /* can't handle these errors - pass up! */
                }
                ckt->CKTniState &= ~NISHOULDREORDER;
            } else {
                startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                if (ckt->CKTkluMODE) {
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                }
#endif

                error = SMPluFac(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                 ckt->CKTdiagGmin);
                ckt->CKTstat->STATdecompTime +=
                    SPfrontEnd->IFseconds() - startTime;

#ifdef KLU
                if ((ckt->CKTkluMODE) && (error == E_SINGULAR)) {

                    /* Francesco Lannutti - 25 Aug 2020
                     * If the matrix is numerically singular during ReFactorization, take the same matrix and factor it from scratch in the same iteration.
                     * This is my mod with KLU. It saves run-time, but also the system at the next iteration may be different.
                     * How do we guarantee that the system is the same at the next iteration? So, the original SPARSE version below sounds like a bug.
                     */
                    if (ft_ngdebug)
                        fprintf (stderr, "Warning: KLU ReFactor failed. Factoring again...\n") ;
                    ckt->CKTniState |= NISHOULDREORDER;
                    ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 0 ;
                    error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol, ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                    ckt->CKTstat->STATreorderTime += SPfrontEnd->IFseconds() - startTime;
                    if (error) {
                        SMPgetError(ckt->CKTmatrix, &i, &j);
                        if (ft_ngdebug || msgcount < 6) {
                            SMPgetError(ckt->CKTmatrix, &i, &j);
                            if (eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                                SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                            else
                                SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                            msgcount += 1;
                        }

                        /* CKTload(ckt); */
                        /* SMPprint(ckt->CKTmatrix, stdout); */
                        /* seems to be singular - pass the bad news up */
                        ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                        printf("lufac returned error \n");
#endif
                        FREE(OldCKTstate0);
                        return(error);
                    }
                } else if (error) {
                    if (!(ckt->CKTkluMODE) && (error == E_SINGULAR)) {

                        /* Francesco Lannutti - 25 Aug 2020
                         * If the matrix is numerically singular during ReFactorization, factor it from scratch at the next iteration.
                         * This is the original SPICE3F5 code and uses SPARSE.
                         */

                        ckt->CKTniState |= NISHOULDREORDER;
                        DEBUGMSG(" forced reordering....\n");
                        continue;
                    }
                    /* CKTload(ckt); */
                    /* SMPprint(ckt->CKTmatrix, stdout); */
                    /* seems to be singular - pass the bad news up */
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("lufac returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error);
                }
#else
                if (error) {
                    if (error == E_SINGULAR) {

                        /* Francesco Lannutti - 25 Aug 2020
                         * If the matrix is numerically singular during ReFactorization, factor it from scratch at the next iteration.
                         * This is the original SPICE3F5 code and uses SPARSE.
                         */

                        ckt->CKTniState |= NISHOULDREORDER;
                        DEBUGMSG(" forced reordering....\n");
                        continue;
                    }
                    /* CKTload(ckt); */
                    /* SMPprint(ckt->CKTmatrix, stdout); */
                    /* seems to be singular - pass the bad news up */
                    ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
                    printf("lufac returned error \n");
#endif
                    FREE(OldCKTstate0);
                    return(error);
                }
#endif

            }

                startTime = SPfrontEnd->IFseconds();
                SMPsolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare);
                ckt->CKTstat->STATsolveTime +=
                    SPfrontEnd->IFseconds() - startTime;
            } else {
                ngspice_gmres_result_t gmres_result;
                int gmres_success = 0;
                double gmres_solve_time = 0.0;
                startTime = SPfrontEnd->IFseconds();
                gmres_success = ngspice_gmres_solve(ckt, &gmres_config, iterno, &gmres_result);
                gmres_solve_time = SPfrontEnd->IFseconds() - startTime;
                ckt->CKTstat->STATsolveTime += gmres_solve_time;
                gmres_result.online_sidecar_enabled = online_sidecar_result.enabled;
                gmres_result.online_sidecar_success = online_sidecar_result.success;
                gmres_result.online_sidecar_exit_code = online_sidecar_result.exit_code;
                gmres_result.online_sidecar_timed_out = online_sidecar_result.timed_out;
                gmres_result.online_sidecar_snapshot_seconds = online_sidecar_result.snapshot_seconds;
                gmres_result.online_sidecar_generation_seconds = online_sidecar_result.generation_seconds;
                gmres_result.online_sidecar_bytes = online_sidecar_result.sidecar_bytes;
                strncpy(gmres_result.online_sidecar_failure_reason, online_sidecar_result.failure_reason, sizeof(gmres_result.online_sidecar_failure_reason) - 1);
                strncpy(gmres_result.online_sidecar_input_path, online_sidecar_result.input_path, sizeof(gmres_result.online_sidecar_input_path) - 1);
                strncpy(gmres_result.online_sidecar_jacobian_path, online_sidecar_result.jacobian_path, sizeof(gmres_result.online_sidecar_jacobian_path) - 1);
                strncpy(gmres_result.online_sidecar_output_path, online_sidecar_result.output_path, sizeof(gmres_result.online_sidecar_output_path) - 1);
                strncpy(gmres_result.online_sidecar_status_path, online_sidecar_result.status_path, sizeof(gmres_result.online_sidecar_status_path) - 1);
                gmres_metrics_append_jsonl(
                    ckt,
                    &gmres_config,
                    iterno,
                    SMPmatSize(ckt->CKTmatrix),
                    gmres_success,
                    gmres_solve_time,
                    &gmres_result);
                if (!gmres_success) {
                    if (ft_ngdebug) {
                        const char *detail_sidecar_path =
                            (gmres_result.fallback_reason[0] != '\0' &&
                             strncmp(gmres_result.fallback_reason, "sidecar_invalid_", 16) == 0)
                                ? gmres_result.resolved_sidecar_path
                                : "";
                        fprintf(
                            stderr,
                            "Warning: native GMRES fallback to direct solve (newton_iter=%d, time=%.17e, gmin=%.17e, precond=%s, reason=%s, iter=%d, restart_count=%d, solve_time=%.6e, sidecar_load_time=%.6e, sidecar_path=%s)\n",
                            iterno,
                            ckt->CKTtime,
                            ckt->CKTdiagGmin,
                            gmres_result.executed_precond,
                            gmres_result.fallback_reason[0] ? gmres_result.fallback_reason : "unspecified",
                            gmres_result.iterations,
                            gmres_result.restart_count,
                            gmres_solve_time,
                            gmres_result.sidecar_load_time,
                            detail_sidecar_path
                        );
                    }

                    if (ckt->CKTniState & NISHOULDREORDER) {
                        startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                        if (ckt->CKTkluMODE) {
                            ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                        }
#endif

                        error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                           ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                        ckt->CKTstat->STATreorderTime +=
                            SPfrontEnd->IFseconds() - startTime;
                        if (error) {
                            if (ft_ngdebug || msgcount < 6) {
                                SMPgetError(ckt->CKTmatrix, &i, &j);
                                if(eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                                    SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                                else
                                    SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                                msgcount += 1;
                            }
                            ckt->CKTstat->STATnumIter += iterno;
                            FREE(OldCKTstate0);
                            return(error);
                        }
                        ckt->CKTniState &= ~NISHOULDREORDER;
                    } else {
                        startTime = SPfrontEnd->IFseconds();

#ifdef KLU
                        if (ckt->CKTkluMODE) {
                            ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 1 ;
                        }
#endif

                        error = SMPluFac(ckt->CKTmatrix, ckt->CKTpivotAbsTol,
                                         ckt->CKTdiagGmin);
                        ckt->CKTstat->STATdecompTime +=
                            SPfrontEnd->IFseconds() - startTime;

#ifdef KLU
                        if ((ckt->CKTkluMODE) && (error == E_SINGULAR)) {
                            if (ft_ngdebug)
                                fprintf (stderr, "Warning: KLU ReFactor failed. Factoring again...\n") ;
                            ckt->CKTniState |= NISHOULDREORDER;
                            ckt->CKTmatrix->SMPkluMatrix->KLUloadDiagGmin = 0 ;
                            error = SMPreorder(ckt->CKTmatrix, ckt->CKTpivotAbsTol, ckt->CKTpivotRelTol, ckt->CKTdiagGmin);
                            ckt->CKTstat->STATreorderTime += SPfrontEnd->IFseconds() - startTime;
                            if (error) {
                                SMPgetError(ckt->CKTmatrix, &i, &j);
                                if (ft_ngdebug || msgcount < 6) {
                                    SMPgetError(ckt->CKTmatrix, &i, &j);
                                    if (eq(NODENAME(ckt, i), NODENAME(ckt, j)))
                                        SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check node %s\n", NODENAME(ckt, i));
                                    else
                                        SPfrontEnd->IFerrorf(ERR_WARNING, "singular matrix:  check nodes %s and %s\n", NODENAME(ckt, i), NODENAME(ckt, j));
                                    msgcount += 1;
                                }
                                ckt->CKTstat->STATnumIter += iterno;
                                FREE(OldCKTstate0);
                                return(error);
                            }
                        } else if (error) {
                            if (!(ckt->CKTkluMODE) && (error == E_SINGULAR)) {
                                ckt->CKTniState |= NISHOULDREORDER;
                                DEBUGMSG(" forced reordering....\n");
                                continue;
                            }
                            ckt->CKTstat->STATnumIter += iterno;
                            FREE(OldCKTstate0);
                            return(error);
                        }
#else
                        if (error) {
                            if (error == E_SINGULAR) {
                                ckt->CKTniState |= NISHOULDREORDER;
                                DEBUGMSG(" forced reordering....\n");
                                continue;
                            }
                            ckt->CKTstat->STATnumIter += iterno;
                            FREE(OldCKTstate0);
                            return(error);
                        }
#endif
                    }

                    startTime = SPfrontEnd->IFseconds();
                    SMPsolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare);
                    ckt->CKTstat->STATsolveTime +=
                        SPfrontEnd->IFseconds() - startTime;
                } else if (ft_ngdebug) {
                    fprintf(
                        stderr,
                        "Info: native GMRES converged (newton_iter=%d, time=%.17e, gmin=%.17e, precond=%s, reason=%s, iter=%d, restart_count=%d, solve_time=%.6e, sidecar_load_time=%.6e, raw_residual=%.6e)\n",
                        iterno,
                        ckt->CKTtime,
                        ckt->CKTdiagGmin,
                        gmres_result.executed_precond,
                        gmres_result.fallback_reason[0] ? gmres_result.fallback_reason : "none",
                        gmres_result.iterations,
                        gmres_result.restart_count,
                        gmres_solve_time,
                        gmres_result.sidecar_load_time,
                        gmres_result.final_raw_residual
                    );
                }
            }

#ifdef STEPDEBUG
            /*XXXX*/
            if (ckt->CKTrhs[0] != 0.0)
                printf("NIiter: CKTrhs[0] = %g\n", ckt->CKTrhs[0]);
            if (ckt->CKTrhsSpare[0] != 0.0)
                printf("NIiter: CKTrhsSpare[0] = %g\n", ckt->CKTrhsSpare[0]);
            if (ckt->CKTrhsOld[0] != 0.0)
                printf("NIiter: CKTrhsOld[0] = %g\n", ckt->CKTrhsOld[0]);
            /*XXXX*/
#endif
            ckt->CKTrhs[0] = 0;
            ckt->CKTrhsSpare[0] = 0;
            ckt->CKTrhsOld[0] = 0;

            if (iterno > maxIter) {
                ckt->CKTstat->STATnumIter += iterno;
                /* we don't use this info during transient analysis */
                if (ckt->CKTcurrentAnalysis != DOING_TRAN) {
                    FREE(errMsg);
                    errMsg = copy("Too many iterations without convergence");
#ifdef STEPDEBUG
                    fprintf(stderr, "too many iterations without convergence: %d iter's (max iter == %d)\n",
                    iterno, maxIter);
#endif
                }
                FREE(OldCKTstate0);
                return(E_ITERLIM);
            }

            if ((ckt->CKTnoncon == 0) && (iterno != 1))
                ckt->CKTnoncon = NIconvTest(ckt);
            else
                ckt->CKTnoncon = 1;

#ifdef STEPDEBUG
            printf("noncon is %d\n", ckt->CKTnoncon);
#endif
        }

        if ((ckt->CKTnodeDamping != 0) && (ckt->CKTnoncon != 0) &&
            ((ckt->CKTmode & MODETRANOP) || (ckt->CKTmode & MODEDCOP)) &&
            (iterno > 1))
        {
            CKTnode *node;
            double diff, maxdiff = 0;
            for (node = ckt->CKTnodes->next; node; node = node->next)
                if (node->type == SP_VOLTAGE) {
                    diff = fabs(ckt->CKTrhs[node->number] - ckt->CKTrhsOld[node->number]);
                    if (maxdiff < diff)
                        maxdiff = diff;
                }

            if (maxdiff > 10) {
                double damp_factor = 10 / maxdiff;
                if (damp_factor < 0.1)
                    damp_factor = 0.1;
                for (node = ckt->CKTnodes->next; node; node = node->next) {
                    diff = ckt->CKTrhs[node->number] - ckt->CKTrhsOld[node->number];
                    ckt->CKTrhs[node->number] =
                        ckt->CKTrhsOld[node->number] + (damp_factor * diff);
                }
                for (i = 0; i < ckt->CKTnumStates; i++) {
                    diff = ckt->CKTstate0[i] - OldCKTstate0[i];
                    ckt->CKTstate0[i] = OldCKTstate0[i] + (damp_factor * diff);
                }
            }
        }

            if (if_value && strcmp(if_value, "1") == 0 && iterno > 5 && iterno < 100) {
                if (if_close_loop_train && strcmp(if_close_loop_train, "1")==0) {
                    const char *close_loop_train_output_path_str = getenv("CLOSE_LOOP_PATH");
                    if (close_loop_train_output_path_str) {
                    FILE *close_fp_out = fopen(close_loop_train_output_path_str, "w");
                    if (close_fp_out) {
                        fprintf(close_fp_out, "************WP_OUT************\n");
                        for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                            fprintf(close_fp_out, "%.17e\n", ckt->CKTrhs[i]);
                        }
                        fclose(close_fp_out);
                    }
                }
                exit(0);
                }
            }

            if (continuation_capture && filename[0] != '\0') {
                FILE *continuation_fp_out = fopen(filename, "a");
                if (continuation_fp_out) {
                    fprintf(continuation_fp_out, "************WP_OUT************\n");
                    for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
                        fprintf(continuation_fp_out, "%.17e\n", ckt->CKTrhs[i]);
                    }
                    fclose(continuation_fp_out);
                }
                if ((continuation_max_steps > 0 && iterno >= continuation_stop_iter) || (ckt->CKTnoncon == 0)) {
                    exit(0);
                }
            }

        if (ckt->CKTmode & MODEINITFLOAT) {
            if ((ckt->CKTmode & MODEDC) && ckt->CKThadNodeset) {
                if (ipass)
                    ckt->CKTnoncon = ipass;
                ipass = 0;
            }
            if (ckt->CKTnoncon == 0) {
                if (trace_enabled) {
                    fprintf(
                        stderr,
                        "PALS_NIITER_TRACE event=converged_return time=%.17e mode=0x%lx iterno=%d\n",
                        ckt->CKTtime,
                        ckt->CKTmode,
                        iterno
                    );
                }
                ckt->CKTstat->STATnumIter += iterno;
                FREE(OldCKTstate0);
                return(OK);
            }
        } else if (ckt->CKTmode & MODEINITJCT) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFIX;
            ckt->CKTniState |= NISHOULDREORDER;
        } else if (ckt->CKTmode & MODEINITFIX) {
            if (ckt->CKTnoncon == 0)
                ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
            ipass = 1;
        } else if (ckt->CKTmode & MODEINITSMSIG) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITTRAN) {
            if (iterno <= 1)
                ckt->CKTniState |= NISHOULDREORDER;
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else if (ckt->CKTmode & MODEINITPRED) {
            ckt->CKTmode = (ckt->CKTmode & ~INITF) | MODEINITFLOAT;
        } else {
            ckt->CKTstat->STATnumIter += iterno;
#ifdef STEPDEBUG
            printf("bad initf state \n");
#endif
            FREE(OldCKTstate0);
            return(E_INTERN);
            /* impossible - no such INITF flag! */
        }




        /* build up the lvnim1 array from the lvn array */
        SWAP(double *, ckt->CKTrhs, ckt->CKTrhsOld);
        /* printf("after loading, after solving\n"); */
        /* CKTdump(ckt); */
    }
    /*NOTREACHED*/
}

void NIresetwarnmsg(void) {
    msgcount = 0;
}
