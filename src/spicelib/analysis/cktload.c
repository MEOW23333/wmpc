/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

/* CKTload(ckt)
 * this is a driver program to iterate through all the various
 * load functions provided for the circuit elements in the
 * given circuit
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/enh.h"
/* gtri - add - wbk - 11/26/90 - add include for MIF global data */
#include "ngspice/mif.h"
/* gtri - end - wbk - 11/26/90 */
#endif

static int ZeroNoncurRow(SMPmatrix *matrix, CKTnode *nodes, int rownum);

// /* 全局标志：标记是否在第1次 CKTload 调用时成功从文件读入了工作点 */
// int CKTload_wp_read_flag = 0;

// #define PER_DEVICE_STATS
int
CKTload(CKTcircuit *ckt)
{
    int i;
    int size;
    double startTime, td0;
    CKTnode *node;
    int error;
    // const char *if_value = getenv("VALUE");
    // const char *if_close_loop_train = getenv("CLOSE_LOOP_TRAIN");
#ifdef STEPDEBUG
    int noncon;
#endif /* STEPDEBUG */

#ifdef XSPICE
    /* gtri - begin - Put resistors to ground at all nodes */
    /*   SMPmatrix  *matrix; maschmann : deleted , because unused */

    double gshunt;
    int num_nodes;

    /* gtri - begin - Put resistors to ground at all nodes */
#endif

    td0 = SPfrontEnd->IFseconds();

#ifdef PER_DEVICE_STATS
    ckt->CKTstat->devTimes[DEVmaxnum] += SPfrontEnd->IFseconds() - td0;
    ckt->CKTstat->devCounts[DEVmaxnum]++;
#endif

    startTime = SPfrontEnd->IFseconds();
    size = SMPmatSize(ckt->CKTmatrix);
    for (i = 0; i <= size; i++) {
        ckt->CKTrhs[i] = 0;
    }
    SMPclear(ckt->CKTmatrix);
#ifdef STEPDEBUG
    noncon = ckt->CKTnoncon;
#endif /* STEPDEBUG */

    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVload && ckt->CKThead[i]) {
#ifdef PER_DEVICE_STATS
            td0 = SPfrontEnd->IFseconds();
#endif
            error = DEVices[i]->DEVload (ckt->CKThead[i], ckt);
#ifdef PER_DEVICE_STATS
            ckt->CKTstat->devTimes[i] += SPfrontEnd->IFseconds() - td0;
            ckt->CKTstat->devCounts[i]++;
#endif
            if (ckt->CKTnoncon)
                ckt->CKTtroubleNode = 0;
#ifdef STEPDEBUG
            if (noncon != ckt->CKTnoncon) {
                printf("device type %s nonconvergence\n",
                       DEVices[i]->DEVpublic.name);
                noncon = ckt->CKTnoncon;
            }
#endif /* STEPDEBUG */
            if (error) return(error);
        }
    }

            // // vvvvvvvvvvvvvv 【【【新的钩子逻辑】】】 vvvvvvvvvvvvvvv
            // if (if_value && strcmp(if_value, "1") == 0 && CKTload_wp_read_flag) {

            //     // ---  打印所有结果 ---
            //     const char *f_path_str = getenv("F_PATH");
            //     FILE *fp_out_f = fopen(f_path_str, "w");
            //     fprintf(fp_out_f, "************RES************\n");
            //     for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
            //         fprintf(fp_out_f, "%.15e\n", ckt->CKTrhs[i]);
            //     }
            //     fclose(fp_out_f);

            //     const char *j_path_str = getenv("J_PATH");
            //     SMPprint(ckt->CKTmatrix, j_path_str);
            //     if (!(if_close_loop_train && strcmp(if_close_loop_train, "1")==0))
            //     {exit(0);}
            // }

#ifdef XSPICE
    /* gtri - add - wbk - 11/26/90 - reset the MIF init flags */

    /* init is set by CKTinit and should be true only for first load call */
    g_mif_info.circuit.init = MIF_FALSE;

    /* anal_init is set by CKTdoJob and is true for first call */
    /* of a particular analysis type */
    g_mif_info.circuit.anal_init = MIF_FALSE;

    /* gtri - end - wbk - 11/26/90 */

    /* gtri - begin - Put resistors to ground at all nodes. */
    /* Value of resistor is set by new "rshunt" option.     */

    if (ckt->enh->rshunt_data.enabled) {
        gshunt = ckt->enh->rshunt_data.gshunt;
        num_nodes = ckt->enh->rshunt_data.num_nodes;
        for (i = 0; i < num_nodes; i++) {
            *(ckt->enh->rshunt_data.diag[i]) += gshunt;
        }
    }

    /* gtri - end - Put resistors to ground at all nodes */
#endif

    // if (if_value && strcmp(if_value, "1") == 0) {
    //     /* 从文件读入工作点 V(k-1) */
    //     const char *wp_in_str = getenv("WP_IN_PATH");
    //     if (wp_in_str) {
    //         FILE *fp_in = fopen(wp_in_str, "r");
    //         if (fp_in) {
    //             for (i = 1; i <= SMPmatSize(ckt->CKTmatrix); i++) {
    //                 if (fscanf(fp_in, "%le", &ckt->CKTrhsOld[i]) != 1) {
    //                     ckt->CKTrhsOld[i] = 0.0;
    //                 }
    //             }
    //             fclose(fp_in);
                
    //             /* 将工作点应用到节点，类似 nodeset 的处理方式 */
    //             for (node = ckt->CKTnodes; node; node = node->next) {
    //                 if (node->number == 0) continue;
    //                 double *diag_ptr = node->ptr;
    //                 if (diag_ptr == NULL) {
    //                     // 如果缓存指针为空，显式向 SMP 矩阵请求对角线元素 (row, row)
    //                     // 最后的参数 0 表示 "如果不存在不创建"，但在 CKTload 阶段对角线肯定存在
    //                     diag_ptr = (double *)SMPfindElt(ckt->CKTmatrix, node->number, node->number, 1);
    //                 }

    //                 if (ZeroNoncurRow(ckt->CKTmatrix, ckt->CKTnodes,
    //                                   node->number)) {
    //                     ckt->CKTrhs[node->number] = 1.0e10 * 
    //                                                  ckt->CKTrhsOld[node->number] *
    //                                                  ckt->CKTsrcFact;
                        
    //                     *(diag_ptr) = 1e10;
                        
    //                 } else {
    //                     ckt->CKTrhs[node->number] = ckt->CKTrhsOld[node->number] *
    //                                                  ckt->CKTsrcFact;
                    
    //                     *(diag_ptr) = 1;
                        
    //                 }
    //             }
    //             /* 标记：成功从文件读入工作点 */
    //             CKTload_wp_read_flag = 1;
    //         }
    //     }
    // }

    if (ckt->CKTmode & MODEDC) {
        /* consider doing nodeset & ic assignments */
        if (ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) {
            /* do nodesets */
            for (node = ckt->CKTnodes; node; node = node->next) {
                if (node->nsGiven) {
                    if (ZeroNoncurRow(ckt->CKTmatrix, ckt->CKTnodes,
                                      node->number)) {
                        ckt->CKTrhs[node->number] = 1.0e10 * node->nodeset *
                                                      ckt->CKTsrcFact;
                        *(node->ptr) = 1e10;
                    } else {
                        ckt->CKTrhs[node->number] = node->nodeset *
                                                      ckt->CKTsrcFact;
                        *(node->ptr) = 1;
                    }
                    /* DAG: Original CIDER fix. If above fix doesn't work,
                     * revert to this.
                     */
                    /*
                     *  ckt->CKTrhs[node->number] += 1.0e10 * node->nodeset;
                     *  *(node->ptr) += 1.0e10;
                     */
                }
            }
        }
        if ((ckt->CKTmode & MODETRANOP) && (!(ckt->CKTmode & MODEUIC))) {
            for (node = ckt->CKTnodes; node; node = node->next) {
                if (node->icGiven) {
                    if (ZeroNoncurRow(ckt->CKTmatrix, ckt->CKTnodes,
                                      node->number)) {
                        /* Original code:
                         ckt->CKTrhs[node->number] += 1.0e10 * node->ic;
                        */
                        ckt->CKTrhs[node->number] = 1.0e10 * node->ic *
                                                      ckt->CKTsrcFact;
                        *(node->ptr) += 1.0e10;
                    } else {
                        /* Original code:
                          ckt->CKTrhs[node->number] = node->ic;
                        */
                        ckt->CKTrhs[node->number] = node->ic*ckt->CKTsrcFact; /* AlansFixes */
                        *(node->ptr) = 1;
                    }
                    /* DAG: Original CIDER fix. If above fix doesn't work,
                     * revert to this.
                     */
                    /*
                     *  ckt->CKTrhs[node->number] += 1.0e10 * node->ic;
                     *  *(node->ptr) += 1.0e10;
                     */
                }
            }
        }
    }
    /* SMPprint(ckt->CKTmatrix, stdout); if you want to debug, this is a
    good place to start ... */

    ckt->CKTstat->STATloadTime += SPfrontEnd->IFseconds()-startTime;
    return(OK);
}

static int
ZeroNoncurRow(SMPmatrix *matrix, CKTnode *nodes, int rownum)
{
    CKTnode     *n;
    double      *x;
    int         currents;

    currents = 0;
    for (n = nodes; n; n = n->next) {
        x = (double *) SMPfindElt(matrix, rownum, n->number, 0);
        if (x) {
            if (n->type == SP_CURRENT)
                currents = 1;
            else
                *x = 0.0;
        }
    }

    return currents;
}
