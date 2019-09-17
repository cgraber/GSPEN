#include <Python.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <thread>
#include <chrono>
#include <omp.h>
#include "gurobi_c++.h"

using namespace std;

/*************************
 *ILPInf class definition*
 *************************
 */
typedef struct {
    PyObject_HEAD
    int num_vals;
    int num_nodes;
    int num_pair_regions;
    int belief_size;
    float* potentials;
    float* beliefs;
    PyObject *beliefs_obj;
    PyObject *potentials_obj;
    GRBEnv *env;
    GRBModel *model;
    GRBVar* vars;
    int num_vars;

} ILPInf;


static void ILPInf_dealloc(ILPInf* self);
static int ILPInf_init(ILPInf *self, PyObject *args, PyObject *kwds);
static PyObject* ilpinf_runinf(ILPInf *self, PyObject *args);
static PyObject* ilpinf_get_num_beliefs(ILPInf *self, PyObject *args);
static PyObject* ilpinf_update_potentials(ILPInf *self, PyObject *args);
static PyObject* ilpinf_update_beliefs_pointer(ILPInf *self, PyObject *args);
static PyObject* ilpinf_get_belief(ILPInf* self, PyObject* args);
static PyObject* ilpinf_get_beliefs(ILPInf* self, PyObject* args);

void create_obj(GRBModel *model, GRBVar *vars, float *potentials, int num_vars);
void copy_beliefs(GRBVar *vars, float *beliefs, int num_vars);


