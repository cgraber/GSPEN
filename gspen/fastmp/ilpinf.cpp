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
#include "ilpinf.hpp"

using namespace std;

static void ILPInf_dealloc(ILPInf* self)
{
    PyArray_Free((PyObject*)self->potentials_obj, self->potentials);
    if (self->beliefs != NULL) 
    {
        PyArray_Free((PyObject*)self->beliefs_obj, self->beliefs);
        Py_DECREF(self->beliefs_obj);
    }
    Py_DECREF(self->potentials_obj);

    delete [] self->vars;

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int ILPInf_init(ILPInf *self, PyObject *args, PyObject *kwds)
{
    PyObject *pair_regions;
    int relax_program;
    if (!PyArg_ParseTuple(args, "iiOOp", &self->num_nodes, &self->num_vals,
                &pair_regions, &(self->potentials_obj), &relax_program))
    {
        return -1;
    }
    Py_INCREF(self->potentials_obj); //Since we hold onto this bad boy
    npy_intp* pot_obj_dims = PyArray_DIMS(self->potentials_obj);
    
    PyArray_Descr *descr = PyArray_DescrFromType(PyArray_TYPE(self->potentials_obj));

    if (PyArray_AsCArray(&(self->potentials_obj), (void *)& self->potentials, pot_obj_dims, 1, descr) < 0) {
        Py_DECREF(self->potentials_obj);
        PyErr_SetString(PyExc_TypeError, "error converting node potentials to c array");
        return -1;
    }
    try{
    self->env = new GRBEnv();
    self->model = new GRBModel(*self->env);
    self->model->set(GRB_IntParam_OutputFlag, 0);
    if (!PyList_Check(pair_regions)) {
        PyErr_SetString(PyExc_TypeError, "pair_regions not a list!");
        return -1;
    }
    self->num_pair_regions = PyList_Size(pair_regions);
    self->num_vars = self->num_nodes*self->num_vals + self->num_pair_regions*self->num_vals*self->num_vals;
    if (relax_program) {
        std::cout << "USING RELAXED VERSION" << std::endl;
        self->vars = self->model->addVars(self->num_vars, GRB_CONTINUOUS);
        for (int i = 0; i < self->num_vars; i++) {
            self->vars[i].set(GRB_DoubleAttr_UB, 1.0);
        }
    } else {
        self->vars = self->model->addVars(self->num_vars, GRB_BINARY);
    }

    // Node marginalization constraints
    for (int i = 0, offset=0; i < self->num_nodes; i++, offset += self->num_vals)
    {
        GRBLinExpr constr = 0;
        for (int val_ind = 0; val_ind < self->num_vals; val_ind++)
        {
            constr += self->vars[offset+val_ind];
        }
        self->model->addConstr(constr == 1);
    }
    int pair_start = self->num_nodes*self->num_vals;


    for (int i = 0, offset = pair_start; i < self->num_pair_regions; i++, offset+=self->num_vals*self->num_vals)
    {
        // Pair marginalization constraints
        GRBLinExpr constr = 0;
        for (int val_ind = 0; val_ind < self->num_vals*self->num_vals; val_ind++)
        {
            constr += self->vars[offset+val_ind];
        }
        self->model->addConstr(constr == 1);
        
        // Inter-region marginalization constraints
        PyObject *pair = PyList_GetItem(pair_regions, i);
        if (PyTuple_Check(pair)) {
            int first_node = PyLong_AsLong(PyTuple_GetItem(pair, (Py_ssize_t) 0));
            int second_node = PyLong_AsLong(PyTuple_GetItem(pair, (Py_ssize_t) 1));
            int offset1 = self->num_vals*first_node;
            int offset2 = self->num_vals*second_node;
            for (int n1_val_ind = 0; n1_val_ind < self->num_vals; n1_val_ind++)
            {
                GRBLinExpr constr = 0;
                for (int n2_val_ind = 0; n2_val_ind < self->num_vals; n2_val_ind++)
                {
                    int pair_val_ind = n2_val_ind*self->num_vals + n1_val_ind;
                    constr += self->vars[offset+pair_val_ind];
                }
                self->model->addConstr(constr == self->vars[offset1+n1_val_ind]);
            }
            for (int n2_val_ind = 0; n2_val_ind < self->num_vals; n2_val_ind++)
            {
                GRBLinExpr constr = 0;
                for (int n1_val_ind = 0; n1_val_ind < self->num_vals; n1_val_ind++)
                {
                    int pair_val_ind = n2_val_ind*self->num_vals + n1_val_ind;
                    constr += self->vars[offset+pair_val_ind];
                }
                self->model->addConstr(constr == self->vars[offset2+n2_val_ind]);
            }


        } else {
            PyErr_SetString(PyExc_TypeError, "pairs are not tuples!");
            return -1;
        }
    }
    } catch (GRBException e) {
        cout << "FAILED DURING INITIALIZATION. REASON: " << endl;
        cout << e.getMessage() << endl;
        throw e;
    }
    self->beliefs = NULL;
    self->belief_size = self->num_vars;
    return 0;    

}

static PyObject* ilpinf_runinf(ILPInf *self, PyObject *args)
{
    create_obj(self->model, self->vars, self->potentials, self->num_vars);
    self->model->optimize();
    copy_beliefs(self->vars, self->beliefs, self->num_vars);
    float obj = self->model->get(GRB_DoubleAttr_ObjVal);
    self->model->write("testmodel.mps");
    return Py_BuildValue("f", obj);

}

static PyObject* ilpinf_get_num_beliefs(ILPInf *self, PyObject *args)
{
    return Py_BuildValue("i", self->belief_size);
}

static PyObject* ilpinf_get_num_msgs(ILPInf *self, PyObject *args)
{
    return Py_BuildValue("i", 0);
}

static PyObject* ilpinf_update_potentials(ILPInf *self, PyObject *args)
{
    PyArray_Free(self->potentials_obj, self->potentials);
    Py_DECREF(self->potentials_obj);
    if (!PyArg_ParseTuple(args, "O", &(self->potentials_obj)))
    {
        return NULL;
    }
    Py_INCREF(self->potentials_obj); 

    npy_intp* pot_obj_dims = PyArray_DIMS(self->potentials_obj);
    PyArray_Descr *descr = PyArray_DescrFromType(PyArray_TYPE(self->potentials_obj));
    if (PyArray_AsCArray(&(self->potentials_obj), (void *)&self->potentials, pot_obj_dims, 1, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting node potentials to c array");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* ilpinf_update_beliefs_pointer(ILPInf *self, PyObject *args)
{
    if (self->beliefs != NULL) {
        PyArray_Free(self->beliefs_obj, self->beliefs);
        Py_DECREF(self->beliefs_obj);
    }
    if (!PyArg_ParseTuple(args, "O", &self->beliefs_obj)) {
        return NULL;
    }
    Py_INCREF(self->beliefs_obj);

    npy_intp* belief_obj_dims = PyArray_DIMS(self->beliefs_obj);
    PyArray_Descr *descr = PyArray_DescrFromType(PyArray_TYPE(self->beliefs_obj));
    if (PyArray_AsCArray(&self->beliefs_obj, (void *)&self->beliefs, belief_obj_dims, 1, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting beliefs to c array");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* ilpinf_get_belief(ILPInf* self, PyObject* args)
{
    int region_ind, val_ind;
    if (!PyArg_ParseTuple(args, "ii", &region_ind, &val_ind)) return NULL;
    int belief_ind;
    if (region_ind < self->num_nodes) {
        belief_ind = region_ind*self->num_vals + val_ind;
    } else {
        belief_ind = self->num_nodes*self->num_vals 
                     + (region_ind-self->num_nodes)*self->num_vals*self->num_vals + val_ind;
    }
    float result = self->beliefs[belief_ind];
    return Py_BuildValue("f", result);
}

static PyObject* ilpinf_get_beliefs(ILPInf* self, PyObject* args)
{
    return Py_BuildValue("O", self->beliefs_obj);
}



///Utility methods
void create_obj(GRBModel *model, GRBVar *vars, float *potentials, int num_vars)
{
    GRBLinExpr obj = 0.0;
    for (int i = 0; i < num_vars; i++)
    {
        obj += vars[i]*potentials[i];
    }
    model->setObjective(obj, GRB_MAXIMIZE);
}

void copy_beliefs(GRBVar *vars, float *beliefs, int num_vars)
{
    for (int i = 0; i < num_vars; i++)
    {
        beliefs[i] = vars[i].get(GRB_DoubleAttr_X);
    }
}

static PyMethodDef ILPInf_methods[] = {
    {"get_num_beliefs", (PyCFunction)ilpinf_get_num_beliefs, METH_VARARGS,
     "Get number of beliefs"},
    {"get_num_msgs", (PyCFunction)ilpinf_get_num_msgs, METH_VARARGS,
     "Get number of messages (here, 0 - hopefully will refactor this at some point to make this unnecessary"},
    {"update_potentials", (PyCFunction)ilpinf_update_potentials, METH_VARARGS,
     "Update pointer to potentials array"},
    {"update_beliefs_pointer", (PyCFunction)ilpinf_update_beliefs_pointer, METH_VARARGS,
     "Update pointer to beliefs array"},
    {"get_belief", (PyCFunction)ilpinf_get_belief, METH_VARARGS,
     "Get single belief value"},
    {"get_beliefs", (PyCFunction)ilpinf_get_beliefs, METH_VARARGS,
     "Get entire array of beliefs"},
    {"runinf", (PyCFunction)ilpinf_runinf, METH_VARARGS,
     "Run ILP Inference"},
    {NULL} /* Sentinel */
};

static PyTypeObject ILPInfType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "fastmp.ILPInf",            /* tp_name */
    sizeof(ILPInf),             /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)ILPInf_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,         /* tp_flags */
    "ILPInf objects",           /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    ILPInf_methods,             /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)ILPInf_init,      /* tp_init */
    0,                          /* tp_alloc */
    PyType_GenericNew,          /* tp_new */
};


/*
 **********************
 *Other module methods*
 **********************
 */

void run_ilp(GRBModel *model, GRBVar *vars, float *potentials, float *beliefs, int num_vars) {
    create_obj(model, vars, potentials, num_vars);
    model->optimize();
    copy_beliefs(vars, beliefs, num_vars);

}


static PyObject* ilpinfmod_runilp(ILPInf* self, PyObject* args)
{
    PyObject *ilpinfs;
    if (!PyArg_ParseTuple(args, "O", &ilpinfs))
    {
        return NULL;
    }
    if (!PyList_Check(ilpinfs)) {
        PyErr_SetString(PyExc_TypeError, "ilpinfs not a list!");
        return NULL;
    }
    Py_ssize_t num_problems = PyList_Size(ilpinfs);
    //std::vector<std::thread*> threads;
#pragma omp parallel for
    for (Py_ssize_t problem_ind = 0; problem_ind < num_problems; problem_ind++) {
        PyObject *problem = PyList_GetItem(ilpinfs, problem_ind);
        GRBModel *model = ((ILPInf *)problem)->model;
        GRBVar *vars = ((ILPInf *)problem)->vars;
        float *potentials = ((ILPInf *)problem)->potentials;
        float *beliefs = ((ILPInf *)problem)->beliefs;
        int num_vars = ((ILPInf *)problem)->num_vars;
        run_ilp(model, vars, potentials, beliefs, num_vars);
    }
    /*
    for (auto& th: threads) {
        th->join();
        delete th;
    }
    */
    Py_RETURN_NONE;
} 

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static PyMethodDef ILPInf_module_methods[] = {
    {"runilp", (PyCFunction)ilpinfmod_runilp, METH_VARARGS,
     "run multithreaded ilp inference"},
    {NULL, NULL, 0, NULL} //Sentinel
};


static int ilpinf_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int ilpinf_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ilpinf", /* name of module */
        NULL,     /* module documentation */
        sizeof(struct module_state), /* size of per-interpreter state of the module */
        ILPInf_module_methods,
        NULL,
        ilpinf_traverse,
        ilpinf_clear,
        NULL
};

/*
 *******************
 *Module init stuff*
 *******************
 */

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC PyInit_ilpinf(void)
{
    PyObject *m;
    ILPInfType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ILPInfType) < 0)
    {
        return NULL;
    }

    //m = Py_InitModule3("fastmp", FastMP_module_methods,
    //        "Module running fast message passing");
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;
    Py_INCREF(&ILPInfType);
    PyModule_AddObject(m, "ILPInf", (PyObject *)&ILPInfType);


    import_array(); /* So we can use Numpy Stuff */
    return m;
}

