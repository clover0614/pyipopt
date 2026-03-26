#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <coin-or/IpTNLP.hpp>
#include <coin-or/IpIpoptApplication.hpp>
#include <iostream>

namespace py = pybind11;
using namespace Ipopt;

class PyIpoptNLP : public TNLP {
private:
    int n_, m_;
    py::array_t<double> x0_, x_L_, x_U_, g_L_, g_U_;
    py::array_t<int> jac_iRow_, jac_jCol_; 
    py::array_t<int> hess_iRow_, hess_jCol_; // 🚨 新增：海森矩阵的稀疏行列索引
    py::function eval_f_cb_, eval_grad_f_cb_, eval_g_cb_, eval_jac_g_cb_, eval_h_cb_;

    SmartPtr<IpoptApplication> app_;

public:
    py::array_t<double> x_opt_, mult_g_, mult_x_L_, mult_x_U_, g_val_;
    double obj_val_;
    int status_;

    PyIpoptNLP(int n, int m, 
               py::array_t<double> x0, py::array_t<double> x_L, py::array_t<double> x_U,
               py::array_t<double> g_L, py::array_t<double> g_U,
               py::array_t<int> jac_iRow, py::array_t<int> jac_jCol, 
               py::array_t<int> hess_iRow, py::array_t<int> hess_jCol, // 🚨 新增接收参数
               py::function eval_f, py::function eval_grad_f,
               py::function eval_g, py::function eval_jac_g, py::function eval_h)
        : n_(n), m_(m), x0_(x0), x_L_(x_L), x_U_(x_U), g_L_(g_L), g_U_(g_U),
          jac_iRow_(jac_iRow), jac_jCol_(jac_jCol),
          hess_iRow_(hess_iRow), hess_jCol_(hess_jCol), // 🚨 初始化
          eval_f_cb_(eval_f), eval_grad_f_cb_(eval_grad_f),
          eval_g_cb_(eval_g), eval_jac_g_cb_(eval_jac_g), eval_h_cb_(eval_h) {
          
        app_ = IpoptApplicationFactory();
        // 现在默认使用精确海森矩阵
        // 自动判断是否开启 L-BFGS
        if (hess_iRow_.size() == 0) {
            // 如果用户传入的海森矩阵结构为空（长度为 0），说明用户不想算二阶导。
            // 封装层自动替用户开启 L-BFGS，防止 IPOPT 崩溃
            app_->Options()->SetStringValue("hessian_approximation", "limited-memory");
        }
    }

    virtual ~PyIpoptNLP() {}

    void add_option(const std::string& keyword, py::handle value) {
        if (py::isinstance<py::str>(value)) {
            app_->Options()->SetStringValue(keyword, value.cast<std::string>());
        } else if (py::isinstance<py::float_>(value)) {
            app_->Options()->SetNumericValue(keyword, value.cast<double>());
        } else if (py::isinstance<py::int_>(value)) {
            app_->Options()->SetIntegerValue(keyword, value.cast<int>());
        } else {
            throw std::runtime_error("Unsupported option type. Must be str, float, or int.");
        }
    }

    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style) {
        n = n_;
        m = m_;
        nnz_jac_g = jac_iRow_.size();
        nnz_h_lag = hess_iRow_.size(); // 🚨 海森矩阵非零元素个数由传入的结构决定
        index_style = TNLP::C_STYLE;
        return true;
    }

    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u) {
        auto buf_xL = x_L_.unchecked<1>(); auto buf_xU = x_U_.unchecked<1>();
        for (Index i = 0; i < n; i++) { x_l[i] = buf_xL(i); x_u[i] = buf_xU(i); }
        auto buf_gL = g_L_.unchecked<1>(); auto buf_gU = g_U_.unchecked<1>();
        for (Index i = 0; i < m; i++) { g_l[i] = buf_gL(i); g_u[i] = buf_gU(i); }
        return true;
    }

    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda, Number* lambda) {
        if (init_x) { 
            auto buf_x0 = x0_.unchecked<1>();
            for (Index i = 0; i < n; i++) x[i] = buf_x0(i); 
        }
        return true;
    }

    virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
        py::array_t<double> py_x(n, x);
        obj_value = eval_f_cb_(py_x).cast<double>();
        return true;
    }

    virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
        py::array_t<double> py_x(n, x);
        auto res = eval_grad_f_cb_(py_x).cast<py::array_t<double>>();
        auto buf = res.unchecked<1>();
        for (Index i = 0; i < n; i++) grad_f[i] = buf(i);
        return true;
    }

    virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
        if (m == 0) return true;
        py::array_t<double> py_x(n, x);
        auto res = eval_g_cb_(py_x).cast<py::array_t<double>>();
        auto buf = res.unchecked<1>();
        for (Index i = 0; i < m; i++) g[i] = buf(i);
        return true;
    }

    virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow, Index *jCol, Number* values) {
        if (m == 0) return true;
        if (values == NULL) {
            auto r = jac_iRow_.unchecked<1>();
            auto c = jac_jCol_.unchecked<1>();
            for (Index i = 0; i < nele_jac; i++) {
                iRow[i] = r(i); jCol[i] = c(i);
            }
        } else {
            py::array_t<double> py_x(n, x);
            auto res = eval_jac_g_cb_(py_x).cast<py::array_t<double>>();
            auto buf = res.unchecked<1>();
            for (Index i = 0; i < nele_jac; i++) values[i] = buf(i);
        }
        return true;
    }

    // 🚨 核心稀疏海森矩阵组装
    virtual bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, 
                        Index m, const Number* lambda, bool new_lambda, 
                        Index nele_hess, Index* iRow, Index* jCol, Number* values) {
        if (values == NULL) {
            // 第 1 次调用：提供海森矩阵的行和列结构 (通常是下三角矩阵)
            auto r = hess_iRow_.unchecked<1>();
            auto c = hess_jCol_.unchecked<1>();
            for (Index i = 0; i < nele_hess; i++) {
                iRow[i] = r(i); jCol[i] = c(i);
            }
        } else {
            // 第 2 次及后续调用：计算精确的二阶导数并提供非零元素的值
            py::array_t<double> py_x(n, x);
            py::array_t<double> py_lambda(m, lambda); // 🚨 把乘子打包传给 Python
            
            // 🚨 调用 Python 函数时，传入 x, lambda(lagrange) 和 obj_factor
            auto res = eval_h_cb_(py_x, py_lambda, obj_factor).cast<py::array_t<double>>();
            auto buf = res.unchecked<1>();
            for (Index i = 0; i < nele_hess; i++) values[i] = buf(i);
        }
        return true; 
    }

    virtual void finalize_solution(SolverReturn status,
                                   Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda,
                                   Number obj_value, const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq) {
        status_ = static_cast<int>(status);
        obj_val_ = obj_value;

        x_opt_ = py::array_t<double>(n);
        mult_x_L_ = py::array_t<double>(n);
        mult_x_U_ = py::array_t<double>(n);
        auto buf_x = x_opt_.mutable_unchecked<1>();
        auto buf_zL = mult_x_L_.mutable_unchecked<1>();
        auto buf_zU = mult_x_U_.mutable_unchecked<1>();
        for (Index i = 0; i < n; i++) {
            buf_x(i) = x[i]; buf_zL(i) = z_L[i]; buf_zU(i) = z_U[i];
        }

        g_val_ = py::array_t<double>(m);
        mult_g_ = py::array_t<double>(m);
        auto buf_g = g_val_.mutable_unchecked<1>();
        auto buf_lam = mult_g_.mutable_unchecked<1>();
        for (Index i = 0; i < m; i++) {
            buf_g(i) = g[i]; buf_lam(i) = lambda[i];
        }
    }

    py::tuple solve() {
        app_->Initialize();
        SmartPtr<TNLP> nlp = this;
        app_->OptimizeTNLP(nlp);

        py::dict info;
        info["status"] = status_;
        info["obj_val"] = obj_val_;
        info["g"] = g_val_;
        info["mult_g"] = mult_g_;
        info["mult_x_L"] = mult_x_L_;
        info["mult_x_U"] = mult_x_U_;

        return py::make_tuple(x_opt_, info);
    }
};

PYBIND11_MODULE(pyipopt, m) {
    m.doc() = "Industrial grade IPOPT wrapper with Exact Hessian support";

    py::class_<PyIpoptNLP>(m, "Problem")
        // 🚨 初始化参数从 14 个变成了 16 个！
        .def(py::init<int, int, py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>,
                      py::array_t<int>, py::array_t<int>,   // 雅可比矩阵稀疏结构
                      py::array_t<int>, py::array_t<int>,   // 🚨 海森矩阵稀疏结构
                      py::function, py::function, py::function, py::function, py::function>())
        .def("add_option", &PyIpoptNLP::add_option)
        .def("solve", &PyIpoptNLP::solve);
}