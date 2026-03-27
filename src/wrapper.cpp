#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <coin-or/IpTNLP.hpp>
#include <coin-or/IpIpoptApplication.hpp>
#include <iostream>

namespace py = pybind11;
using namespace Ipopt;

// 继承IPOPT的TNLP类
class PyIpoptNLP : public TNLP {
private:
    int n_, m_; // 变量个数 (n) 和约束个数 (m)
    py::array_t<double> x0_, x_L_, x_U_, g_L_, g_U_; 
    // 存放雅可比矩阵和海森矩阵稀疏结构 (COO格式) 的 NumPy 数组
    py::array_t<int> jac_iRow_, jac_jCol_;  
    py::array_t<int> hess_iRow_, hess_jCol_;
    // 存放来自 Python 的 5 个核心计算函数
    py::function eval_f_cb_, eval_grad_f_cb_, eval_g_cb_, eval_jac_g_cb_, eval_h_cb_;

    SmartPtr<IpoptApplication> app_; // IPOPT 求解器实例的智能指针

public:
    py::array_t<double> x_opt_, mult_g_, mult_x_L_, mult_x_U_, g_val_;
    double obj_val_;
    int status_;

    // 与类名相同的构造函数，无返回值，相当于 Python 中的 __init__
    PyIpoptNLP(int n, int m, 
               py::array_t<double> x0, py::array_t<double> x_L, py::array_t<double> x_U,
               py::array_t<double> g_L, py::array_t<double> g_U,
               py::array_t<int> jac_iRow, py::array_t<int> jac_jCol, 
               py::array_t<int> hess_iRow, py::array_t<int> hess_jCol,
               py::function eval_f, py::function eval_grad_f,
               py::function eval_g, py::function eval_jac_g, py::function eval_h)
        : n_(n), m_(m), x0_(x0), x_L_(x_L), x_U_(x_U), g_L_(g_L), g_U_(g_U),
          jac_iRow_(jac_iRow), jac_jCol_(jac_jCol),
          hess_iRow_(hess_iRow), hess_jCol_(hess_jCol),
          eval_f_cb_(eval_f), eval_grad_f_cb_(eval_grad_f),
          eval_g_cb_(eval_g), eval_jac_g_cb_(eval_jac_g), eval_h_cb_(eval_h) {
          
        app_ = IpoptApplicationFactory(); // 创建求解器实例

        if (hess_iRow_.size() == 0) {
            // 如果用户传入的海森矩阵结构为空（长度为 0），说明用户不想算二阶导。
            // 封装层自动替用户开启 L-BFGS，防止 IPOPT 崩溃
            app_->Options()->SetStringValue("hessian_approximation", "limited-memory");
        }
    }

    virtual ~PyIpoptNLP() {}

    // 参数配置接口，将 python 传入的动态类型解析为 c++ 的强类型
    // 使用 pybind11 的 isinstance 判断类型
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

    // 信息初始化，继承自IPOPT，接口中重写
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style) {
        n = n_;
        m = m_;
        nnz_jac_g = jac_iRow_.size(); // 雅可比矩阵非零元素数量
        nnz_h_lag = hess_iRow_.size(); // 海森矩阵非零元素数量
        index_style = TNLP::C_STYLE; // 告诉 IPOPT 用 0 作为数组起始索引
        return true;
    }

    // 边界与初值传递，继承自IPOPT，接口中重写
    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u) {
        // unchecked<1>()无视安全边界检查 x_L_.at(i)，将 Python 类型转化为 C++
        // x_L_ - py::array_t<double>   x_l - Number*，c++指针
        auto buf_xL = x_L_.unchecked<1>(); auto buf_xU = x_U_.unchecked<1>();
        for (Index i = 0; i < n; i++) { x_l[i] = buf_xL(i); x_u[i] = buf_xU(i); }
        auto buf_gL = g_L_.unchecked<1>(); auto buf_gU = g_U_.unchecked<1>();
        for (Index i = 0; i < m; i++) { g_l[i] = buf_gL(i); g_u[i] = buf_gU(i); }
        return true;
    }

    // 提供初值，继承自IPOPT，接口中重写
    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda, Number* lambda) {
        if (init_x) { // 初值 x 热启动，其余参数如乘子初值冷启动
            auto buf_x0 = x0_.unchecked<1>();
            for (Index i = 0; i < n; i++) x[i] = buf_x0(i); 
        }
        return true;
    }

    // 计算目标函数值
    virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
        py::array_t<double> py_x(n, x); // 零拷贝，py_x指针指向传来的x
        obj_value = eval_f_cb_(py_x).cast<double>(); // 调用 Python 端函数，再返回给 IPOPT
        return true;
    }

    // 计算目标函数的梯度
    virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
        // 1. 把 C++ 的裸指针 x 包装成不需要拷贝内存的 NumPy 数组
        py::array_t<double> py_x(n, x);
        
        // 2. 调用 Python 端的回调函数，并将返回值强转为 NumPy 数组
        auto res = eval_grad_f_cb_(py_x).cast<py::array_t<double>>();
        
        // 3. 把 Python 返回的数组里的值，填回给 IPOPT 的裸指针 grad_f
        auto buf = res.unchecked<1>();
        for (Index i = 0; i < n; i++) grad_f[i] = buf(i);
        return true;
    }

    // 计算约束函数的值
    virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
        if (m == 0) return true; // 无约束函数
        py::array_t<double> py_x(n, x); // 零拷贝封装 x
        auto res = eval_g_cb_(py_x).cast<py::array_t<double>>(); // Python 端计算
        auto buf = res.unchecked<1>();
        for (Index i = 0; i < m; i++) g[i] = buf(i);
        return true;
    }

    // 计算约束的稀疏雅可比矩阵
    virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow, Index *jCol, Number* values) {
        if (m == 0) return true;
        // 第 1 阶段：IPOPT 传了 NULL 进来，只需要坐标，即 COO 结构
        // 求解第一步会走这里，IPOPT 在底层分配内存
        if (values == NULL) {
            auto r = jac_iRow_.unchecked<1>();
            auto c = jac_jCol_.unchecked<1>();
            for (Index i = 0; i < nele_jac; i++) {
                iRow[i] = r(i); jCol[i] = c(i);
            }
        } else {
            // 第 2 阶段：IPOPT 准备好了内存 values，它要求填入具体的导数值
            py::array_t<double> py_x(n, x);
            auto res = eval_jac_g_cb_(py_x).cast<py::array_t<double>>();
            auto buf = res.unchecked<1>();
            for (Index i = 0; i < nele_jac; i++) values[i] = buf(i);
        }
        return true;
    }

    // 计算拉格朗日函数的二阶偏导数（海森矩阵）
    virtual bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, 
                        Index m, const Number* lambda, bool new_lambda, 
                        Index nele_hess, Index* iRow, Index* jCol, Number* values) {
        if (values == NULL) {
            // 第 1 次调用：提供海森矩阵的行和列结构 (通常是下三角矩阵，因为海森矩阵是对称阵)
            auto r = hess_iRow_.unchecked<1>();
            auto c = hess_jCol_.unchecked<1>();
            for (Index i = 0; i < nele_hess; i++) {
                iRow[i] = r(i); jCol[i] = c(i);
            }
        } else {
            // 第 2 次及后续调用：计算精确的二阶导数并提供非零元素的值
            py::array_t<double> py_x(n, x);
            py::array_t<double> py_lambda(m, lambda); // 拉格朗日乘子
            
            // 调用 Python 函数时，传入 x, lambda(lagrange) 和 obj_factor(缩放因子)
            auto res = eval_h_cb_(py_x, py_lambda, obj_factor).cast<py::array_t<double>>();
            auto buf = res.unchecked<1>();
            for (Index i = 0; i < nele_hess; i++) values[i] = buf(i);
        }
        return true; 
    }

    // 求解后的包装
    virtual void finalize_solution(SolverReturn status,
                                   Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda,
                                   Number obj_value, const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq) {
        // 标量数据的简单接收
        status_ = static_cast<int>(status); // IPOPT 返回的状态码
        obj_val_ = obj_value; // 最终的目标函数极小值

        // 深拷贝，因为 IPOPT 求解结束后会释放内存
        x_opt_ = py::array_t<double>(n);
        mult_x_L_ = py::array_t<double>(n);
        mult_x_U_ = py::array_t<double>(n);

        // 写入数据，需要 mutable_
        auto buf_x = x_opt_.mutable_unchecked<1>();
        auto buf_zL = mult_x_L_.mutable_unchecked<1>();
        auto buf_zU = mult_x_U_.mutable_unchecked<1>();
        for (Index i = 0; i < n; i++) {
            buf_x(i) = x[i]; buf_zL(i) = z_L[i]; buf_zU(i) = z_U[i];
        }

        // 约束相关数据
        g_val_ = py::array_t<double>(m);
        mult_g_ = py::array_t<double>(m);
        auto buf_g = g_val_.mutable_unchecked<1>();
        auto buf_lam = mult_g_.mutable_unchecked<1>();
        for (Index i = 0; i < m; i++) {
            buf_g(i) = g[i]; buf_lam(i) = lambda[i];
        }
    }

    // 求解函数
    py::tuple solve() {
        app_->Initialize(); // 调用 IPOPT 主程序的初始化方法，add_option会在这里检查
        SmartPtr<TNLP> nlp = this; // this 代表当前 PyIpoptNLP 类的实例
        app_->OptimizeTNLP(nlp); // 启动优化主循环，IPOPT 开始求解。结束前调用 finalize_solution 函数

        // 将结果打包成 Python 字典
        py::dict info;
        info["status"] = status_;
        info["obj_val"] = obj_val_;
        info["g"] = g_val_;
        info["mult_g"] = mult_g_;
        info["mult_x_L"] = mult_x_L_;
        info["mult_x_U"] = mult_x_U_;

        // 打包成元组返回
        return py::make_tuple(x_opt_, info);
    }
};

// 宏定义
PYBIND11_MODULE(pyipopt, m) { // 创建一个名为 pyipopt 的 Python 模块（Module），并在接下来的代码块里，用变量 m 代表这个模块
    // 模块的文档字符串，help(pyipopt) 的返回值
    m.doc() = "Industrial grade IPOPT wrapper with Exact Hessian support";

    py::class_<PyIpoptNLP>(m, "Problem") // Python 中命名为 Problem
        // 将 C++ 类的构造函数暴露为 Python 的 __init__ 方法
        .def(py::init<int, int, py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>,
                      py::array_t<int>, py::array_t<int>,   // 雅可比矩阵稀疏结构
                      py::array_t<int>, py::array_t<int>,   // 海森矩阵稀疏结构
                      py::function, py::function, py::function, py::function, py::function>())
        
        // 绑定成员方法，把 C++ 类的成员函数，暴露给 Python 的实例对象。 后者为c++中取成员函数指针
        .def("add_option", &PyIpoptNLP::add_option)
        .def("solve", &PyIpoptNLP::solve);
}