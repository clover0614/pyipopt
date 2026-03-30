# Python 包装层，最终的 API

import numpy as np
import os
import sys
import os
import sys
import numpy as np

# 1. 获取当前包的绝对路径
package_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 【核心修改】针对 Windows 的强力注入
if sys.platform == 'win32':
    # 满足 Python 3.8+ 的新机制 (用于加载 _core.pyd 及其直接依赖)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(package_dir)
    
    # 满足底层 C++ (LoadLibraryA) 的旧机制 (用于 IPOPT 加载 libhsl.dll)
    # 我们把包路径强行插到 PATH 的最前面，确保最高优先级
    os.environ['PATH'] = package_dir + os.pathsep + os.environ.get('PATH', '')

# 3. 验证一下（调试用，正式版可删）
print(f"DEBUG: pyipopt package_dir is {package_dir}")
print(f"DEBUG: Current PATH starts with {os.environ['PATH'][:100]}")


# --- 寻址完毕，安全导入底层 C++ 引擎 ---
# 从 C++ 底层模块中导入真正的 Problem 类
from ._core import Problem as _CoreProblem

class IPOPTInterface:
    """
    面向用户的 IPOPT 包装类。
    负责接收 Python 数据、存储配置选项，并在 solve 时实例化底层 C++ 引擎。
    """
    def __init__(self, x0, obj, obj_grad, cons, cons_jac, cons_index, lb, ub, cl, cu, use_exact_hessian):
        self.x0_default = x0
        self.obj = obj
        self.obj_grad = obj_grad
        self.cons = cons
        self.cons_jac = cons_jac
        self.cons_index = cons_index
        self.lb = lb
        self.ub = ub
        self.cl = cl
        self.cu = cu
        self.use_exact_hessian = use_exact_hessian
        
        # 建立一个字典，暂时存下用户 add_option 的所有配置
        self.options = {}

    def add_option(self, keyword, value):
        self.options[keyword] = value

    def solve(self, x0=None):
        """
        真正的执行点。支持在这里传入新的 x0。
        """
        # 🚨 满足你的需求 2：允许在 solve 时传入 x0
        if x0 is None:
            x0 = self.x0_default
            
        n = len(x0)
        
        # --- 维度获取 ---
        dummy_cons = self.cons(x0)
        m = len(dummy_cons) if dummy_cons is not None else 0

        # --- 极端严格的数据清洗 (强转 float64 和处理无穷大) ---
        IPOPT_INF = 2.0e19
        x0_np = np.array(x0, dtype=np.float64)
        
        lb_np = np.full(n, -IPOPT_INF, dtype=np.float64) if self.lb is None else np.array(self.lb, dtype=np.float64)
        lb_np[lb_np <= -1e19] = -IPOPT_INF
            
        ub_np = np.full(n, IPOPT_INF, dtype=np.float64) if self.ub is None else np.array(self.ub, dtype=np.float64)
        ub_np[ub_np >= 1e19] = IPOPT_INF

        cl_np = np.zeros(m, dtype=np.float64) if self.cl is None else np.array(self.cl, dtype=np.float64)
        cu_np = np.zeros(m, dtype=np.float64) if self.cu is None else np.array(self.cu, dtype=np.float64)

        # --- 稀疏雅可比坐标 ---
        if self.cons_index is not None:
            jac_iRow = np.array(self.cons_index[0], dtype=np.int32)
            jac_jCol = np.array(self.cons_index[1], dtype=np.int32)
        else:
            jac_iRow = np.array([], dtype=np.int32)
            jac_jCol = np.array([], dtype=np.int32)

        # --- 海森矩阵占位 ---
        if self.use_exact_hessian:
            raise NotImplementedError("Exact Hessian interface is strictly required but not provided in inputs yet.")
        else:
            hess_iRow = np.array([], dtype=np.int32)
            hess_jCol = np.array([], dtype=np.int32)
            eval_h_dummy = lambda x, lag, obj_f: np.array([], dtype=np.float64)

        # 实例化底层的 C++ V8 引擎, 调用cpp文件中包装的模块
        core_nlp = _CoreProblem(
            n, m, 
            x0_np, lb_np, ub_np, cl_np, cu_np, 
            jac_iRow, jac_jCol, hess_iRow, hess_jCol, 
            self.obj, self.obj_grad, self.cons, self.cons_jac, eval_h_dummy
        )

        # 将 Python 层暂存的所有 options 倾倒给 C++ 引擎
        for opt_name, opt_val in self.options.items():
            core_nlp.add_option(opt_name, opt_val)

        # 真正开始求解
        return core_nlp.solve()


def solve_ipopt(x0, obj=None, obj_grad=None, cons=None, cons_jac=None, cons_index=None, 
                lb=None, ub=None, cl=None, cu=None, options=None, use_exact_hessian=False):
    """
    使用 IPOPT 求解优化问题 (开箱即用的顶层函数)
    """
    # 目标函数和梯度智能默认化 (如果没传，默认求解 F(x)=0)
    if obj is None:
        obj = lambda x: 0.0
    if obj_grad is None:
        obj_grad = lambda x: np.zeros(len(x0), dtype=np.float64)
    if cons is None:
        cons = lambda x: np.array([], dtype=np.float64)
    if cons_jac is None:
        cons_jac = lambda x: np.array([], dtype=np.float64)

    # 实例化我们的 Python 包装类
    nlp = IPOPTInterface(x0, obj, obj_grad, cons, cons_jac, cons_index, lb, ub, cl, cu, use_exact_hessian)

    # 智能合并选项
    default_options = {
        'print_level': 5,
        'max_iter': 500,
        'tol': 1e-8,
        'linear_solver': 'ma57', # 默认的线性求解器
    }
    
    if options is not None:
        default_options.update(options)

    for option, value in default_options.items():
        nlp.add_option(option, value)

    # 以 nlp.solve(x0) 的形式调用
    x_opt, info = nlp.solve(x0)

    return x_opt, info