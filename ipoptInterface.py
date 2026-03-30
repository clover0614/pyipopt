import cyipopt
from cyipopt import minimize_ipopt
import autograd.numpy as adnp
from autograd import jacobian, hessian
import numpy as np


class IPOPTInterface(cyipopt.Problem):
    def __init__(self, x0, obj, cons, cons_jac, cons_index, lb, ub, cl, cu, use_exact_hessian=False):
        """
        初始化IPOPT优化问题接口

        参数:
        x0: 初始点
        obj: 目标函数
        cons: 约束函数，输入是n维数组，输出也是数组
        lb: 变量下界
        ub: 变量上界
        cl: 约束下界
        cu: 约束上界
        use_exact_hessian: 是否使用精确Hessian矩阵计算，默认为False
        """
        # 获取问题维度
        n = len(x0)  # 变量个数
        m = len(cl)  # 约束个数

        # 保存问题维度
        self.n = n
        self.m = m

        self.cons_index = cons_index
        self.H = np.zeros((self.n, self.n))
        self.row, self.col = self.hessianstructure()
        # 调用父类初始化
        super().__init__(
            n=n,  # 变量个数
            m=m,  # 约束个数
            lb=lb,  # 变量下界
            ub=ub,  # 变量上界
            cl=cl,  # 约束下界
            cu=cu  # 约束上界
        )

        # 保存初始点
        self.x0 = x0

        # 保存目标函数
        self.obj = obj
        # 保存约束函数
        self.cons = cons

        # 保存边界条件
        self.lb = lb
        self.ub = ub
        self.cl = cl
        self.cu = cu

        # 是否使用精确Hessian
        self.use_exact_hessian = use_exact_hessian

        # 使用autograd计算目标函数的梯度
        self.grad = jacobian(self.obj)

        # 使用autograd计算约束的雅可比矩阵
        self.jac = cons_jac

        self.hess = hessian(self.cons)
        # # 只有在需要使用精确Hessian时才计算
        # if self.use_exact_hessian:
        #     # 使用autograd计算目标函数的Hessian矩阵
        #     self.obj_hess = hessian(self.obj)

        #     # 定义拉格朗日函数（用于计算整体的Hessian矩阵）
        #     def lagrangian(x, lagrange_mult):
        #         """
        #         拉格朗日函数: f(x) + λ^T * c(x)

        #         参数:
        #         x: 优化变量
        #         lagrange_mult: 拉格朗日乘子
        #         """
        #         return self.obj(x) + adnp.dot(lagrange_mult, self.cons(x))

        #     # 为了计算Hessian，我们需要一个给定λ的拉格朗日函数
        #     def fixed_lagrangian(x):
        #         # 这个函数会在hessian计算时被更新
        #         return lagrangian(x, self.current_lambda)

        #     self.lagrangian = lagrangian
        #     self.fixed_lagrangian = fixed_lagrangian
        #     # 计算Hessian矩阵
        #     self.hess_lagrangian = hessian(fixed_lagrangian)
        #     # 初始化拉格朗日乘子
        #     self.current_lambda = adnp.zeros(self.m)

    def objective(self, x):
        """返回目标函数值"""
        return self.obj(x)

    def gradient(self, x):
        """返回目标函数的梯度"""
        return self.grad(x)

    def constraints(self, x):
        """返回约束函数值"""
        return self.cons(x)

    def jacobian(self, x):
        """返回约束的雅可比矩阵"""
        return self.jac(x).flatten()

    def jacobianstructure(self):
        return self.cons_index

    def hessianstructure(self):
        """返回Hessian矩阵的非零元素的行列索引"""
        # 默认返回完全稠密的下三角矩阵结构
        # return np.nonzero(np.tril(np.ones((self.n, self.n))))
        return np.nonzero(np.tril(np.eye(self.n)))

    def hessian(self, x, lagrange, obj_factor):
        # """
        # 计算拉格朗日函数的Hessian矩阵

        # 参数:
        # x: 当前点
        # lagrange: 拉格朗日乘子
        # obj_factor: 目标函数的系数

        # 返回:
        # H: Hessian矩阵的非零元素值
        # """
        # if not self.use_exact_hessian:
        #     # 如果不使用精确Hessian，返回一个全零矩阵
        #     row, col = self.hessianstructure()
        #     return np.zeros_like(row, dtype=float)

        # # # 更新拉格朗日乘子（用于fixed_lagrangian函数）
        # # self.current_lambda = obj_factor * lagrange

        # # # 计算Hessian矩阵
        # # H = self.hess_lagrangian(x)

        # # # 获取下三角部分的索引
        # # row, col = self.hessianstructure()

        # # # 返回下三角部分的元素
        # # return H[row, col]
        # return self.hess_lagrangian(x)
        # hess_mat = self.hess(x)
        # print("shape: ", hess_mat.shape)

        # for i in range(len(hess_mat)):
        #     H += lagrange[i] * hess_mat[i]
        # row, col = self.hessianstructure()

        return self.H[self.row, self.col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """每次迭代时打印信息"""
        # print("迭代次数:", iter_count)
        # print("目标函数值:", obj_value)
        # print("约束可行性违反度:", inf_pr)
        # print("对偶可行性违反度:", inf_du)
        # print("互补性参数:", mu)
        # print("搜索方向范数:", d_norm)
        # print("对偶步长:", alpha_du)
        # print("原始步长:", alpha_pr)
        # print("线搜索尝试次数:", ls_trials)

        # # 每10次迭代输出一次详细的当前迭代点
        # if iter_count % 10 == 0:
        #     # 获取当前迭代点
        #     iterate = self.get_current_iterate()
        #     if iterate and "x" in iterate:
        #         print("当前迭代点:", iterate["x"])
        #         # 获取约束违反度
        #         violations = self.get_current_violations()
        #         if violations:
        #             # 检查violations字典中的键
        #             if "g" in violations:
        #                 print("约束违反度:", violations["g"])
        #             else:
        #                 print("约束违反度信息不可用。可用的键:", list(violations.keys()))

        # # 返回True表示继续优化，返回False表示中止优化
        # return True
        iterate = self.get_current_iterate()
        infeas = self.get_current_violations()
        primal = iterate["x"]
        # jac = self.jacobian(primal)

        # print("Iteration:", iter_count)
        # print("Primal iterate:", primal)
        # print("Flattened Jacobian:", jac[0:5])
        # print("Dual infeasibility:", infeas["grad_lag_x"])


def solve_ipopt(x0, obj=None, cons=None, cons_jac=None, cons_index=None, lb=None, ub=None, cl=None, cu=None,
                options=None, use_exact_hessian=False):
    """
    使用IPOPT求解优化问题

    参数:
    x0: 初始点
    obj: 目标函数
    cons: 约束函数
    lb: 变量下界
    ub: 变量上界
    cl: 约束下界
    cu: 约束上界
    options: IPOPT求解器选项
    use_exact_hessian: 是否使用精确Hessian矩阵计算，默认为False

    返回:
    x: 最优解
    info: 求解器信息
    """
    # 创建优化问题实例
    if obj is None:
        obj = lambda x: 0.0  # 返回浮点数0.0而不是整数0
    if cons is None:
        cons = lambda x: np.zeros(len(x0))
    if lb is None:
        lb = adnp.array([-adnp.inf] * len(x0))
    if ub is None:
        ub = adnp.array([adnp.inf] * len(x0))
    if cl is None:
        cl = adnp.zeros(len(cons(x0)))
    if cu is None:
        cu = adnp.zeros(len(cons(x0)))

    nlp = IPOPTInterface(x0, obj, cons, cons_jac, cons_index, lb, ub, cl, cu, use_exact_hessian)

    # 设置默认选项
    if options is None:
        options = {
            'linear_solver': 'ma57',
            'ma57_automatic_scaling': 'yes',
            'print_level': 5,  # 打印级别
            'max_iter': 500,  # 最大迭代次数
            'tol': 1e-8,  # 收敛容差
            # 'mu_strategy': 'monotone',  # 互补性参数更新策略
            'ma57_pivtol': 1e-7,
            'required_infeasibility_reduction': 0.95,
            'limited_memory_aug_solver': 'extended',
            'slack_bound_push': 1e-6,
            'slack_bound_frac': 1e-6,
            #  'alpha_for_y': 'max',
            #  'alpha_for_y_tol': 9.73429364095845,
            #  'barrier_tol_factor': 8.15003717524080,
            'bound_frac': 1e-6,
            #  'bound_mult_init_method': 'mu-based',
            #  'bound_mult_init_val': 1.05963349619431,
            'bound_push': 1e-6,
            #  'fixed_mu_oracle': 'loqo',
            #  'max_soc': 3,
            #  'mu_linear_decrease_factor': 0.33760441672535,
            #  'mu_max': 20068.01700984188938,
            'mu_init': 1e-10,
            #  'mu_max_fact': 621.58816173313403,
            #  'mu_min': 0.00000000001168,
            #  'mu_oracle': 'probing',
            #  'mu_superlinear_decrease_power': 1.50304818024808,
            #  'nlp_scaling_max_gradient': 191.93957528720486,
            #  'obj_scaling_factor': 0.72984418113895,
            #  'quality_function_max_section_steps': 18,
            # 'mu_strategy': 'adaptive',  # 互补性参数更新策略
            # 'hessian_approximation': 'limited-memory',

        }

        # # 根据是否使用精确Hessian设置对应选项
        # if not use_exact_hessian:
        #     options['hessian_approximation'] = 'limited-memory'  # 使用BFGS近似Hessian

    # 设置求解器选项
    for option, value in options.items():
        nlp.add_option(option, value)

    # 求解问题
    x, info = nlp.solve(x0)

    return x, info
