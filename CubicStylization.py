import igl
import numpy as np
from scipy.sparse import csc_matrix


# 进行 cubic stylization 操作
class CubeStylization:
    def __init__(self, mesh_file, cube_lambda=10):
        self.V, self.F = igl.read_triangle_mesh(mesh_file)
        self.NV = self.V.shape[0]

        # ADMM parameters
        self.cube_lambda = cube_lambda
        self.rho = 1e-4
        self.epi_abs = 1e-5
        self.epi_rel = 1e-3
        self.mu = 10
        self.tao = 2
        self.ADMM_IterNum = 20

        # 取出网格的 arap 所需数据
        self.L = igl.cotmatrix(self.V, self.F)  # 取出 cot 值矩阵，为一个 (NV,NV) 的稀疏阵
        self.N = igl.per_vertex_normals(self.V, self.F)  # (NV,3) 大小的矩阵
        self.area_weight = igl.massmatrix(self.V, self.F).diagonal()  # 取出质量矩阵的对角阵，大小为 (NV, )

        # 固定 arap 一个点
        self.fixed_pt_handle = np.array([0])
        self.fixed_pt_handle_pos = self.V[self.fixed_pt_handle]

        # 计算 arap 右端项，主要是获得能量计算的右端项，乘以一个旋转矩阵之后就是右端项
        self.arap_rhs = igl.arap_rhs(self.V, self.F, 3, igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)

        # 构建一个边的表，其记录方式是某个下标 idx_i-idx_{i+1} 的范围内都是 v_i 对应邻面内的所有边，对应着 v_i 的 spokes and rims
        self.NI = self.edge_list = self.edge_vec = self.w2edge = None
        self.halfedge_build()
        # ADMM 所需的迭代量，记录每一次迭代的值
        self.new_v = self.zAll = self.uAll = self.rhoAll = self.RAll = None
        self.ADMM_Iter_Param()

    # 构建有向边的列表
    def halfedge_build(self):
        # VF 为一个包含相邻三角形面的顶点索引的一维数组。它将每个顶点的相邻三角形面连接在一起
        # NI 为一个表示 VF 中每个顶点相邻三角形面数量的累积和数组。这个数组用于指示每个顶点在 VF 中的开始和结束位置。
        VF, NI = igl.vertex_triangle_adjacency(self.F, self.NV)
        self.NI = NI * 3

        # 构建了一个边的列表，形如：
        # [ [f0.v0  f0.v1  f0.v2], [f1.v0  f1.v1  f1.v2], ...
        #   [f0.v1  f0.v2  f0.v0], [f1.v1  f1.v2  f1.v0], ... ]
        # 其中 f0,f1,..,fn1 均为某个 v 的邻接面，因此从 NI[i]-NI[i+1] 均为点 v 的 spokes and rims
        self.edge_list = np.empty((VF.shape[0] * 3, 2), dtype=np.int32)
        self.edge_list[::3, 0] = self.F[VF, 0]
        self.edge_list[1::3, 0] = self.F[VF, 1]
        self.edge_list[2::3, 0] = self.F[VF, 2]
        self.edge_list[::3, 1] = self.F[VF, 1]
        self.edge_list[1::3, 1] = self.F[VF, 2]
        self.edge_list[2::3, 1] = self.F[VF, 0]
        self.edge_vec = self.V[self.edge_list[:, 1]] - self.V[self.edge_list[:, 0]]
        self.w2edge = self.L[self.edge_list[:, 0], self.edge_list[:, 1]]

    def ADMM_Iter_Param(self):
        self.new_v = self.V.copy()
        self.RAll = np.zeros((self.NV, 3, 3))
        self.zAll = np.zeros((self.NV, 3, 1))
        self.uAll = np.zeros((self.NV, 3, 1))
        self.rhoAll = np.full((self.NV,), self.rho)

    @staticmethod
    def admm_R_iter(M):
        U, X, V_T = np.linalg.svd(M)
        R = (U @ V_T).T
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = (U @ V_T).T
        return R

    @staticmethod
    def shrinkage(x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)

    # admm iteration 中的一步操作
    def Update(self):
        updated_edge_vec = self.new_v[self.edge_list[:, 1]] - self.new_v[self.edge_list[:, 0]]
        for ii in range(self.NV):
            z = self.zAll[ii]
            u = self.uAll[ii]
            rho = self.rhoAll[ii]
            n = self.N[ii].reshape(3, 1)

            # 构造 spoke and rims edges 对角阵
            W = self.w2edge[0, self.NI[ii]:self.NI[ii + 1]]
            W.resize(W.shape[1])
            W = np.diag(W)

            # 计算出迭代过程中的不变量 M_tmp，后续再加上每次都变的部分成为 M
            D = self.edge_vec[self.NI[ii]:self.NI[ii + 1], :]
            D_BAR = updated_edge_vec[self.NI[ii]:self.NI[ii + 1], :]
            M_tmp = D.T @ W @ D_BAR

            for _ in range(self.ADMM_IterNum):
                M = M_tmp + rho * (n @ (z - u).T)
                R = self.admm_R_iter(M)
                z_old = z
                z = self.shrinkage(R @ n + u, self.cube_lambda * self.area_weight[ii] / rho)
                u += R @ n - z

                # r_norm 表示 primal residual, s_norm 表示 dual residual
                r_norm = np.linalg.norm(z - R @ n)
                s_norm = np.linalg.norm(-rho * (z - z_old))
                if r_norm > self.mu * s_norm:
                    rho *= self.tao
                    u /= self.tao
                elif s_norm > self.mu * r_norm:
                    rho /= self.tao
                    u *= self.tao

                # eps_pri = sqrt(p) * epi_abs + epi_rel * max{norm(R*n),norm(z)}
                # eps_dual = sqrt(n) * epi_abs + epi_rel * y, y = rho * u
                # p,n 为约束 I*Rn-z=0 的 I 行列数
                eps_pri = (np.sqrt(3) * self.epi_abs +
                           self.epi_rel * np.maximum(np.linalg.norm(R @ n), np.linalg.norm(z)))
                eps_dual = np.sqrt(3) * self.epi_abs + self.epi_rel * np.linalg.norm(rho * u)

                if r_norm < eps_pri and s_norm < eps_dual:
                    self.zAll[ii] = z
                    self.uAll[ii] = u
                    self.rhoAll[ii] = rho
                    self.RAll[ii] = R
                    break

        # arap_rhs 乘以 R 就是能量的 BTx 中的 B，下面相当于要解 minimize 0.5*xT*L*x + BTx , subject to Aeq*x=Beq
        Rcol = self.RAll.reshape(self.NV * 3 * 3, 1, order='F')
        Bcol = self.arap_rhs @ Rcol
        B = Bcol.reshape(int(Bcol.shape[0] / 3), 3, order='F')
        Aeq = csc_matrix((0, 0))
        Beq = np.array([])
        _, self.new_v = igl.min_quad_with_fixed(self.L, B, self.fixed_pt_handle, self.fixed_pt_handle_pos,
                                                Aeq, Beq, False)

    def Iteration(self, algo_iter=10):
        for ii in range(algo_iter):
            print(f"Interation number: {ii}")
            self.Update()
