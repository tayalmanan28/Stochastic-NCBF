import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


############################################
# learn rate scheduling
############################################
def calc_safe_u(x_domain, h_domain, d_h_domain, d2_h_domain):
    kp = 10
    u_ref = kp*(x_domain)
    u = 0*x_domain

    return u


# def norm(x, y):
#     return sqrt(x**2 + y**2)

# class QP_Controller_Unicycle(QP_Controller):
#     def __init__(self, gamma:float, obs_radius=0.1):
#         """
#         Constructor creates QP_Controller object for Multi Agent system where 
#         each agent is a Unicycle model bot.        

#         Parameters
#         ----------
#         gamma : float
#             class k function is taken as simple function f(x) = \gamma x. The 
#             valueo f gamma is assumed to 1 for all the agents for time being.
#         no_of_agents : int
#             DESCRIPTION.

#         Returns
#         -------
#         None.

#         """
#         self.gamma = gamma
#         self.u_ref = None
#         self.u_star = None
#         self.G = 9.81
#         self.obs_r = obs_radius
        
#     def set_reference_control(self, u_ref:np.ndarray):
#         """
#         sets the reference controller

#         Parameters
#         ----------
#         u_ref : numpy.ndarray
#             numpy array of size (n, 2) containing the reference controls.

#         Returns
#         -------
#         None.

#         """
#         self.u_ref = u_ref
#         self.u_star = np.copy(u_ref)
    
#     def get_optimal_control(self):
#         self.u_star = self.u_star / self.kf
#         if self.u_star[0][0] <0:
#             self.u_star[0][0] =0 
#             print(0)
#         if self.u_star[1][0] <0:
#             self.u_star[1][0] =0
#             print(1)
#         if self.u_star[2][0] <0:
#             self.u_star[2][0] =0
#             print(2)
#         if self.u_star[3][0] <0:
#             self.u_star[3][0] =0
#             print(3)

#         # print(self.u_star[1][0])
#         propellers_1_rpm = sqrt(self.u_star[1][0])
#         propellers_3_rpm = sqrt(self.u_star[3][0])
#         propellers_0_rpm = sqrt(self.u_star[0][0])
#         propellers_2_rpm = sqrt(self.u_star[2][0])

#         return np.array([propellers_0_rpm, propellers_1_rpm,
#                          propellers_2_rpm, propellers_3_rpm])
    
#     def get_reference_control(self):
#         return self.u_ref
    
#     def setup_QP(self, bot, c, c_d):
#         """
#         the function takes bot list and creates symbolic varaibles associated 
#         with each bot required for computation in QP. The functions also 
#         precomputes the required symbolic expressions for QP such as L_f, L_g,
#         etc...
#         Parameters
#         ----------
#         bot_list : list
#             list of Unicycle bot objects 
            
#         Returns
#         -------
#         None.

#         """
#         c_x, c_y, c_z = c
#         c_x_d, c_y_d, c_z_d = c_d
#         # Create placholders for symbolic expressions
#         self.f = [0] # f Matrix in control system
#         self.g = [0] # g Matrix in control system
#         self.h = [[0]] # C3BF Matrix
        
#         # Create placholder for terms in QP
#         self.Psi = []
#         self.B = [[]]
#         self.C = [[]]
        
#         # create state and parameter symbolic varaibles for each bot
        
#         symbols_string = 'x y z x_d y_d z_d phi theta psi w_1 w_2 w_3 L Ixx Iyy Izz m l r'
#         bot.sym_x, bot.sym_y, bot.sym_z, bot.sym_x_d, bot.sym_y_d, bot.sym_z_d, bot.sym_phi, bot.sym_theta, bot.sym_psi, bot.sym_w_1, bot.sym_w_2, bot.sym_w_3, bot.sym_L, bot.sym_Ixx, bot.sym_Iyy, bot.sym_Izz, bot.sym_m, bot.sym_l, bot.sym_r =  symbols(symbols_string)
#         self.f = Matrix([bot.sym_x_d, 
#                         bot.sym_y_d,
#                         bot.sym_z_d,
#                         0,
#                         0, 
#                         - self.G,
#                         bot.sym_w_1 + bot.sym_w_2*sin(bot.sym_phi)*tan(bot.sym_theta) + bot.sym_w_3*cos(bot.sym_phi)*tan(bot.sym_theta),
#                         bot.sym_w_2*cos(bot.sym_phi) - bot.sym_w_3*sin(bot.sym_phi),
#                         (bot.sym_w_2*sin(bot.sym_phi) + bot.sym_w_3*cos(bot.sym_phi))/cos(bot.sym_theta),
#                         (bot.sym_Iyy - bot.sym_Izz)*bot.sym_w_2*bot.sym_w_3/bot.sym_Ixx,
#                         (bot.sym_Izz - bot.sym_Ixx)*bot.sym_w_1*bot.sym_w_3/bot.sym_Iyy,
#                         (bot.sym_Ixx - bot.sym_Iyy)*bot.sym_w_1*bot.sym_w_2/bot.sym_Izz])

#         p = (cos(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) + sin(bot.sym_psi)*sin(bot.sym_phi))/bot.sym_m
#         q = (sin(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) - cos(bot.sym_psi)*sin(bot.sym_phi))/bot.sym_m
#         r = (cos(bot.sym_theta)*cos(bot.sym_phi))/bot.sym_m
#         self.g = Matrix([[0, 0, 0, 0],
#                         [0, 0, 0, 0],
#                         [0, 0, 0, 0],
#                         [p, p, p, p],
#                         [q, q, q, q],
#                         [r, r, r, r],
#                         [0, 0, 0, 0],
#                         [0, 0, 0, 0],
#                         [0, 0, 0, 0],
#                         [0, bot.sym_L/bot.sym_Iyy, 0, -bot.sym_L/bot.sym_Iyy],
#                         [bot.sym_L/bot.sym_Ixx, 0, -bot.sym_L/bot.sym_Ixx, 0],
#                         [0, 0, 0, 0]])
            
#         # for CBF h
#         # Bot objects 
#         r_1_x = (cos(bot.sym_psi)*cos(bot.sym_theta))
#         r_1_y = (sin(bot.sym_psi)*cos(bot.sym_theta))
#         r_1_z = (-sin(bot.sym_theta))

#         r_2_x = (cos(bot.sym_psi)*sin(bot.sym_theta)*sin(bot.sym_phi) - sin(bot.sym_psi)*cos(bot.sym_phi))
#         r_2_y = (sin(bot.sym_psi)*sin(bot.sym_theta)*sin(bot.sym_phi) - cos(bot.sym_psi)*cos(bot.sym_phi))
#         r_2_z = (cos(bot.sym_theta)*sin(bot.sym_phi))

#         self.x_i = r_1_x
#         self.y_i = r_1_y
#         self.z_i = r_1_z

#         self.x_j = r_2_x
#         self.y_j = r_2_y
#         self.z_j = r_2_z

#         self.x_d_i = -bot.sym_w_3*r_1_y + bot.sym_w_2*r_1_z
#         self.y_d_i = -bot.sym_w_1*r_1_z + bot.sym_w_3*r_1_x
#         self.z_d_i = -bot.sym_w_2*r_1_x + bot.sym_w_1*r_1_y

#         self.x_d_j = -bot.sym_w_3*r_2_y + bot.sym_w_2*r_2_z
#         self.y_d_j = -bot.sym_w_1*r_2_z + bot.sym_w_3*r_2_x
#         self.z_d_j = -bot.sym_w_2*r_2_x + bot.sym_w_1*r_2_y

#         r_x = (cos(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) + sin(bot.sym_psi)*sin(bot.sym_phi))
#         r_y = (sin(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) - cos(bot.sym_psi)*sin(bot.sym_phi))
#         r_z = (cos(bot.sym_theta)*cos(bot.sym_phi))

#         # Relative position terms
#         p_rel_x = c_x - (bot.sym_x + bot.sym_l*r_x)
#         p_rel_y = c_y - (bot.sym_y + bot.sym_l*r_y)
#         p_rel_z = c_z - (bot.sym_z + bot.sym_l*r_z)
        
#         # Relative velocity terms
#         v_rel_x = c_x_d - (bot.sym_x_d + bot.sym_l*(-bot.sym_w_3*r_y + bot.sym_w_2*r_z))
#         v_rel_y = c_y_d - (bot.sym_y_d + bot.sym_l*(-bot.sym_w_1*r_z + bot.sym_w_3*r_x))
#         v_rel_z = c_z_d - (bot.sym_z_d + bot.sym_l*(-bot.sym_w_2*r_x + bot.sym_w_1*r_y))
        
#         # 3-D C3BF Candidate
#         self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z \
#             + norm(v_rel_x, v_rel_y, v_rel_z)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.sym_r**2)

#         # # HO-CBF Candidate
#         # gamma = 1
#         # p = 0.5
#         # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z \
#         #     + gamma*(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.sym_r**2)**p

#         # self.h =1

#         # # 2D HO-CBF Candidate - z
#         # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
#         #     + norm(1, 1, 1)*sqrt(norm(p_rel_x, p_rel_y, 0.0001)**2 - bot.sym_r**2)
        
#         # # 2D HO-CBF Candidate - x
#         # self.h = p_rel_z*v_rel_z + p_rel_y*v_rel_y \
#         #     + sqrt(norm(p_rel_z, p_rel_y, 0.0001)**2 - bot.sym_r**2)

#         # # 2-D C3BF Candidate - z
#         # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
#         #     + norm(v_rel_x, v_rel_y,0.001)*sqrt(norm(p_rel_x, p_rel_y,0.001)**2 - bot.sym_r**2)
        
#         # # 2-D C3BF Candidate - x
#         # self.h = p_rel_z*v_rel_z + p_rel_y*v_rel_y \
#         #     + norm(0.001, v_rel_y,v_rel_z)*sqrt(norm(0.001, p_rel_y,p_rel_z)**2 - bot.sym_r**2)

#         # # 2-D HO-CBF Candidate
#         # gamma = 1
#         # p = 0.5
#         # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
#         #     + gamma*(norm(p_rel_x, p_rel_y, 0.00001)**2 - bot.sym_r**2)**p

#         # # # Classical CBF
#         # self.h = norm(c_x - bot.sym_x, c_y - bot.sym_y, c_z - bot.sym_z)**2/bot.sym_r**2 -1
            
#         rho_h_by_rho_x = diff(self.h, bot.sym_x)
#         rho_h_by_rho_y = diff(self.h, bot.sym_y)
#         rho_h_by_rho_z = diff(self.h, bot.sym_z)
#         rho_h_by_rho_x_d = diff(self.h, bot.sym_x_d)
#         rho_h_by_rho_y_d = diff(self.h, bot.sym_y_d)
#         rho_h_by_rho_z_d = diff(self.h, bot.sym_z_d)
#         rho_h_by_rho_phi = diff(self.h, bot.sym_phi)
#         rho_h_by_rho_theta = diff(self.h, bot.sym_theta)
#         rho_h_by_rho_psi = diff(self.h, bot.sym_psi)
#         rho_h_by_rho_w_1 = diff(self.h, bot.sym_w_1)
#         rho_h_by_rho_w_2 = diff(self.h, bot.sym_w_2)
#         rho_h_by_rho_w_3 = diff(self.h, bot.sym_w_3)
        
#         Delta_h_wrt_bot = Matrix([[rho_h_by_rho_x, 
#                                     rho_h_by_rho_y, 
#                                     rho_h_by_rho_z,
#                                     rho_h_by_rho_x_d, 
#                                     rho_h_by_rho_y_d, 
#                                     rho_h_by_rho_z_d,
#                                     rho_h_by_rho_phi,
#                                     rho_h_by_rho_theta, 
#                                     rho_h_by_rho_psi,
#                                     rho_h_by_rho_w_1, 
#                                     rho_h_by_rho_w_2,
#                                     rho_h_by_rho_w_3]])
        
#         self.C = Delta_h_wrt_bot*self.g
#         self.u_ref = self.u_ref.reshape((4,1))
#         n = self.C * self.u_ref
#         n_f = Delta_h_wrt_bot*self.f
#         self.Psi = self.gamma*self.h
#         self.Psi += n_f[0] + n[0]
                 
#     def solve_QP(self, bot):
#         """
#         Solving Quadratic Program to set the optimal controls. This functions
#         substitutes the values in symbolic expression and evalutes closed form
#         solution to QP, modifies the reference control and sets the optimal 
#         control

#         Parameters
#         ----------
#         bot_list : list
#             list of Unicycle bot objects .

#         Returns
#         -------
#         TYPE: tuple of 2 numpy arrays
#             first numpy array in the tuple returns of state of CBF if they are
#             active or inactive 1 denotes active CBF and 0 denotes inactive CBF.
#             second numpy array in the tupes returns the value of CBF since in 
#             C3BF the value of the function is directly proportional to how 
#             unsafe system is.
#         """

#         # build value substitution list        
#         self.h = np.array(re(self.h.xreplace(d)))
#         self.Psi = np.array(re(self.Psi.xreplace(d)))
#         self.C = np.array(re(self.C.xreplace(d)))
#         self.B = (self.C).transpose()

#         if self.Psi<0:
#             self.u_safe = - np.matmul(self.B, np.linalg.inv(np.matmul(self.C,self.B).astype('float64'))).dot(self.Psi)
#         else:
#             self.u_safe = 0
            
#         self.u_star = self.u_ref + self.u_safe

#         # print(self.u_star)
#         state_of_h1, state_of_h2 = 0, 0 
#         term_h1 , term_h2 = 0, 0
#         return ((state_of_h1, state_of_h2), (term_h1, term_h2))