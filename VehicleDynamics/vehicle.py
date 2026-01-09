import numpy as np
import yaml
from pathlib import Path



dt=0.1

class Aircraftmodel:
    """For the UAV definition of values not changing"""
    def __init__(self,model):

        script_dir = Path(__file__).parent
        config_path = script_dir / 'Vehicleparams.yaml'
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        self.model=model

        #Physical properties
        self.mass = params['mass']
        self.I = np.diag(params['inertia'])
        self.I_inv = np.linalg.inv(self.I)
        self.S = params['wing_area']

        #aerodynamic model
        self.CO=np.zeros(6)
        self.derivatives = params.get('stability_derivatives', {})
        self.control_deriv = params.get('control_derivatives', {})
        
        # Propulsion
        self.T_max = params['propulsion']['T_max']
        self.delta_t = params['propulsion']['delta_t']


    def stability_derivatives(self):
        """
        Build stability derivative matrix from configuration.
        
        Returns:
            A: 6x6 stability derivative matrix
        """
        self.A = np.zeros((6, 6))
        
        # Longitudinal derivatives (with defaults to prevent errors if a derivative is missing)
        self.A[0, 0] = self.derivatives.get('Xu', 0.0)  # X-force due to u
        self.A[0, 2] = self.derivatives.get('Xw', 0.0)  # X-force due to w
        self.A[2, 2] = self.derivatives.get('Zw', 0.0)  # Z-force due to w
        self.A[4, 4] = self.derivatives.get('Mq', 0.0)  # Pitching moment due to q
        
        # Lateral-directional derivatives (with defaults to prevent errors if a derivative is missing)
        self.A[1, 1] = self.derivatives.get('Yv', 0.0)  # Y-force due to v
        self.A[3, 3] = self.derivatives.get('Lp', 0.0)  # Rolling moment due to p
        self.A[5, 5] = self.derivatives.get('Nr', 0.0)  # Yawing moment due to r
        
        return self.A
    
    def control_derivatives(self):
        """
        Build control derivative matrix (B matrix: 6x3).
        Maps control vector [delta_e, delta_a, delta_r] to coefficients [CL, CD, CY, Cl, Cm, Cn]
        
        Returns:
            B: 6x3 control derivative matrix
        """
        self.B = np.array([
            # CL control derivatives [delta_e, delta_a, delta_r]
            [self.control_deriv.get('CLdelta_e', 0.0),
             self.control_deriv.get('CLdelta_a', 0.0),
             self.control_deriv.get('CLdelta_r', 0.0)],
            
            # CD control derivatives
            [self.control_deriv.get('CDdelta_e', 0.0),
             self.control_deriv.get('CDdelta_a', 0.0),
             self.control_deriv.get('CDdelta_r', 0.0)],
            
            # CY control derivatives
            [self.control_deriv.get('CYdelta_e', 0.0),
             self.control_deriv.get('CYdelta_a', 0.0),
             self.control_deriv.get('CYdelta_r', 0.0)],
            
            # Cl control derivatives
            [self.control_deriv.get('Cldelta_e', 0.0),
             self.control_deriv.get('Cldelta_a', 0.0),
             self.control_deriv.get('Cldelta_r', 0.0)],
            
            # Cm control derivatives
            [self.control_deriv.get('Cmdelta_e', 0.0),
             self.control_deriv.get('Cmdelta_a', 0.0),
             self.control_deriv.get('Cmdelta_r', 0.0)],
            
            # Cn control derivatives
            [self.control_deriv.get('Cndelta_e', 0.0),
             self.control_deriv.get('Cndelta_a', 0.0),
             self.control_deriv.get('Cndelta_r', 0.0)]
        ])
        
        return self.B
    
    def get_derivative(self, name):
        """Get a specific stability derivative value."""
        return self.derivatives.get(name, None)
    
    def update_derivative(self, name, value):
        """Update a stability derivative value at runtime."""
        self.derivatives[name] = value
    
    def stability_analysis(self):
        """
        Re(λ) < 0 → stable
        Re(λ) > 0 → unstable
        Imaginary → oscillatory modes
        As such it reveals phugoid,short period,dutch roll and spiral
        """
        eigvals, eigvecs = np.linalg.eig(self.A)
        return eigvals,eigvecs
    
    def thrust_force(self):
        """The delta_t isn't hardcoded nor is it put in parameters given it is a control input coming from the autopilot
        but for the sake of learning first given this part was challenging I had to 
        put it in parameters upon learning more about it I'll make adjustments to it """
        self.config = self._load_config()
        script_dir = Path(__file__).parent
        config_path = script_dir / 'Vehicleparams.yaml'
        with open(config_path, 'r') as f:
             params = yaml.safe_load(f)
        T_max= params['propulsion']['T_max']
        delta_t=params['propulsion']['delta_t']
        T = T_max * np.clip(delta_t, 0.0, 1.0)
        return np.array([T, 0.0, 0.0])

class UAVState:
    """For state variables evolving with time"""
    def __init__(self,model:Aircraftmodel):
        self.model=Aircraftmodel
        # Classic 12-state vector
        self.pos=np.zeros(3)
        self.V=np.zeros(3)
        self.omega=np.zeros(3)
        self.eta=np.zeros(3)

        #Aerodynamic force
        self.F=np.zeros(3)

        #Aerodynamic moment
        self.M=np.zeros(3)

        self.config = self._load_config()
        script_dir = Path(__file__).parent
        config_path = script_dir / 'Vehicleparams.yaml'
        with open(config_path, 'r') as f:
             params = yaml.safe_load(f)


        #Trim coefficients
        self.CO=np.zeros(6)

        #First calculations
        self._V_mag=0.0
        self.mass = params['mass']
        self.I = np.diag(params['inertia'])
        self.I_inv = np.linalg.inv(self.I)
        self.S = params['wing_area']

        self.derivatives = params.get('stability_derivatives', {})
        self.control_deriv = params.get('control_derivatives', {})

        
    @ property
    def V_mag(self):
        self._V_mag = np.linalg.norm(self.V) + 1e-6
        return np.linalg.norm(self.V) + 1e-6
    
    def vector(self):
        return np.concatenate([self.pos,self.V,self.eta,self.omega])
    
    def update(self,statedot,dt=0.1):
        state=self.vector()
        state=state + statedot* dt

        self.pos=state[0:3]
        self.V=state[3:6]
        self.eta=state[6:9]
        self.omega=state[9:12]
        self._V_mag = np.linalg.norm(self.V) + 1e-6
    
    def R_yaw(self,psi):
        c=np.cos(psi)
        s=np.sin(psi)
        return np.array([
            [c,-s,0],
            [s,c,-0],
            [0,0,1]
        ])
    
    def R_pitch(self,theta):
        c=np.cos(theta)
        s=np.sin(theta)
        return np.array([
            [c,0,s],
            [0,1,0],
            [-s,0,c]
        ])
    
    def R_roll(self,phi):
        c=np.cos(phi)
        s=np.sin(phi)
        return np.array([
            [1,0,0],
            [0,c,-s],
            [0,s,c]
        ])
    
    def R_body_to_inertial(self):
        psi,theta,phi= self.eta
        return (
            self.R_yaw(psi)@ self.R_pitch(theta) @ self.R_roll(phi))
    
    def R_inertial_to_body(self):
        return self.R_body_to_inertial().T
    
    
    """
    Implementation of aerodynamic co-efficients;
    CL(lift),CD(drag),CY(Side force)
    Cl(Roll moment),Cn(Yaw moment),Cm(Pitch moment)
    
    Linear combination of states;
    alpha(angle of attack),beta(side slip)
    p(roll rate),q(pitch rate),r(yaw rate)
    
    Control-surfaces that create moments in body frame influenced by
    dynamic pressure,wing area,wing span,wing chord
    delta_e(elevators),delta_a(ailerons),delta_r(rudders)"""
    
    #Aerodynamic coefficients=trim-coefficients + A-matrix + B-matrix
    # C=CO + A@x +B@u
    def aero_state_vector(self):
        #Linear combination of states[alpha,beta,p,q,r]
        alpha=np.arctan2(self.V[2],self.V[0])
        beta=np.arcsin(self.V[1]/self.V_mag)
        p,q,r = self.omega
        return np.array([alpha,beta,p,q,r])
    
    def control_surface_vector(self,delta_e,delta_a,delta_r):
        return np.array([delta_e,delta_a,delta_r])
    
    def aerodynamic_coefficients (self):
        x=self.aero_state_vector()
        u=self.control_surface_vector(delta_e=0,delta_a=0,delta_r=0)
        C = self.CO + (self.stability_derivatives() @ x) + (self.control_derivatives() @ u)
        return C
    
    def dynamic_pressure(self,rho=1.225):
        return 0.5 * rho * self.V_mag **2
    
    def wind_forces(self):
        C=self.aerodynamic_coefficients()
        CL,CD,CY = C[0],C[1],C[2]
        q=self.dynamic_pressure()
        F_wind = np.array([
        -CD * q * self.S,   # Drag (negative X-wind)
        CY * q * self.S,    # Side force (Y-wind)
        -CL * q * self.S    # Lift (negative Z-wind)
    ])
        return F_wind
    
    def R_wind_to_bodyframe(self,alpha,beta):
        ca=np.cos(alpha)
        cb=np.cos(beta)
        sa=np.sin(alpha)
        sb=np.sin(beta)
        return np.array([
            [ca*cb,-ca*sb,-sa],
            [sa,cb,0],
            [sa*cb,-sa*sb,ca]
        ])
    
    def wind_force_vector(self):
        F_wind=self.wind_forces()
        D= F_wind[1]
        L= F_wind[0]
        Y= F_wind[2]
        F_wind=np.array([
            -D,
            Y,
            -L
        ])
        alpha, beta = self.aero_state_vector()[:2]
        R_wb = self.R_wind_to_bodyframe(alpha,beta)
        self.F = R_wb @ F_wind
        return self.F
    
    def linearstatespace(self,delta_e=0,delta_a=0,delta_r=0):
        self.A=self.stability_derivatives()
        self.B=self.control_derivatives()
        x=self.aero_state_vector()
        u=self.control_surface_vector(delta_e,delta_a,delta_r)
        xdot = self.A @ x + self.B @ u
        return xdot
    
    def thrust_force(self):
        """The delta_t isn't hardcoded nor is it put in parameters given it is a control input coming from the autopilot
        but for the sake of learning first given this part was challenging I had to 
        put it in parameters upon learning more about it I'll make adjustments to it """
        self.config = self._load_config()
        script_dir = Path(__file__).parent
        config_path = script_dir / 'Vehicleparams.yaml'
        with open(config_path, 'r') as f:
             params = yaml.safe_load(f)
        T_max= params['propulsion']['T_max']
        delta_t=params['propulsion']['delta_t']
        T = T_max * np.clip(delta_t, 0.0, 1.0)
        return np.array([T, 0.0, 0.0])

    def inertia_matrix (self):
        """To encode ;roll agility
                      pitch stiffness
                      yaw sluggishness"""
        
        self.config = self._load_config()
        script_dir = Path(__file__).parent
        config_path = script_dir / 'Vehicleparams.yaml'
        with open(config_path, 'r') as f:
             params = yaml.safe_load(f)
        Ixx = params['Inertia'][0]
        Iyy=params['Inertia'][1]
        Izz=params['Inertia'][2]
        I=np.diag([Ixx,Iyy,Izz])
        return I
    
    def translational_dynamics(self,g=9.81):
        m=self.mass

        omega = self.omega
        V=self.V
        coriolis = np.cross(omega, V)

        F_wind=self.wind_forces()
        alpha, beta = self.aero_state_vector()[:2]
        R_wb = self.R_wind_to_bodyframe(alpha,beta)
        R_ib = self.R_inertial_to_body()

        F_thrust = self.thrust_force()
        F_aero = R_wb @ F_wind
        F=F_thrust + F_aero

        gravity = g * R_ib @ np.array([0, 0,-1])

        V_dot = (F / m) - coriolis - gravity
        return V_dot
    
    def update_linearvelocity(self):
        V_dot=self.translational_dynamics(g=9.81)
        self.V += V_dot * dt
