import sympy
from sympy import UnevaluatedExpr
import numpy as np
import control
import control.matlab
from scipy.integrate import solve_ivp


class LQRController:
    def __init__(self, A, B, C, D, Q, R, timestep):
        # Create a system and then discretize it
        continuous_sys = control.StateSpace(A, B, C, D)
        discrete_sys = control.matlab.c2d(continuous_sys, timestep)

        # Calculate LQR gain for infinite horizon LQR
        K, _, _ = control.dlqr(discrete_sys, Q, R)
        self.K = K

        # Get state dimensions
        A_dimensions = np.shape(A)
        self.state_dim = A_dimensions[1]

    def get_control_input(self, state):
        return -(self.K @ state.reshape(self.state_dim, 1))[0, 0]


class CartPolePlant:
    def __init__(
        self,
        length_value,
        mass_cart_value,
        mass_pole_value,
        inertia_pole_value,
        initial_state=np.zeros((4, 1)),
    ):
        # Variables
        t = sympy.symbols("t")  # time
        theta = sympy.symbols("theta", cls=sympy.Function)  # angle of the pole
        x = sympy.symbols("x", cls=sympy.Function)  # position of the cart
        force = sympy.symbols("F")

        # Parameters
        mass_cart = sympy.symbols("m_c")
        mass_pole = sympy.symbols("m_p")
        inertia_pole = sympy.symbols("I_p")
        length = sympy.symbols("l")
        gravity = sympy.symbols("g")

        # Define the state
        state = sympy.Matrix([[x(t)], [x(t).diff(t)], [theta(t)], [theta(t).diff(t)]])

        # Set model params in a dictionary
        params = {
            gravity: 9.81,
            length: length_value,
            mass_cart: mass_cart_value,
            mass_pole: mass_pole_value,
            inertia_pole: inertia_pole_value,
        }

        # Define governing equations
        eq1 = sympy.Eq(
            length * mass_pole * gravity * theta(t)
            - length * mass_pole * x(t).diff(t, 2),
            (inertia_pole + mass_pole * length**2) * theta(t).diff(t, 2),
        )
        eq2 = sympy.Eq(
            force,
            (mass_cart + mass_pole) * x(t).diff(t, 2)
            + mass_pole * length * theta(t).diff(t, 2),
        )

        linearized_sys = sympy.solve([eq1, eq2], [x(t).diff(t, 2), theta(t).diff(t, 2)])

        linearized_sys[x(t).diff(t, 2)] = sympy.expand(
            linearized_sys[x(t).diff(t, 2)]
        ).collect([x(t), x(t).diff(t), theta(t), theta(t).diff(t), force])
        linearized_sys[theta(t).diff(t, 2)] = sympy.expand(
            linearized_sys[theta(t).diff(t, 2)]
        ).collect([x(t), x(t).diff(t), theta(t), theta(t).diff(t), force])

        # Solve for state space matrices
        state_space_matricies = sympy.linear_eq_to_matrix(
            list(linearized_sys.values()),
            [x(t), x(t).diff(t), theta(t), theta(t).diff(t), force],
        )

        A_expression = sympy.zeros(4, 4)
        B_expression = sympy.zeros(4, 1)

        A_expression[0, 1] = 1  # Account for x dot
        A_expression[2, 3] = 1  # Account for theta dot

        A_expression[1, :] = state_space_matricies[0][1, :4]
        A_expression[3, :] = state_space_matricies[0][0, :4]

        b1 = state_space_matricies[0][1, 4]
        b3 = state_space_matricies[0][0, 4]

        B_expression[1] = b1
        B_expression[3] = b3

        A_evaluated_expression = A_expression.subs(params)
        B_evaluated_expression = B_expression.subs(params)

        self.A = np.array(A_evaluated_expression).astype(np.float64)
        self.B = np.array(B_evaluated_expression).astype(np.float64)
        state_space = UnevaluatedExpr(A_expression) * UnevaluatedExpr(
            state
        ) + UnevaluatedExpr(B_expression) * UnevaluatedExpr(force)

        self.C = np.array([0, 0, 1, 0])
        self.D = 0

        self.derivative = sympy.lambdify(
            [t, state, force], state_space.subs(params).doit()
        )

        self.state = initial_state

    def step(self, control_input, timestep):
        result = solve_ivp(
            self.derivative,
            [0, timestep],
            self.state,
            vectorized=True,
            args=(control_input,),
        )
        self.state = result.y[:, -1]

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_state_derivative(self, state, force):
        return self.derivative(0, state, force)

    def get_A(self):
        return self.A

    def get_B(self):
        return self.B

    def get_C(self):
        return self.C

    def get_D(self):
        return self.D
