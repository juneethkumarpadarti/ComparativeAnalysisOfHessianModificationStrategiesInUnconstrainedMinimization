import numpy as np
from scipy.linalg import eigh
from scipy.optimize import line_search
import warnings

warnings.filterwarnings("ignore")

x0_rosenbrock = np.array([
    -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, 
    -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, 
    -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, 
    -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, 
    -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1, -1.2, 1])

x0_rastrigin = np.array([
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
])

x0_raydan2 =  np.array([-2.6, 0.2, -0.8, 4.4, 1.6, -2., -0.2, -0.7, 0.6, 4., 1.6, -4.9, 1.1, -2.7,
    -0.8, -0.4, -1.6, 0.9, -1.6, 1.8, -0.5, -5.4, 3.2, 0.5, 2., -1.1, 0., -2.5,
    -1.9, -1.6, -1.4, -1.1, 2.8, -4.6, -2.3, -1.7, -0.8, -4.1, -2.5, -0.1, -0.4, -1.6,
    -1.7, -1.4, 1., -1., 1.9, 0.3, 0.3, 1.2, -0.9, 1.6, 0.4, -2.4, -0.9, 2.5,
    2.9, 0.5, 1.6, -0.4, -0.9, -0.1, -1.2, 1., 1.7, 0.1, -2.9, 2.6, 0.6, 0.4,
    -1.3, -1.5, -2., -1.2, 0.7, 0.3, 0.9, -1.8, -2.4, -1.9])

x0_baele = np.array([1.7, 1.5])
x0_matyas = np.array([1., 1.])
x0_powellSingular = np.array([3.1, -1.1, 0.1, 1.1])

x0_sphere = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1.])

x0_Booth = np.array([1.7, 1])
x0_styblinskiTang = np.array([-1.47324152,  0.49673852, -2.89436936,  0.31939736, -2.92024347,
        2.6318428 ,  0.88544741,  0.45674346, -1.30210341,  1.02396322,
       -0.07965365, -0.99344333, -2.43898356, -1.86686443, -1.68874239,
        0.31162211, -0.07195219,  1.32018995, -0.59239476, -1.28090167,
       -1.62006908,  2.02677464, -1.98750287,  0.47603548, -1.59182531,
       -0.57415358, -2.8031093 ,  2.53835644, -0.26715256, -2.90923585,
       -1.66133079,  1.15261476, -2.06930848,  2.40982186,  2.02700595,
       -2.07262423,  1.80883997, -2.61208181, -1.3334028 ,  0.22700875,
       -0.12518561, -1.7190387 , -1.54282633,  0.17397659,  2.21692077,
       -1.87689146,  2.10514312, -2.17331088,  1.38122833, -2.84709877,
        0.70915935, -0.97872913,  2.46317403,  0.44485043, -2.9655821 ,
        1.21846566, -2.96656807, -2.51368721, -0.03859086,  0.59973074,
       -2.35270824,  1.05677596, -2.03409803,  2.79407891,  2.71846961,
        2.56746417, -2.68100112,  0.90296064, -2.03679354, -1.3905106 ,
        1.89493342,  0.40578936,  1.321912  ,  2.98469667, -2.61208159,
       -1.91548807, -1.23394083, -2.47229101, -2.56006037, -2.67610997,
        1.7582876 ,  1.12681896, -0.53983385,  0.01871711, -1.22145186,
        0.64906112, -2.01246027,  1.90502047,  2.38104907,  0.28354208,
       -0.19068322, -1.50977157,  2.11209543, -1.81928124,  2.43947164,
        2.92959686, -0.2733077 ,  2.50927411,  0.23436507,  2.37487693])

x0_mcCormick = np.array([-1, -1])
x0_easom = np.array([3.64159265, 2.64159265])
x0_wood = np.array([-1, 1, 1, 1])
x0_helicalValley = np.array([1, 0.1, 0.1])

### ROSENBROCK FUNCTION
def rosenbrock_function(x):
    return sum(
        100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)
    )


def rosenbrock_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    grad[-1] = 200 * (x[-1] - x[-2] ** 2)
    for i in range(1, len(x) - 1):
        grad[i] = (
            200 * (x[i] - x[i - 1] ** 2)
            - 400 * x[i] * (x[i + 1] - x[i] ** 2)
            - 2 * (1 - x[i])
        )
    return grad

def rosenbrock_hessian(x):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n-1):
        H[i, i] = 1200 * x[i]**2 - 400 * x[i+1] + 2
        H[i, i+1] = -400 * x[i]
        H[i+1, i] = -400 * x[i]
    H[n-1, n-1] = 200
    return H


### RASTRIGIN FUNCTION
def rastrigin_function(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rastrigin_gradient(x):
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

def rastrigin_hessian(x):
    return np.diag(2 - 40 * np.pi**2 * np.cos(2 * np.pi * x))


###RAYDAN2_FUNCTION
def raydan2_function(x):
    return np.sum(np.exp(x) - x)

def raydan2_gradient(x):
    return np.exp(x) - 1

def raydan2_hessian(x):
    return np.diag(np.exp(x))


### BEALE FUNCTION
def beale_function(x):
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


def beale_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = (
        2 * (1.5 - x[0] + x[0] * x[1]) * (-1 + x[1])
        + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (-1 + x[1] ** 2)
        + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (-1 + x[1] ** 3)
    )
    grad[1] = (
        2 * (1.5 - x[0] + x[0] * x[1]) * (x[0])
        + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (2 * x[0] * x[1])
        + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (3 * x[0] * x[1] ** 2)
    )
    return grad


def beale_hessian(x):
    H = np.zeros((2, 2))
    H[0, 0] = (
        2 * (-1 + x[1]) ** 2 + 2 * (-1 + x[1] ** 2) ** 2 + 2 * (-1 + x[1] ** 3) ** 2
    )
    H[0, 1] = (
        2 * (-1 + x[1]) * x[0]
        + 4 * (-1 + x[1] ** 2) * x[0] * x[1]
        + 6 * (-1 + x[1] ** 3) * x[0] * x[1] ** 2
    )
    H[1, 0] = H[0, 1]
    H[1, 1] = (
        2 * x[0] ** 2
        + 4 * (2 * x[0] ** 2 * x[1] ** 2)
        + 6 * (x[0] ** 2 * (x[1] ** 3) ** 2)
    )
    return H


### MATYAS FUNCTION
def matyas_function(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def matyas_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 0.52 * x[0] - 0.48 * x[1]
    grad[1] = 0.52 * x[1] - 0.48 * x[0]
    return grad


def matyas_hessian(x):
    H = np.array([[0.52, -0.48], [-0.48, 0.52]])
    return H


## POWELL SINGULAR FUNCTION
def powell_singular_function(x):
    return (
        (x[0] + 10 * x[1]) ** 2
        + 5 * (x[2] - x[3]) ** 2
        + (x[1] - 2 * x[2]) ** 4
        + 10 * (x[0] - x[3]) ** 4
    )


def powell_singular_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * (x[0] + 10 * x[1]) + 40 * (x[0] - x[3]) ** 3
    grad[1] = 20 * (x[0] + 10 * x[1]) + 4 * (x[1] - 2 * x[2]) ** 3
    grad[2] = 10 * (x[2] - x[3]) - 8 * (x[1] - 2 * x[2]) ** 3
    grad[3] = -10 * (x[2] - x[3]) - 40 * (x[0] - x[3]) ** 3
    return grad


def powell_singular_hessian(x):
    H = np.zeros((4, 4))
    H[0, 0] = 2 + 120 * (x[0] - x[3]) ** 2
    H[0, 1] = H[1, 0] = 20
    H[0, 3] = H[3, 0] = -120 * (x[0] - x[3]) ** 2
    H[1, 1] = 200 + 12 * (x[1] - 2 * x[2]) ** 2
    H[1, 2] = H[2, 1] = -24 * (x[1] - 2 * x[2]) ** 2
    H[2, 2] = 10 + 48 * (x[1] - 2 * x[2]) ** 2
    H[2, 3] = H[3, 2] = -10
    H[3, 3] = 10 + 120 * (x[0] - x[3]) ** 2
    return H


### SPHERE FUNCTION
def sphere_function(x):
    return np.sum(x**2)


def sphere_gradient(x):
    return 2 * x


def sphere_hessian(x):
    return 2 * np.eye(len(x))


### BOOTH FUNCTION
def booth_function(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def booth_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
    grad[1] = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
    return grad


def booth_hessian(x):
    H = np.zeros((2, 2))
    H[0, 0] = 10
    H[0, 1] = H[1, 0] = 8
    H[1, 1] = 10
    return H


### STYBLINSKI TANG
def styblinski_tang_function(x):
    return np.sum(0.5 * (x**4 - 16 * x**2 + 5 * x))


def styblinski_tang_gradient(x):
    return 2 * x**3 - 16 * x + 2.5


def styblinski_tang_hessian(x):
    return np.diag(6 * x**2 - 16)


### MCCORMICK
def mccormick_function(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1


def mccormick_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = np.cos(x[0] + x[1]) + 2 * (x[0] - x[1]) - 1.5
    grad[1] = np.cos(x[0] + x[1]) - 2 * (x[0] - x[1]) + 2.5
    return grad


def mccormick_hessian(x):
    H = np.zeros((2, 2))
    H[0, 0] = -np.sin(x[0] + x[1]) + 2
    H[0, 1] = H[1, 0] = -np.sin(x[0] + x[1]) - 2
    H[1, 1] = -np.sin(x[0] + x[1]) + 2
    return H


### EASOM
def easom_function(x):
    return (
        -np.cos(x[0])
        * np.cos(x[1])
        * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
    )


def easom_gradient(x):
    grad = np.zeros_like(x)
    exp_term = np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
    grad[0] = exp_term * (
        np.sin(x[0]) * np.cos(x[1]) + 2 * (x[0] - np.pi) * np.cos(x[0]) * np.cos(x[1])
    )
    grad[1] = exp_term * (
        np.cos(x[0]) * np.sin(x[1]) + 2 * (x[1] - np.pi) * np.cos(x[0]) * np.cos(x[1])
    )
    return grad


def easom_hessian(x):
    H = np.zeros((2, 2))
    exp_term = np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
    cos_x0 = np.cos(x[0])
    cos_x1 = np.cos(x[1])
    sin_x0 = np.sin(x[0])
    sin_x1 = np.sin(x[1])

    H[0, 0] = exp_term * (
        (2 * (x[0] - np.pi) * sin_x0 * cos_x1)
        + 2
        * cos_x1
        * (cos_x0 + (x[0] - np.pi) * (2 * (x[0] - np.pi) * cos_x0 - sin_x0))
    )
    H[1, 1] = exp_term * (
        (2 * (x[1] - np.pi) * cos_x0 * sin_x1)
        + 2
        * cos_x0
        * (cos_x1 + (x[1] - np.pi) * (2 * (x[1] - np.pi) * cos_x1 - sin_x1))
    )
    H[0, 1] = H[1, 0] = exp_term * (
        (sin_x0 * sin_x1)
        + 2 * (x[0] - np.pi) * (x[1] - np.pi) * cos_x0 * cos_x1
        + (x[0] - np.pi) * cos_x0 * sin_x1
        + (x[1] - np.pi) * sin_x0 * cos_x1
    )

    return H


### WOOD
def wood_function(x):
    f1 = 10 * (x[1] - x[0] ** 2)
    f2 = 1 - x[0]
    f3 = np.sqrt(90) * (x[3] - x[2] ** 2)
    f4 = 1 - x[2]
    f5 = np.sqrt(10) * (x[1] + x[3] - 2)
    f6 = 1 / np.sqrt(10) * (x[1] - x[3])
    return 100 * f1**2 + f2**2 + 100 * f3**2 + f4**2 + 10 * f5**2 + f6**2


def wood_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -20 * (x[1] - x[0] ** 2) * 2 * x[0] - 2 * (1 - x[0])
    grad[1] = (
        20 * (x[1] - x[0] ** 2)
        + 20 * np.sqrt(10) * (x[1] + x[3] - 2)
        + 2 / np.sqrt(10) * (x[1] - x[3])
    )
    grad[2] = -2 * np.sqrt(90) * (x[3] - x[2] ** 2) * 2 * x[2] - 2 * (1 - x[2])
    grad[3] = (
        20 * np.sqrt(90) * (x[3] - x[2] ** 2)
        + 20 * np.sqrt(10) * (x[1] + x[3] - 2)
        - 2 / np.sqrt(10) * (x[1] - x[3])
    )
    return grad


def wood_hessian(x):
    H = np.zeros((4, 4))
    H[0, 0] = 120 * x[0] ** 2 - 40 * x[1] + 2
    H[0, 1] = H[1, 0] = -40 * x[0]
    H[1, 1] = 220 + 200 * np.sqrt(10) + 20 + 2 / np.sqrt(10)
    H[2, 2] = 1080 * x[2] ** 2 - 360 * x[3] + 2
    H[2, 3] = H[3, 2] = -360 * x[2]
    H[3, 3] = 220 * np.sqrt(90) + 200 * np.sqrt(10) + 20 - 2 / np.sqrt(10)
    return H


### HELICAL VALLEY FUNCTION
def theta(x1, x2):
    if x1 > 0:
        return (1 / (2 * np.pi)) * np.arctan(x2 / x1)
    elif x1 < 0:
        return (1 / (2 * np.pi)) * np.arctan(x2 / x1) + 0.5
    else:
        # For x1 = 0, handle according to the sign of x2
        return 0.25 if x2 >= 0 else -0.25  # Assumes continuity at x2 = 0


def helical_valley_function(x):
    theta_val = theta(x[0], x[1])
    f = (
        100 * (x[2] - 10 * theta_val) ** 2
        + (x[0] ** 2 + x[1] ** 2 - 1) ** 2
        + x[2] ** 2
    )
    return f


def helical_valley_gradient(x):
    theta_val = theta(x[0], x[1])
    r = x[0] ** 2 + x[1] ** 2
    df_dx1 = 400 * x[0] * (x[0] ** 2 + x[1] ** 2 - 1) - 1000 * (
        x[2] - 10 * theta_val
    ) * (x[1] / r)
    df_dx2 = 400 * x[1] * (x[0] ** 2 + x[1] ** 2 - 1) + 1000 * (
        x[2] - 10 * theta_val
    ) * (x[0] / r)
    df_dx3 = 200 * x[2] - 200 * (x[2] - 10 * theta_val)
    return np.array([df_dx1, df_dx2, df_dx3])


def helical_valley_hessian(x):
    theta_val = theta(x[0], x[1])
    r = x[0] ** 2 + x[1] ** 2
    df_dx1dx1 = (
        400 * (x[0] ** 2 + x[1] ** 2 - 1)
        + 800 * x[0] ** 2
        + 1000 * (x[2] - 10 * theta_val) * (x[1] ** 2 - x[0] ** 2) / r**2
    )
    df_dx1dx2 = (
        800 * x[0] * x[1] - 1000 * (x[2] - 10 * theta_val) * 2 * x[0] * x[1] / r**2
    )
    df_dx1dx3 = -1000 * x[1] / r
    df_dx2dx1 = df_dx1dx2
    df_dx2dx2 = (
        400 * (x[0] ** 2 + x[1] ** 2 - 1)
        + 800 * x[1] ** 2
        + 1000 * (x[2] - 10 * theta_val) * (x[0] ** 2 - x[1] ** 2) / r**2
    )
    df_dx2dx3 = 1000 * x[0] / r
    df_dx3dx1 = df_dx1dx3
    df_dx3dx2 = df_dx2dx3
    df_dx3dx3 = 200
    return np.array(
        [
            [df_dx1dx1, df_dx1dx2, df_dx1dx3],
            [df_dx2dx1, df_dx2dx2, df_dx2dx3],
            [df_dx3dx1, df_dx3dx2, df_dx3dx3],
        ]
    )


test_functions = {
    "ROSENBROCK FUNCTION": [
        rosenbrock_function,
        rosenbrock_gradient,
        rosenbrock_hessian,
        100,
        x0_rosenbrock,
    ],
    "RASTRIGIN FUNCTION": [
        rastrigin_function,
        rastrigin_gradient,
        rastrigin_hessian,
        90,
        x0_rastrigin,
    ],
    "RAYDAN2 FUNCTION": [
        raydan2_function,
        raydan2_gradient,
        raydan2_hessian,
        80,
        x0_raydan2,
    ],
    "BEALE FUNCTION": [beale_function, beale_gradient, beale_hessian, 2, x0_baele,],
    "MATYAS FUNCTION": [
        matyas_function,
        matyas_gradient,
        matyas_hessian,
        2,
        x0_matyas,
    ],
    "POWELL SINGULAR FUNCTION": [
        powell_singular_function,
        powell_singular_gradient,
        powell_singular_hessian,
        4,
        x0_powellSingular,
    ],
    "SPHERE FUNCTION": [sphere_function, sphere_gradient, sphere_hessian, 75, x0_sphere,],
    "BOOTH FUNCTION": [booth_function, booth_gradient, booth_hessian, 2, x0_Booth,],
    "STYBLINSKI TANG FUNCTION": [
        styblinski_tang_function,
        styblinski_tang_gradient,
        styblinski_tang_hessian,
        100,
        x0_styblinskiTang,
    ],
    "MCCORMICK FUNCTION": [
        mccormick_function,
        mccormick_gradient,
        mccormick_hessian,
        2,
        x0_mcCormick,
    ],
    "EASOM FUNCTION": [easom_function, easom_gradient, easom_hessian, 2, x0_easom, ],
    "WOOD FUNCTION": [wood_function, wood_gradient, wood_hessian, 4, x0_wood,],
    "HELICAL VALLEY FUNCTION": [
        helical_valley_function,
        helical_valley_gradient,
        helical_valley_hessian,
        3,
        x0_helicalValley,
    ],
}


def strategy_1(H, max_num=1000):
    eigenvalues, eigenvectors = eigh(H)
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    delta = 1e-5
    if lambda_min >= delta * lambda_max:
        return eigenvalues, eigenvectors
    tau = (lambda_max - max_num * lambda_min) / (max_num - 1)
    adjusted_eigenvalues = eigenvalues + tau
    return adjusted_eigenvalues, eigenvectors


def strategy_2(H, grad, mu=0.1):
    d_min = np.min(np.diag(H))
    if d_min > 0:
        tau = 0
    else:
        tau = mu - d_min
    I = np.eye(H.shape[0])
    k = 0
    while True:
        H_adj = H + tau * I
        try:
            L = np.linalg.cholesky(H_adj)
            break
        except np.linalg.LinAlgError:
            tau = max(2 * tau, mu)
            k += 1
    y = np.linalg.solve(L, -grad)
    p = np.linalg.solve(L.T, y)

    return p


def strategy_3(H, max_num=1000):
    eigenvalues, eigenvectors = eigh(H)
    max_eigval = np.max(eigenvalues)
    min_eigval = max_eigval / max_num
    adjusted_eigvals = np.maximum(eigenvalues, min_eigval)
    return adjusted_eigvals, eigenvectors


def optimize(
    function_name,
    fn,
    grad_fn,
    hess_fn,
    n,
    x0,
    grad_tol=1e-8,
    step_tol=1e-8,
    max_iter=1000,
    display_flag=1,
    mod_flag=0,
):
    # Initialize starting point
    x = np.array(x0, dtype=np.float64)
    strategies=[]

    if mod_flag == 1:
        strategies = [strategy_1]
    elif mod_flag == 2:
        strategies = [strategy_2]
    elif mod_flag == 3:
        strategies = [strategy_3]

    if display_flag > 0:
        print(f"\n{'='*50}\n{function_name}\n{'='*50}")

    for index in range(len(strategies)):
        if display_flag > 0:
            print(f"\n{strategies[index].__name__}\n")

        num_func_evals = 0
        grad_tol_reached = False
        step_tol_reached = False

        for itr in range(max_iter):
            grad = grad_fn(x)
            num_func_evals += 1

            if np.linalg.norm(grad) < grad_tol:
                grad_tol_reached = True
                if display_flag > 0:
                    print(f"Gradient tolerance reached at iteration {itr+1}")
                break

            hessian_val = hess_fn(x)
            num_func_evals += 1
            hessian_val = (hessian_val + hessian_val.T) / 2
            if mod_flag == 2:
                search_direction = strategies[index](hessian_val,grad)
            elif mod_flag == 1 or mod_flag == 3:
                eigenvalues, eigenvectors = strategies[index](hessian_val)
                search_direction = -eigenvectors@((eigenvectors.T@grad)/eigenvalues)
                

            step_size, feval, _, _, _, _ = line_search(
                fn, grad_fn, x, search_direction, grad, c1=0.01, c2=0.1
            )

            if feval is not None:
                num_func_evals += feval

            if step_size is None or step_size < step_tol:
                step_tol_reached = True
                if display_flag > 0:
                    print(f"Step tolerance reached at iteration {itr+1}")
                break

            x += step_size * search_direction

            if display_flag == 2:
                print(
                    f"Iteration {itr+1}: x = {x}, Gradient Norm = {np.linalg.norm(grad)}, Step Length = {step_size}"
                )

        if display_flag > 0:
            print("Optimized parameters:", x)
            print("Iterations:", itr + 1)
            print("Function evaluations:", num_func_evals)
            print("Final gradient norm:", np.linalg.norm(grad))
            if (itr + 1) == max_iter:
                print("Algorithm halted. Maximum number of iterations reached.")
            if grad_tol_reached:
                print("Algorithm halted. Gradient tolerance reached.")
            if step_tol_reached:
                print("Algorithm halted. Step tolerance reached.")


for function_name, functions in test_functions.items():
    fn, grad_fn, hess_fn, n, x0 = functions
    # x0 = np.random.randn(n)
    optimize(
        function_name,
        fn,
        grad_fn,
        hess_fn,
        n,
        x0,
        grad_tol=1e-8,
        step_tol=1e-8,
        max_iter=1000,
        display_flag=1,
        mod_flag=3,  # 0 for all the strategies
    )
