import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
#import plotly.express as px
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle

# 1. Define the model function
def constrained_exp(t, A, T1, c):
    T2 = c * T1
    return A * np.exp(-t / T1) + (1 - A) * np.exp(-t / T2)

# 2. Refined fit function to return Teff
def fit_constrained(sp, label):
    t = np.arange(len(sp))
    y = np.array(sp)
    
    # normalize
    y = y / y[0]
    
    # initial guesses and bounds
    p0 = [0.5, 10.0, 5.0]  # A, T1, c
    bounds = ([0, 0, 1], [1, np.inf, np.inf])
    
    popt, _ = curve_fit(constrained_exp, t, y, p0=p0, bounds=bounds)
    
    A, T1, c = popt
    T2 = c * T1
    tau_avg = A * T1 + (1 - A) * T2
    
    # Plotting
    plt.plot(t, y, 'o', alpha=0.3, label=f'{label} data')
    plt.plot(t, constrained_exp(t, *popt), '-', linewidth=2, label=f'{label} fit')
    
    return tau_avg


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# triexponential model
def constrained_exp3(t, A1, A2, T1, T2, T3):
    A3 = 1 - A1 - A2
    return A1 * np.exp(-t / T1) + A2 * np.exp(-t / T2) + A3 * np.exp(-t / T3)


def fit_constrained3(sp, label):
    t = np.arange(len(sp))
    y = np.array(sp)

    # normalize
    y = y / y[0]

    # initial guesses
    p0 = [0.3, 0.3, 5.0, 20.0, 100.0]  # A1, A2, T1, T2, T3

    # bounds:
    # A1, A2 between 0 and 1
    # T1, T2, T3 > 0
    bounds = ([0, 0, 0, 0, 0],
              [1, 1, np.inf, np.inf, np.inf])

    popt, _ = curve_fit(constrained_exp3, t, y, p0=p0, bounds=bounds)

    A1, A2, T1, T2, T3 = popt
    A3 = 1 - A1 - A2

    # weighted average lifetime
    tau_avg = A1*T1 + A2*T2 + A3*T3

    # Plotting
    plt.plot(t, y, 'o', alpha=0.3, label=f'{label} data')
    plt.plot(t, constrained_exp3(t, *popt), '-', linewidth=2, label=f'{label} fit')

    print(f"{label}:")
    print(f"A1={A1:.3f}, T1={T1:.2f}")
    print(f"A2={A2:.3f}, T2={T2:.2f}")
    print(f"A3={A3:.3f}, T3={T3:.2f}")
    print(f"tau_avg={tau_avg:.2f}")

    return tau_avg



from scipy.optimize import curve_fit
import numpy as np

# tri-exponential
def exp3(t, A1, A2, T1, T2, T3):
    A3 = 1 - A1 - A2
    return (A1*np.exp(-t/T1) +
            A2*np.exp(-t/T2) +
            A3*np.exp(-t/T3))

def fit_exp3(t, y):
    # remove nan/inf
    mask = np.isfinite(t) & np.isfinite(y)
    t = np.array(t)[mask]
    y = np.array(y)[mask]

    # normalize
    y = y / y[0]

    # initial guess
    p0 = [0.3, 0.3, 5, 50, 500]

    bounds = ([0, 0, 0, 0, 0],
              [1, 1, np.inf, np.inf, np.inf])

    popt, _ = curve_fit(exp3, t, y, p0=p0, bounds=bounds, maxfev=10000)

    A1, A2, T1, T2, T3 = popt
    A3 = 1 - A1 - A2
    tau_avg = A1*T1 + A2*T2 + A3*T3

    return popt, tau_avg
