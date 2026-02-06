import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy.integrate import simps
from scipy.optimize import curve_fit
import math
from datetime import datetime, timedelta

# Stock Name and Date
stock = 'S&P 500'
ticker = '^GSPC'
start_date = '2023-01-01'
end_date = '2023-12-31'

# Fetch Data Function
def fetch_stock_data(ticker, start, end):
    print(f"Fetching data for {ticker} from {start} to {end}")
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False)
        data['Log Return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(20)) #trading period
        data.dropna(inplace=True)
        print("Data successfully retrieved!")
        return data
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return pd.DataFrame()

# Get Stock Data
stock_data = fetch_stock_data(ticker, start_date, end_date)
if stock_data.empty:
    print("No data available to analyze.")
    exit()

log_returns = stock_data['Log Return'].values
mean_return = np.mean(log_returns)
std_return = np.std(log_returns)
scaled_returns = (log_returns - mean_return) / std_return

# QHO Definitions
def quantum_harmonic_oscillator(x, n, m_omega, hbar=1):
    coeff = (m_omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * math.factorial(n))
    herm = eval_hermite(n, np.sqrt(m_omega / hbar) * x)
    gauss = np.exp(-m_omega * x**2 / (2 * hbar))
    return coeff * herm * gauss

x = np.linspace(-3.5, 3.5, 1000)

def qho_pdf(x, m_omega, *coeffs):
    probabilities = np.zeros_like(x)
    for n in range(len(coeffs)):
        probabilities += coeffs[n] * np.abs(quantum_harmonic_oscillator(x, n, m_omega))**2
    return probabilities / simps(probabilities, x)

# Fit QHO to Scaled Data
hist_values, bin_edges = np.histogram(scaled_returns, bins=100, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
initial_guess = [1] + [1] * 5
params, _ = curve_fit(qho_pdf, bin_centers, hist_values, p0=initial_guess)
m_omega = params[0]
coeffs = params[1:]
fitted_probabilities = qho_pdf(x, m_omega, *coeffs)

# Time Evolution Function
def time_evolve_qho(x, t, m_omega, coeffs, hbar=1):
    wavefunction_t = np.zeros_like(x, dtype=complex)
    for n in range(len(coeffs)):
        energy_n = hbar * m_omega * (n + 0.5)
        phase_factor = np.exp(-1j * energy_n * t / hbar)
        wavefunction_t += coeffs[n] * quantum_harmonic_oscillator(x, n, m_omega, hbar) * phase_factor
    probability_density = np.abs(wavefunction_t)**2
    return probability_density / simps(probability_density, x)

# Time Scaling 
# Assume 2π phase change occurs every 100 trading days
def scale_time(days, wrap_period=100):
    return 2 * np.pi * days / wrap_period

# Ask User for Time Evolution
time_unit = input("Enter the time unit for evolution ('weeks', 'months', 'years'): ").lower()
time_scaling = {'weeks': 7, 'months': 30, 'years': 365}
if time_unit not in time_scaling:
    print("Invalid time unit. Using weeks by default.")
    time_unit = 'weeks'

time_input = int(input(f"Enter the number of {time_unit} to evolve: "))
days_to_evolve = time_input * time_scaling[time_unit]
t_scaled = scale_time(days_to_evolve)

# Fetch Future Data
future_start = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
future_end = (pd.to_datetime(end_date) + timedelta(days=days_to_evolve)).strftime('%Y-%m-%d')
future_data = fetch_stock_data(ticker, future_start, future_end)
if future_data.empty:
    print("Future data is unavailable. Cannot complete time evolution comparison.")
    exit()

future_log_returns = future_data['Log Return'].values
future_scaled_returns = (future_log_returns - mean_return) / std_return

# Time Evolve Distribution
time_evolved_probabilities = time_evolve_qho(x, t_scaled, m_omega, coeffs)

# Plot Comparison
plt.figure(figsize=(14, 6))

# Original vs Fitted
plt.subplot(1, 2, 1)
plt.hist(scaled_returns, bins=100, density=True, alpha=0.5, label=f'Empirical ({start_date} to {end_date})')
plt.plot(x, fitted_probabilities, label='QHO Model Fit', linewidth=2)
plt.title(f'{stock} Original: Empirical vs QHO')
plt.xlabel('Scaled Log Return')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Time-Evolved vs Future
plt.subplot(1, 2, 2)
plt.hist(future_scaled_returns, bins=100, density=True, alpha=0.5, label=f'Empirical Future ({future_start} to {future_end})')
plt.plot(x, time_evolved_probabilities, label=f'Time-Evolved QHO ({time_input} {time_unit})', color='orange', linewidth=2)
plt.title(f'{stock} Future: QHO Time Evolution vs Future')
plt.xlabel('Scaled Log Return')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Diagnostic to Visualize Time Evolution Behavior
plt.figure(figsize=(10, 4))
for test_t in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
    evolved = time_evolve_qho(x, test_t, m_omega, coeffs)
    plt.plot(x, evolved, label=f't = {test_t:.2f}')
plt.title('Diagnostic: QHO Time Evolution Over Phase Cycle')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
