# Quantum Harmonic Oscillator Model for Stock Return Distributions

## Overview
This project explores whether a quantum harmonic oscillator (QHO) model can approximate and forecast stock return distributions. 

Instead of assuming returns follow a normal distribution, this model represents the return distribution as a superposition of quantum harmonic oscillator eigenfunctions. The fitted wavefunction is then time-evolved and compared with future empirical return distributions.

The goal is to test whether a physics-inspired probabilistic model can capture the structure and evolution of financial returns.

---

## Data
- Asset: S&P 500 index (^GSPC)
- Source: Yahoo Finance (via `yfinance`)
- Time period:
  - Training: 2023
  - Future comparison: 2024
- Returns:
  - Log returns over fixed trading intervals
  - Standardized (zero mean, unit variance)

---

## Methodology

### 1. Return preprocessing
- Download daily adjusted closing prices
- Compute log returns over a chosen horizon
- Standardize returns:
  
  scaled_return = (return − mean) / standard deviation

---

### 2. Quantum harmonic oscillator model
The probability density is modeled as a weighted superposition of QHO eigenstates:

P(x) = Σ cₙ |ψₙ(x)|²

Where:
- ψₙ(x) = nth eigenfunction of the harmonic oscillator
- cₙ = fitted coefficients
- mω = fitted scaling parameter

The model is fit to the empirical return histogram using nonlinear least squares.

---

### 3. Time evolution
The wavefunction is evolved using the quantum time evolution equation:

ψₙ(x, t) = ψₙ(x) · exp(−iEₙt / ħ)

Where:
- Eₙ = energy level of the nth eigenstate
- t = scaled time

The evolved probability density is then compared to actual future return distributions.

---

## Results
The QHO model:

- Captures the central peak of empirical return distributions
- Produces structured, multi-modal shapes when needed
- Provides a reasonable approximation of future distributions after time evolution

In several cases, the time-evolved model matched the location and spread of future empirical distributions, though tail behavior varied.

---

## Key Insight
Quantum-inspired models may offer a flexible alternative to traditional Gaussian assumptions for modeling financial return distributions, particularly when distributions exhibit non-normal or multi-modal behavior.

---

## Repository Structure
