import numpy as np
import scipy.stats as st

np.random.seed(42)

data_t = np.arange(10) + 1.0
data_N = st.poisson.rvs(10.0**1.618*np.exp(-data_t/np.pi))


