from scipy.stats import norm
from Main import p_map

if __name__ == "__main__":
    vect = norm.ppf([1-0.5, 1-0.2, 1-0.1, 1-5e-2, 1-2e-2, 1-1e-2, 1-5e-3,
                    1-2e-3, 1-1e-3, 1-5e-4, 1-2e-4, 1-1e-4, 1-5e-5,
                    1-2e-5, 1-1e-5, 1-5e-6, 1-2e-6])**2/2
    print(norm.ppf([1-0.5, 1-0.2, 1-0.1, 1-5e-2, 1-2e-2, 1-1e-2, 1-5e-3,
                    1-2e-3, 1-1e-3, 1-5e-4, 1-2e-4, 1-1e-4, 1-5e-5,
                    1-2e-5, 1-1e-5, 1-5e-6, 1-2e-6])**2/2)
    print(p_map(vect, 1))

    norm.cdf(0)