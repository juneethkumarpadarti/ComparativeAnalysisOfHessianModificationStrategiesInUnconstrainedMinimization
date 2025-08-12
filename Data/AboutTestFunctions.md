## 📊 Data

This project uses 11 benchmark test functions commonly used in numerical optimization research. These functions were chosen to represent a broad spectrum of optimization challenges, from simple convex surfaces to highly non-linear, multi-modal landscapes.

| Function Name             | Dimension | Characteristics                                   |
|--------------------------|-----------|-------------------------------------------------|
| Rosenbrock (n vars)       | Variable  | Narrow curved valley; challenging for gradient methods |
| Beale                    | 2         | Nonlinear, multimodal; multiple local minima    |
| Matyas                   | 2         | Simple, convex quadratic surface                 |
| Powell Singular          | 4         | Steep, narrow valleys; ill-conditioned           |
| Sphere (n vars)           | Variable  | Convex, smooth; easy to optimize                  |
| Booth                    | 2         | Quadratic with single global minimum             |
| Styblinski–Tang (n vars)  | Variable  | Multi-modal, many local minima                    |
| McCormick                | 2         | Non-convex, saddle points                         |
| Easom                    | 2         | Sharp global minimum in flat region               |
| Wood Function            | 4         | Coupled non-linear variables; ill-conditioned     |
| Helical Valley           | 3         | Twisted valley; challenging for Hessian-based methods |

### Data Characteristics:

- **Dimensionality:** Functions include 2D, 3D, 4D, and n-dimensional cases to test scalability.
- **Complexity:** Includes simple convex, ill-conditioned, multi-modal, and highly nonlinear problems.
- **Purpose:** To evaluate how each Hessian modification strategy adapts to varying landscape difficulty and dimensionality.

All functions are mathematically defined within the source code rather than stored as static datasets, ensuring flexibility for parameter changes and reproducibility.
