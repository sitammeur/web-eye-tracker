hyperparameters = {
    "Lasso Regression": {
        "param_grid": {
            "lasso__alpha": [
                1e-15,
                1e-10,
                1e-8,
                1e-3,
                1e-2,
                1e-1,
                0.5,
                1,
                5,
                10,
                20,
                30,
                35,
                40,
                45,
                50,
                55,
                100,
            ]
        }
    },
    "Ridge Regression": {
        "param_grid": {
            "ridge__alpha": [
                1e-15,
                1e-10,
                1e-8,
                1e-3,
                1e-2,
                1e-1,
                0.5,
                1,
                5,
                10,
                20,
                30,
                35,
                40,
                45,
                50,
                55,
                100,
            ]
        }
    },
    "Elastic Net": {
        "param_grid": {
            "elasticnet__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
            "elasticnet__l1_ratio": [0, 0.01, 0.2, 0.5, 0.8, 1],
        }
    },
    "Bayesian Ridge": {
        "param_grid": {
            "bayesianridge__alpha_init": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
            "bayesianridge__lambda_init": [1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        }
    },
    "SGD Regressor": {
        "param_grid": {
            "sgdregressor__alpha": [0.0001, 0.001, 0.01, 0.1],
            "sgdregressor__l1_ratio": [0, 0.2, 0.5, 0.7, 1],
            "sgdregressor__max_iter": [500, 1000],
            "sgdregressor__eta0": [0.0001, 0.001, 0.01],
        }
    },
    "Support Vector Regressor": {
        "param_grid": {
            "svr__C": [0.1, 1, 10, 100, 1000],
            "svr__gamma": [0.0001, 0.001, 0.01, 0.1, 1],
            "svr__kernel": ["linear", "rbf", "poly"],
        }
    },
}
