def print_metrics(nb_tabs, category, best_idx, lambdaa, rmse_full, rmse_diagonal, rmcw, wavenumbers_size):
    tabs = "\t" * nb_tabs
    print(
        f"{tabs}"
        f"{category}: {best_idx:6},\t"
        f"Lambda: {lambdaa:7.4f},\t"
        f"RMSE: {rmse_full:.4f},\t"
        f"RMSE_DIAG: {rmse_diagonal:.4f},\t"
        f"RMCW: {rmcw:3}/{wavenumbers_size:3} ({rmcw / wavenumbers_size:6.4f})"
    )
