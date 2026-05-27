# app.py
# Proyecto Final - Métodos de Optimización
# Aplicación Streamlit para minimizar funciones de n variables usando:
# 1) Gradiente Descendente
# 2) Gradiente Conjugado no lineal (Fletcher-Reeves)
# 3) Método de Newton globalizado
# Todos usan búsqueda de línea con Condiciones de Wolfe.

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="Optimizador con Condiciones de Wolfe",
    page_icon="📉",
    layout="wide",
)

EPS = 1e-12


# ============================================================
# Estructuras de datos
# ============================================================

@dataclass
class OptimizationResult:
    method: str
    x_star: np.ndarray
    f_star: float
    iterations: int
    final_error: float
    stop_reason: str
    elapsed_time: float
    history: pd.DataFrame


# ============================================================
# Utilidades numéricas
# ============================================================

def format_vector(x: np.ndarray, decimals: int = 8) -> str:
    """Convierte un vector numpy en texto legible."""
    return "(" + ", ".join([f"{value:.{decimals}g}" for value in x]) + ")"


def safe_float(value) -> float:
    """Convierte una evaluación simbólica o numérica a float."""
    value = float(value)
    if not np.isfinite(value):
        raise FloatingPointError("La función devolvió un valor no finito.")
    return value


def parse_initial_point(text: str, n: int) -> np.ndarray:
    """Lee el punto inicial desde texto separado por comas."""
    try:
        values = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
    except Exception as exc:
        raise ValueError("El punto de partida debe tener números separados por comas. Ejemplo: 2, -1, 0") from exc

    if len(values) != n:
        raise ValueError(f"El punto de partida debe contener exactamente {n} valores.")

    return np.array(values, dtype=float)


def build_sympy_functions(expr_text: str, n: int) -> Tuple[sp.Expr, List[sp.Symbol], Callable, Callable, Callable]:
    """
    Construye función objetivo, gradiente y Hessiana desde texto usando SymPy.

    Variables esperadas:
    x1, x2, ..., xn

    Ejemplos válidos:
    x1**2 + x2**2
    (x1-1)^2 + 10*(x2-x1^2)^2
    exp(x1) + x2^2
    ln(x1^2 + x2^2) - 2*x1*x2
    """
    variables = sp.symbols(f"x1:{n+1}")
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    allowed: Dict[str, object] = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "ln": sp.log,
        "pi": sp.pi,
        "E": sp.E,
        "abs": sp.Abs,
    }

    for var in variables:
        allowed[str(var)] = var

    try:
        expr = parse_expr(
            expr_text,
            local_dict=allowed,
            transformations=transformations,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(
            "No se pudo interpretar la función. Usa variables x1, x2, ..., xn. "
            "Ejemplo: (x1-1)^2 + (x2+2)^2"
        ) from exc

    free_symbols = expr.free_symbols
    invalid_symbols = [s for s in free_symbols if s not in set(variables)]
    if invalid_symbols:
        raise ValueError(f"La función contiene símbolos no permitidos: {invalid_symbols}")

    grad_expr = [sp.diff(expr, var) for var in variables]
    hess_expr = sp.Matrix(grad_expr).jacobian(variables)

    f_raw = sp.lambdify(variables, expr, modules=["numpy"])
    g_raw = sp.lambdify(variables, grad_expr, modules=["numpy"])
    h_raw = sp.lambdify(variables, hess_expr, modules=["numpy"])

    def f(x: np.ndarray) -> float:
        return safe_float(f_raw(*x))

    def grad(x: np.ndarray) -> np.ndarray:
        value = np.array(g_raw(*x), dtype=float).reshape(n)
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("El gradiente devolvió valores no finitos.")
        return value

    def hess(x: np.ndarray) -> np.ndarray:
        value = np.array(h_raw(*x), dtype=float).reshape(n, n)
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("La Hessiana devolvió valores no finitos.")
        return value

    return expr, list(variables), f, grad, hess


# ============================================================
# Búsqueda de línea con Wolfe
# ============================================================

def wolfe_line_search(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    c1: float,
    c2: float,
    alpha0: float = 1.0,
    alpha_max: float = 50.0,
    max_ls_iter: int = 40,
) -> Tuple[float, str]:
    """
    Búsqueda de línea que intenta cumplir Condiciones de Wolfe débiles:

    1) Armijo:
       f(x + alpha*p) <= f(x) + c1*alpha*grad(x)^T*p

    2) Curvatura:
       grad(x + alpha*p)^T*p >= c2*grad(x)^T*p

    Requiere que p sea dirección de descenso:
       grad(x)^T*p < 0
    """
    phi0 = f(x)
    g0 = grad(x)
    derphi0 = float(np.dot(g0, p))

    if derphi0 >= 0:
        return 0.0, "La dirección no es de descenso; no se puede aplicar Wolfe."

    def phi(alpha: float) -> float:
        return f(x + alpha * p)

    def derphi(alpha: float) -> float:
        return float(np.dot(grad(x + alpha * p), p))

    def zoom(alpha_lo: float, alpha_hi: float) -> Tuple[float, str]:
        """Fase zoom por bisección dentro de un intervalo que contiene un alpha aceptable."""
        phi_lo = phi(alpha_lo)

        for _ in range(max_ls_iter):
            alpha_j = 0.5 * (alpha_lo + alpha_hi)

            if alpha_j <= EPS:
                return max(alpha_j, EPS), "Zoom llegó a un alpha muy pequeño."

            try:
                phi_j = phi(alpha_j)
            except Exception:
                alpha_hi = alpha_j
                continue

            if (phi_j > phi0 + c1 * alpha_j * derphi0) or (phi_j >= phi_lo):
                alpha_hi = alpha_j
            else:
                try:
                    derphi_j = derphi(alpha_j)
                except Exception:
                    alpha_hi = alpha_j
                    continue

                if derphi_j >= c2 * derphi0:
                    return alpha_j, "Wolfe satisfecho."

                if derphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha_j
                phi_lo = phi_j

            if abs(alpha_hi - alpha_lo) < EPS:
                return max(alpha_j, EPS), "Zoom terminó por intervalo muy pequeño."

        return max(0.5 * (alpha_lo + alpha_hi), EPS), "Zoom alcanzó el máximo de iteraciones."

    alpha_prev = 0.0
    phi_prev = phi0
    alpha = min(max(alpha0, EPS), alpha_max)

    for i in range(max_ls_iter):
        try:
            phi_alpha = phi(alpha)
        except Exception:
            alpha = 0.5 * (alpha + alpha_prev)
            continue

        armijo_rhs = phi0 + c1 * alpha * derphi0

        if (phi_alpha > armijo_rhs) or (i > 0 and phi_alpha >= phi_prev):
            return zoom(alpha_prev, alpha)

        try:
            derphi_alpha = derphi(alpha)
        except Exception:
            return zoom(alpha_prev, alpha)

        if derphi_alpha >= c2 * derphi0:
            return alpha, "Wolfe satisfecho."

        if derphi_alpha >= 0:
            return zoom(alpha, alpha_prev)

        alpha_prev = alpha
        phi_prev = phi_alpha
        alpha = min(2.0 * alpha, alpha_max)

    return alpha, "Búsqueda de línea alcanzó el máximo de iteraciones."


def verify_wolfe_conditions(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    alpha: float,
    c1: float,
    c2: float,
) -> Tuple[bool, bool, float, float]:
    """
    Verifica explícitamente las dos condiciones de Wolfe para el paso aceptado.

    Retorna:
    - Armijo satisfecho: bool
    - Curvatura satisfecha: bool
    - grad(x_k)^T p_k
    - grad(x_k + alpha p_k)^T p_k
    """
    if alpha <= EPS:
        return False, False, np.nan, np.nan

    f0 = f(x)
    g0 = grad(x)
    derphi0 = float(np.dot(g0, p))
    x_next = x + alpha * p
    f_next = f(x_next)
    g_next = grad(x_next)
    derphi_next = float(np.dot(g_next, p))

    armijo_rhs = f0 + c1 * alpha * derphi0
    armijo_ok = bool(f_next <= armijo_rhs + 1e-10)
    curvature_ok = bool(derphi_next >= c2 * derphi0 - 1e-10)

    return armijo_ok, curvature_ok, derphi0, derphi_next


# ============================================================
# Algoritmos de optimización
# ============================================================

def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int,
    tol: float,
    c1: float,
    c2: float,
    alpha0: float,
) -> OptimizationResult:
    """Gradiente descendente con búsqueda de línea Wolfe."""
    start = time.perf_counter()
    x = x0.astype(float).copy()
    rows = []
    stop_reason = "Máximo número de iteraciones alcanzado."

    for k in range(max_iter + 1):
        fx = f(x)
        g = grad(x)
        error = float(np.linalg.norm(g, ord=2))

        rows.append({
            "iteracion": k,
            "f(x)": fx,
            "error_norma_gradiente": error,
            "alpha": np.nan,
            "Wolfe Armijo": "",
            "Wolfe Curvatura": "",
            "grad(x)^T p": np.nan,
            "grad(x+alpha*p)^T p": np.nan,
            "criterio": "Evaluación inicial" if k == 0 else "",
            **{f"x{i+1}": x[i] for i in range(len(x))}
        })

        if error <= tol:
            stop_reason = "Convergencia: norma del gradiente menor o igual a la tolerancia."
            break

        if k == max_iter:
            break

        p = -g
        alpha, ls_msg = wolfe_line_search(f, grad, x, p, c1, c2, alpha0=alpha0)
        armijo_ok, curvature_ok, derphi0, derphi_next = verify_wolfe_conditions(f, grad, x, p, alpha, c1, c2)
        rows[-1]["Wolfe Armijo"] = "Sí" if armijo_ok else "No"
        rows[-1]["Wolfe Curvatura"] = "Sí" if curvature_ok else "No"
        rows[-1]["grad(x)^T p"] = derphi0
        rows[-1]["grad(x+alpha*p)^T p"] = derphi_next

        if alpha <= EPS:
            stop_reason = "Parada: la búsqueda de línea no encontró un paso útil."
            rows[-1]["criterio"] = ls_msg
            break

        x = x + alpha * p
        rows[-1]["alpha"] = alpha
        rows[-1]["criterio"] = ls_msg

    elapsed = time.perf_counter() - start
    history = pd.DataFrame(rows)

    return OptimizationResult(
        method="Gradiente Descendente",
        x_star=x,
        f_star=f(x),
        iterations=len(history) - 1,
        final_error=float(np.linalg.norm(grad(x), ord=2)),
        stop_reason=stop_reason,
        elapsed_time=elapsed,
        history=history,
    )


def conjugate_gradient_fr(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int,
    tol: float,
    c1: float,
    c2: float,
    alpha0: float,
) -> OptimizationResult:
    """
    Gradiente conjugado no lineal con fórmula Fletcher-Reeves:

    beta_k = ||g_{k+1}||^2 / ||g_k||^2
    p_{k+1} = -g_{k+1} + beta_k p_k

    Si la dirección deja de ser de descenso, se reinicia con -gradiente.
    """
    start = time.perf_counter()
    x = x0.astype(float).copy()
    g = grad(x)
    p = -g
    rows = []
    stop_reason = "Máximo número de iteraciones alcanzado."

    for k in range(max_iter + 1):
        fx = f(x)
        error = float(np.linalg.norm(g, ord=2))

        rows.append({
            "iteracion": k,
            "f(x)": fx,
            "error_norma_gradiente": error,
            "alpha": np.nan,
            "beta_FR": np.nan,
            "Wolfe Armijo": "",
            "Wolfe Curvatura": "",
            "grad(x)^T p": np.nan,
            "grad(x+alpha*p)^T p": np.nan,
            "criterio": "Evaluación inicial" if k == 0 else "",
            **{f"x{i+1}": x[i] for i in range(len(x))}
        })

        if error <= tol:
            stop_reason = "Convergencia: norma del gradiente menor o igual a la tolerancia."
            break

        if k == max_iter:
            break

        if float(np.dot(g, p)) >= 0:
            p = -g
            rows[-1]["criterio"] = "Reinicio: dirección no descendente."

        alpha, ls_msg = wolfe_line_search(f, grad, x, p, c1, c2, alpha0=alpha0)
        armijo_ok, curvature_ok, derphi0, derphi_next = verify_wolfe_conditions(f, grad, x, p, alpha, c1, c2)
        rows[-1]["Wolfe Armijo"] = "Sí" if armijo_ok else "No"
        rows[-1]["Wolfe Curvatura"] = "Sí" if curvature_ok else "No"
        rows[-1]["grad(x)^T p"] = derphi0
        rows[-1]["grad(x+alpha*p)^T p"] = derphi_next

        if alpha <= EPS:
            stop_reason = "Parada: la búsqueda de línea no encontró un paso útil."
            rows[-1]["criterio"] = ls_msg
            break

        x_new = x + alpha * p
        g_new = grad(x_new)

        denom = max(float(np.dot(g, g)), EPS)
        beta = float(np.dot(g_new, g_new)) / denom

        p_new = -g_new + beta * p

        # Reinicio defensivo para mantener dirección de descenso.
        if float(np.dot(g_new, p_new)) >= 0:
            p_new = -g_new
            beta = 0.0
            ls_msg += " Reinicio FR por dirección no descendente."

        rows[-1]["alpha"] = alpha
        rows[-1]["beta_FR"] = beta
        rows[-1]["criterio"] = ls_msg

        x = x_new
        g = g_new
        p = p_new

    elapsed = time.perf_counter() - start
    history = pd.DataFrame(rows)

    return OptimizationResult(
        method="Gradiente Conjugado Fletcher-Reeves",
        x_star=x,
        f_star=f(x),
        iterations=len(history) - 1,
        final_error=float(np.linalg.norm(grad(x), ord=2)),
        stop_reason=stop_reason,
        elapsed_time=elapsed,
        history=history,
    )


def newton_method(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int,
    tol: float,
    c1: float,
    c2: float,
    alpha0: float,
) -> OptimizationResult:
    """
    Método de Newton globalizado con búsqueda de línea Wolfe.

    Dirección ideal:
        H(x_k) p_k = -grad(x_k)

    Si la Hessiana es singular o la dirección de Newton no es de descenso,
    se usa -gradiente como respaldo para que Wolfe pueda funcionar.
    """
    start = time.perf_counter()
    x = x0.astype(float).copy()
    rows = []
    stop_reason = "Máximo número de iteraciones alcanzado."

    for k in range(max_iter + 1):
        fx = f(x)
        g = grad(x)
        H = hess(x)
        error = float(np.linalg.norm(g, ord=2))

        rows.append({
            "iteracion": k,
            "f(x)": fx,
            "error_norma_gradiente": error,
            "alpha": np.nan,
            "direccion": "",
            "Wolfe Armijo": "",
            "Wolfe Curvatura": "",
            "grad(x)^T p": np.nan,
            "grad(x+alpha*p)^T p": np.nan,
            "criterio": "Evaluación inicial" if k == 0 else "",
            **{f"x{i+1}": x[i] for i in range(len(x))}
        })

        if error <= tol:
            stop_reason = "Convergencia: norma del gradiente menor o igual a la tolerancia."
            break

        if k == max_iter:
            break

        direction_label = "Newton"

        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            p = -g
            direction_label = "Gradiente por Hessiana singular"

        if not np.all(np.isfinite(p)) or float(np.dot(g, p)) >= 0:
            p = -g
            direction_label = "Gradiente por dirección Newton no descendente"

        alpha, ls_msg = wolfe_line_search(f, grad, x, p, c1, c2, alpha0=alpha0)
        armijo_ok, curvature_ok, derphi0, derphi_next = verify_wolfe_conditions(f, grad, x, p, alpha, c1, c2)
        rows[-1]["Wolfe Armijo"] = "Sí" if armijo_ok else "No"
        rows[-1]["Wolfe Curvatura"] = "Sí" if curvature_ok else "No"
        rows[-1]["grad(x)^T p"] = derphi0
        rows[-1]["grad(x+alpha*p)^T p"] = derphi_next

        if alpha <= EPS:
            stop_reason = "Parada: la búsqueda de línea no encontró un paso útil."
            rows[-1]["criterio"] = ls_msg
            rows[-1]["direccion"] = direction_label
            break

        x = x + alpha * p
        rows[-1]["alpha"] = alpha
        rows[-1]["direccion"] = direction_label
        rows[-1]["criterio"] = ls_msg

    elapsed = time.perf_counter() - start
    history = pd.DataFrame(rows)

    return OptimizationResult(
        method="Newton globalizado",
        x_star=x,
        f_star=f(x),
        iterations=len(history) - 1,
        final_error=float(np.linalg.norm(grad(x), ord=2)),
        stop_reason=stop_reason,
        elapsed_time=elapsed,
        history=history,
    )


def run_selected_method(
    method_name: str,
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int,
    tol: float,
    c1: float,
    c2: float,
    alpha0: float,
) -> OptimizationResult:
    """Ejecuta un método a partir de su nombre en la interfaz."""
    if method_name == "Gradiente Descendente":
        return gradient_descent(f, grad, x0, max_iter, tol, c1, c2, alpha0)
    if method_name == "Gradiente Conjugado":
        return conjugate_gradient_fr(f, grad, x0, max_iter, tol, c1, c2, alpha0)
    if method_name == "Newton":
        return newton_method(f, grad, hess, x0, max_iter, tol, c1, c2, alpha0)
    raise ValueError(f"Método no reconocido: {method_name}")


def choose_best_result(results: List[OptimizationResult]) -> OptimizationResult:
    """Escoge el resultado más robusto priorizando menor error y luego menor f(x*)."""
    return min(results, key=lambda r: (r.final_error, r.f_star, r.iterations, r.elapsed_time))


# ============================================================
# Visualizaciones
# ============================================================

def plot_convergence(history: pd.DataFrame):
    """Gráfico error vs iteraciones en escala logarítmica."""
    fig, ax = plt.subplots()
    errors = history["error_norma_gradiente"].to_numpy(dtype=float)
    errors = np.maximum(errors, EPS)

    ax.plot(history["iteracion"], errors, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("Error: ||∇f(x)||₂")
    ax.set_title("Convergencia del método")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    return fig


def plot_1d_function(f: Callable[[np.ndarray], float], x_star: np.ndarray):
    """Visualización simple para funciones de una variable."""
    center = float(x_star[0])
    xs = np.linspace(center - 5, center + 5, 300)
    ys = []

    for value in xs:
        try:
            ys.append(f(np.array([value], dtype=float)))
        except Exception:
            ys.append(np.nan)

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.scatter([center], [f(x_star)], s=60)
    ax.set_xlabel("x1")
    ax.set_ylabel("f(x1)")
    ax.set_title("Visualización 1D alrededor del mínimo encontrado")
    ax.grid(True, linestyle="--", linewidth=0.5)

    return fig


def plot_2d_contour(f: Callable[[np.ndarray], float], history: pd.DataFrame):
    """Curvas de nivel para funciones de dos variables."""
    x_values = history["x1"].to_numpy(dtype=float)
    y_values = history["x2"].to_numpy(dtype=float)

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)

    # Margen para que la trayectoria no quede pegada al borde.
    margin_x = max(1.0, 0.3 * (x_max - x_min + EPS))
    margin_y = max(1.0, 0.3 * (y_max - y_min + EPS))

    x_grid = np.linspace(x_min - margin_x, x_max + margin_x, 120)
    y_grid = np.linspace(y_min - margin_y, y_max + margin_y, 120)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.empty_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f(np.array([X[i, j], Y[i, j]], dtype=float))
            except Exception:
                Z[i, j] = np.nan

    fig, ax = plt.subplots()
    finite_z = Z[np.isfinite(Z)]

    if finite_z.size > 0:
        levels = 25
        ax.contour(X, Y, Z, levels=levels)
    else:
        ax.text(0.5, 0.5, "No se pudo graficar la función en esta zona.", ha="center")

    ax.plot(x_values, y_values, marker="o")
    ax.scatter([x_values[-1]], [y_values[-1]], s=80)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Curvas de nivel y trayectoria del algoritmo")
    ax.grid(True, linestyle="--", linewidth=0.5)

    return fig


def plot_comparison_convergence(results: List[OptimizationResult]):
    """Compara la convergencia de varios métodos en un mismo gráfico."""
    fig, ax = plt.subplots()

    for result in results:
        hist = result.history
        errors = hist["error_norma_gradiente"].to_numpy(dtype=float)
        errors = np.maximum(errors, EPS)
        ax.plot(hist["iteracion"], errors, marker="o", label=result.method)

    ax.set_yscale("log")
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("Error: ||∇f(x)||₂")
    ax.set_title("Comparación de convergencia entre métodos")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    return fig


def comparison_dataframe(results: List[OptimizationResult]) -> pd.DataFrame:
    """Tabla resumen para comparar desempeño de los métodos."""
    rows = []
    for result in results:
        rows.append({
            "Método": result.method,
            "x*": format_vector(result.x_star, decimals=6),
            "f(x*)": result.f_star,
            "Iteraciones": result.iterations,
            "Error final ||∇f||₂": result.final_error,
            "Tiempo [s]": result.elapsed_time,
            "Criterio de parada": result.stop_reason,
        })
    return pd.DataFrame(rows)


def hessian_diagnostic(grad: Callable[[np.ndarray], np.ndarray], hess: Callable[[np.ndarray], np.ndarray], x_star: np.ndarray, tol: float) -> Dict[str, object]:
    """Evalúa gradiente, Hessiana y autovalores en el punto encontrado."""
    g_star = grad(x_star)
    H_star = hess(x_star)
    H_sym = 0.5 * (H_star + H_star.T)
    eigvals = np.linalg.eigvalsh(H_sym)
    grad_norm = float(np.linalg.norm(g_star, ord=2))

    eig_tol = max(1e-8, 100 * tol)
    if np.all(eigvals > eig_tol):
        hess_class = "Definida positiva"
        conclusion = "El punto encontrado cumple condición de segundo orden suficiente: mínimo local estricto."
    elif np.all(eigvals >= -eig_tol):
        hess_class = "Semidefinida positiva"
        conclusion = "El punto es compatible con mínimo local, pero la Hessiana semidefinida no garantiza mínimo estricto."
    elif np.all(eigvals < -eig_tol):
        hess_class = "Definida negativa"
        conclusion = "La Hessiana sugiere máximo local, no mínimo. Revisa función o punto inicial."
    else:
        hess_class = "Indefinida"
        conclusion = "La Hessiana sugiere punto silla o región no convexa; el punto no puede certificarse como mínimo local estricto."

    if grad_norm > max(tol, 1e-6) * 10:
        conclusion += " Además, la norma del gradiente no quedó suficientemente baja, por lo que conviene aumentar iteraciones o revisar parámetros."

    return {
        "gradiente": g_star,
        "norma_gradiente": grad_norm,
        "hessiana": H_star,
        "autovalores": eigvals,
        "clasificacion_hessiana": hess_class,
        "conclusion": conclusion,
    }


def show_hessian_diagnostic(result: OptimizationResult, grad: Callable[[np.ndarray], np.ndarray], hess: Callable[[np.ndarray], np.ndarray], tol: float):
    """Muestra diagnóstico matemático del resultado."""
    st.subheader("Valor agregado: diagnóstico matemático del punto encontrado")

    try:
        diag = hessian_diagnostic(grad, hess, result.x_star, tol)
        c1_diag, c2_diag, c3_diag = st.columns(3)
        c1_diag.metric("Norma del gradiente", f"{diag['norma_gradiente']:.3e}")
        c2_diag.metric("Clasificación Hessiana", str(diag["clasificacion_hessiana"]))
        c3_diag.metric("Autovalor mínimo", f"{np.min(diag['autovalores']):.3e}")

        st.markdown(f"**Conclusión:** {diag['conclusion']}")

        with st.expander("Ver gradiente, Hessiana y autovalores"):
            st.markdown("**Gradiente evaluado en el punto encontrado:**")
            st.code(np.array2string(diag["gradiente"], precision=6, suppress_small=True))

            st.markdown("**Hessiana evaluada en el punto encontrado:**")
            st.dataframe(pd.DataFrame(diag["hessiana"]), use_container_width=True)

            st.markdown("**Autovalores de la Hessiana:**")
            st.code(np.array2string(diag["autovalores"], precision=6, suppress_small=True))
    except Exception as exc:
        st.warning("No se pudo construir el diagnóstico de Hessiana para esta función/punto.")
        st.exception(exc)


def build_execution_report(expr_text: str, variables: List[sp.Symbol], result: OptimizationResult, c1: float, c2: float, alpha0: float, tol: float) -> str:
    """Genera un resumen descargable de la ejecución."""
    lines = [
        "INFORME DE EJECUCIÓN - OPTIMIZADOR CON CONDICIONES DE WOLFE",
        "",
        f"Función objetivo: {expr_text}",
        f"Variables: {', '.join(str(v) for v in variables)}",
        f"Método: {result.method}",
        f"Punto mínimo encontrado: {format_vector(result.x_star)}",
        f"Valor f(x*): {result.f_star:.12g}",
        f"Iteraciones realizadas: {result.iterations}",
        f"Error final ||grad f(x*)||_2: {result.final_error:.6e}",
        f"Tiempo de ejecución [s]: {result.elapsed_time:.6f}",
        f"Criterio de parada: {result.stop_reason}",
        "",
        "Parámetros Wolfe:",
        f"c1 Armijo: {c1}",
        f"c2 Curvatura: {c2}",
        f"Alpha inicial: {alpha0}",
        f"Tolerancia: {tol}",
        "",
        "Últimas iteraciones:",
        result.history.tail(10).to_string(index=False),
    ]
    return "\n".join(lines)


def show_math_details():
    """Sección de fundamentos matemáticos en LaTeX."""
    with st.expander("Detalles matemáticos usados", expanded=False):
        st.markdown("### Fundamento matemático del algoritmo")
        st.markdown(
            "Esta sección muestra las fórmulas usadas por la aplicación en notación matemática "
            "para que el procedimiento sea trazable y defendible."
        )

        st.markdown("#### 1. Actualización iterativa")
        st.latex(r"x_{k+1}=x_k+\alpha_k p_k")
        st.markdown(
            "- $x_k$: punto actual.\n"
            "- $p_k$: dirección de búsqueda.\n"
            "- $\alpha_k$: tamaño de paso obtenido mediante búsqueda de línea."
        )

        st.markdown("#### 2. Error final o criterio de convergencia")
        st.latex(r"e_k=\|\nabla f(x_k)\|_2")
        st.markdown(
            "El algoritmo se considera convergente cuando la norma euclidiana del gradiente "
            "es menor o igual que la tolerancia definida por el usuario."
        )

        st.markdown("#### 3. Condiciones de Wolfe")
        st.markdown("**Primera condición de Wolfe / Armijo:**")
        st.latex(r"f(x_k+\alpha_k p_k)\leq f(x_k)+c_1\alpha_k\nabla f(x_k)^T p_k")

        st.markdown("**Segunda condición de Wolfe / condición de curvatura:**")
        st.latex(r"\nabla f(x_k+\alpha_k p_k)^T p_k\geq c_2\nabla f(x_k)^T p_k")

        st.markdown(
            "Estas condiciones controlan que el paso $\alpha_k$ produzca una disminución "
            "suficiente de la función objetivo y que la dirección mantenga una curvatura aceptable."
        )

        st.markdown("#### 4. Direcciones de búsqueda usadas")
        st.markdown("**Gradiente Descendente:**")
        st.latex(r"p_k=-\nabla f(x_k)")

        st.markdown("**Gradiente Conjugado no lineal Fletcher-Reeves:**")
        st.latex(r"p_{k+1}=-g_{k+1}+\beta_k p_k")
        st.latex(r"\beta_k=\frac{g_{k+1}^Tg_{k+1}}{g_k^Tg_k}")

        st.markdown("**Método de Newton:**")
        st.latex(r"\nabla^2 f(x_k)p_k=-\nabla f(x_k)")
        st.markdown(
            "En Newton, si la Hessiana es singular o la dirección calculada no es de descenso, "
            "la aplicación usa temporalmente la dirección de máximo descenso para mantener válida "
            "la búsqueda de línea con Wolfe."
        )


def display_single_result(
    result: OptimizationResult,
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    n_int: int,
    expr_text: str,
    variables: List[sp.Symbol],
    c1: float,
    c2: float,
    alpha0: float,
    tol: float,
    show_visuals: bool = True,
):
    """Renderiza resultados completos de un método."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Método", result.method)
    col2.metric("Iteraciones", result.iterations)
    col3.metric("Tiempo de ejecución", f"{result.elapsed_time:.6f} s")

    col4, col5 = st.columns(2)
    col4.metric("Valor f(x*)", f"{result.f_star:.10g}")
    col5.metric("Error final ||∇f(x*)||₂", f"{result.final_error:.3e}")

    st.markdown(f"**Punto mínimo encontrado:** `{format_vector(result.x_star)}`")
    st.markdown(f"**Criterio de parada alcanzado:** {result.stop_reason}")

    show_hessian_diagnostic(result, grad, hess, tol)

    st.subheader("Gráfico de convergencia")
    st.pyplot(plot_convergence(result.history))

    st.subheader("Valor agregado: tabla de iteraciones con verificación Wolfe")
    st.caption("Las columnas Wolfe muestran si el paso aceptado cumple Armijo y curvatura en cada iteración.")
    st.dataframe(result.history, use_container_width=True)

    csv = result.history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar historial en CSV",
        data=csv,
        file_name=f"historial_{result.method.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"download_historial_{result.method}",
    )

    report = build_execution_report(expr_text, variables, result, c1, c2, alpha0, tol).encode("utf-8")
    st.download_button(
        "Descargar informe de ejecución TXT",
        data=report,
        file_name=f"informe_ejecucion_{result.method.lower().replace(' ', '_')}.txt",
        mime="text/plain",
        key=f"informe_txt_{result.method}_{show_visuals}_{result.iterations}",
    )

    if show_visuals:
        if n_int == 1:
            st.subheader("Valor agregado: visualización de la función")
            st.pyplot(plot_1d_function(f, result.x_star))
        elif n_int == 2:
            st.subheader("Valor agregado: curvas de nivel y trayectoria")
            st.pyplot(plot_2d_contour(f, result.history))
        else:
            st.info("Para n > 2 se muestra la convergencia, la tabla de iteraciones, la verificación Wolfe y el diagnóstico de Hessiana. Las curvas de nivel solo aplican para n = 2.")

    show_math_details()


# ============================================================
# Interfaz Streamlit
# ============================================================

st.title("📉 Optimizador multivariable con Condiciones de Wolfe")
st.caption(
    "Proyecto Final de Métodos de Optimización: Gradiente Descendente, "
    "Gradiente Conjugado y Newton con búsqueda de línea Wolfe."
)

with st.sidebar:
    st.header("Datos de entrada")

    n = st.number_input(
        "Número de variables",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="La función debe usar variables x1, x2, ..., xn.",
    )

    method = st.selectbox(
        "Método de optimización",
        options=[
            "Gradiente Descendente",
            "Gradiente Conjugado",
            "Newton",
        ],
    )

    compare_methods = st.checkbox(
        "Valor agregado: comparar automáticamente los 3 métodos",
        value=False,
        help="Ejecuta Gradiente Descendente, Gradiente Conjugado y Newton con la misma función, punto inicial y tolerancia.",
    )

    st.markdown("### Función objetivo")
    example_by_n = {
        1: "(x1-2)^2 + 1",
        2: "(x1-1)^2 + 10*(x2-x1^2)^2",
        3: "x1^2 + x2^2 + x3^2",
    }
    default_expr = example_by_n.get(int(n), " + ".join([f"x{i+1}^2" for i in range(int(n))]))

    expr_text = st.text_area(
        "f(x)",
        value=default_expr,
        height=100,
        help="Usa x1, x2, ..., xn. Puedes escribir ^ o ** para potencias.",
    )

    x0_text = st.text_input(
        "Punto de partida",
        value=", ".join(["0"] * int(n)),
        help="Valores separados por comas. Ejemplo para 2 variables: 2, -1",
    )

    max_iter = st.number_input(
        "Número máximo de iteraciones",
        min_value=1,
        max_value=10000,
        value=200,
        step=10,
    )

    tol = st.number_input(
        "Tolerancia de convergencia",
        min_value=1e-12,
        max_value=1.0,
        value=1e-6,
        step=1e-6,
        format="%.12f",
    )

    st.markdown("### Parámetros Wolfe")
    c1 = st.number_input(
        "c1 Armijo",
        min_value=1e-8,
        max_value=0.49,
        value=1e-4,
        format="%.8f",
        help="Valor típico: 1e-4. Debe cumplir 0 < c1 < c2 < 1.",
    )

    c2 = st.number_input(
        "c2 Curvatura",
        min_value=0.01,
        max_value=0.99,
        value=0.90,
        format="%.4f",
        help="Valores típicos: 0.9 para Newton/Gradiente; 0.1 a 0.9 para CG.",
    )

    alpha0 = st.number_input(
        "Alpha inicial",
        min_value=1e-8,
        max_value=100.0,
        value=1.0,
        format="%.8f",
    )

    calculate = st.button("Calcular", type="primary")


with st.expander("Ayuda rápida: cómo escribir funciones"):
    st.markdown(
        """
        **Variables:** `x1`, `x2`, ..., `xn`.

        **Operaciones válidas:**
        - Potencias: `x1^2` o `x1**2`
        - Exponencial: `exp(x1)`
        - Logaritmo natural: `ln(x1)` o `log(x1)`
        - Trigonométricas: `sin(x1)`, `cos(x2)`, `tan(x1)`

        **Ejemplos:**
        - Cuadrática 2D: `(x1-3)^2 + (x2+1)^2`
        - Rosenbrock: `(x1-1)^2 + 10*(x2-x1^2)^2`
        - Logarítmica: `ln(x1^2+x2^2)-2*x1*x2`

        **Advertencia:** si la función tiene logaritmos, raíces o divisiones, el punto inicial y la trayectoria deben estar dentro del dominio.
        """
    )


if calculate:
    try:
        if not (0 < c1 < c2 < 1):
            st.error("Los parámetros Wolfe deben cumplir: 0 < c1 < c2 < 1.")
            st.stop()

        n_int = int(n)
        x0 = parse_initial_point(x0_text, n_int)
        expr, variables, f, grad, hess = build_sympy_functions(expr_text, n_int)

        st.subheader("Función interpretada")
        st.latex(r"f(" + ",".join([str(v) for v in variables]) + r") = " + sp.latex(expr))

        if compare_methods:
            st.success("Cálculo comparativo finalizado.")
            method_names = ["Gradiente Descendente", "Gradiente Conjugado", "Newton"]
            results = [
                run_selected_method(
                    name,
                    f,
                    grad,
                    hess,
                    x0,
                    int(max_iter),
                    float(tol),
                    float(c1),
                    float(c2),
                    float(alpha0),
                )
                for name in method_names
            ]

            best_result = choose_best_result(results)

            st.subheader("Valor agregado: comparación automática de métodos")
            st.markdown(
                "La aplicación ejecuta los tres algoritmos con los mismos datos de entrada y permite comparar "
                "iteraciones, error final, valor objetivo y tiempo de ejecución."
            )

            comp_df = comparison_dataframe(results)
            st.dataframe(comp_df, use_container_width=True)

            comp_csv = comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar comparación en CSV",
                data=comp_csv,
                file_name="comparacion_metodos.csv",
                mime="text/csv",
                key="download_comparacion_metodos",
            )

            st.markdown(
                f"**Método recomendado por menor error final:** `{best_result.method}` "
                f"con error `{best_result.final_error:.3e}` y `{best_result.iterations}` iteraciones."
            )

            st.subheader("Gráfico comparativo de convergencia")
            st.pyplot(plot_comparison_convergence(results))

            st.subheader("Detalle por método")
            tabs = st.tabs([r.method for r in results])
            for tab, result in zip(tabs, results):
                with tab:
                    display_single_result(
                        result,
                        f,
                        grad,
                        hess,
                        n_int,
                        expr_text,
                        variables,
                        float(c1),
                        float(c2),
                        float(alpha0),
                        float(tol),
                        show_visuals=False,
                    )

            st.info(
                "Lectura sugerida: Newton suele converger en menos iteraciones cuando la Hessiana es estable; "
                "Gradiente Descendente suele ser más lento pero robusto; Gradiente Conjugado suele quedar en un punto intermedio."
            )

        else:
            result = run_selected_method(
                method,
                f,
                grad,
                hess,
                x0,
                int(max_iter),
                float(tol),
                float(c1),
                float(c2),
                float(alpha0),
            )

            st.success("Cálculo finalizado.")
            st.subheader("Resultados esperados")
            display_single_result(
                result,
                f,
                grad,
                hess,
                n_int,
                expr_text,
                variables,
                float(c1),
                float(c2),
                float(alpha0),
                float(tol),
                show_visuals=True,
            )

    except Exception as exc:
        st.error("No se pudo ejecutar el algoritmo.")
        st.exception(exc)

else:
    st.info("Completa los datos en el panel lateral y presiona **Calcular**.")

    st.markdown(
        """
        ### Qué entrega esta aplicación

        - Punto mínimo encontrado.
        - Valor de la función objetivo en ese punto.
        - Número de iteraciones realizadas.
        - Error final usando norma del gradiente.
        - Criterio de parada.
        - Gráfico de convergencia en escala logarítmica.
        - Tabla de iteraciones con verificación explícita de Wolfe.
        - Comparación automática entre los tres métodos.
        - Diagnóstico de Hessiana, autovalores y clasificación del punto encontrado.
        - Informe descargable de ejecución.
        - Visualización adicional para funciones de 1 o 2 variables.
        """
    )
