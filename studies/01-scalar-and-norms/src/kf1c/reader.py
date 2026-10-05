from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from scipy.optimize import minimize


CLASS_ORDER = (
    "APPLICABLE_ALIGNED",
    "APPLICABLE_OPPOSED",
    "NOT_APPLICABLE_NONOPPOSED",
    "NOT_APPLICABLE_APPARENTLY_OPPOSED",
)


@dataclasses.dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    zero_variance: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        _matrix(features)
        mean = features.mean(axis=0, dtype=np.float64)
        std = features.std(axis=0, ddof=0, dtype=np.float64)
        zero = std == 0.0
        scale = std.copy()
        scale[zero] = 1.0
        return cls(mean=mean, scale=scale, zero_variance=zero)

    def transform(self, features: np.ndarray) -> np.ndarray:
        _matrix(features)
        if features.shape[1] != self.mean.shape[0]:
            raise ValueError("STANDARDIZER_DIMENSION_MISMATCH")
        result = (features.astype(np.float64, copy=False) - self.mean) / self.scale
        result[:, self.zero_variance] = 0.0
        if not np.isfinite(result).all():
            raise ValueError("NONFINITE_STANDARDIZED_FEATURE")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": [float(value) for value in self.mean],
            "population_standard_deviation": [float(value) for value in self.scale],
            "zero_variance": [bool(value) for value in self.zero_variance],
        }


def _matrix(value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("FEATURE_MATRIX_INVALID")
    if not np.isfinite(value).all():
        raise ValueError("FEATURE_MATRIX_NONFINITE")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("SOFTMAX_NONFINITE")
    return probabilities


def convergence_contract(
    *,
    optimizer_success: bool,
    iterations: int,
    iterations_max: int,
    gradient_infinity_norm: float,
    gradient_infinity_norm_max: float,
    absolute_gradient_guard_max: float,
    objective_relative_change: float,
    objective_relative_change_max: float,
    finite_outputs: bool,
) -> bool:
    """A02 contract aligned with L-BFGS's alternative stop surfaces."""
    return bool(
        optimizer_success
        and iterations <= iterations_max
        and finite_outputs
        and gradient_infinity_norm <= absolute_gradient_guard_max
        and (
            gradient_infinity_norm <= gradient_infinity_norm_max
            or objective_relative_change <= objective_relative_change_max
        )
    )


@dataclasses.dataclass
class FixedMultinomialReader:
    standardizer: Standardizer
    coefficients: np.ndarray
    intercepts: np.ndarray
    convergence: dict[str, Any]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: list[str],
        *,
        l2_lambda: float = 1.0,
        gradient_tolerance: float = 1.0e-8,
        relative_objective_tolerance: float = 1.0e-12,
        max_iterations: int = 1000,
        absolute_gradient_guard: float = 1.0e-6,
    ) -> "FixedMultinomialReader":
        _matrix(features)
        if len(labels) != features.shape[0] or any(label not in CLASS_ORDER for label in labels):
            raise ValueError("READER_LABEL_DOMAIN_OR_LENGTH_INVALID")
        standardizer = Standardizer.fit(features.astype(np.float64, copy=False))
        x = standardizer.transform(features)
        y = np.asarray([CLASS_ORDER.index(label) for label in labels], dtype=np.int64)
        rows, columns = x.shape
        classes = len(CLASS_ORDER)
        history: list[float] = []

        # Remove the softmax common-shift nullspace using an exact zero-sum
        # parameterization.  This does not change the model or objective: the
        # L2-optimal coefficient rows already sum to zero, and intercepts are
        # identifiable only up to a common shift.
        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            split = (classes - 1) * columns
            free_weights = theta[:split].reshape(classes - 1, columns)
            free_intercepts = theta[split:]
            weights = np.vstack([free_weights, -free_weights.sum(axis=0, keepdims=True)])
            intercepts = np.concatenate([free_intercepts, [-free_intercepts.sum()]])
            return weights, intercepts

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            # Evaluate the objective and analytic gradient in extended
            # precision.  SciPy owns float64 iterates, but the frozen 1e-10
            # postcondition must not be defeated by avoidable reduction error.
            work_type = np.longdouble
            weights, intercepts = unpack(theta.astype(work_type, copy=False))
            work_x = x.astype(work_type, copy=False)
            logits = work_x @ weights.T + intercepts
            probabilities = _softmax(logits)
            loss = -np.log(np.maximum(probabilities[np.arange(rows), y], np.finfo(work_type).tiny)).mean()
            loss += 0.5 * l2_lambda * float(np.sum(weights * weights))
            diff = probabilities
            diff[np.arange(rows), y] -= 1.0
            diff /= rows
            weight_gradient = diff.T @ work_x + l2_lambda * weights
            intercept_gradient = diff.sum(axis=0)
            reduced_weight_gradient = weight_gradient[:-1] - weight_gradient[-1]
            reduced_intercept_gradient = intercept_gradient[:-1] - intercept_gradient[-1]
            gradient = np.concatenate([reduced_weight_gradient.ravel(), reduced_intercept_gradient])
            if not np.isfinite(loss) or not np.isfinite(gradient).all():
                raise ValueError("READER_OBJECTIVE_NONFINITE")
            return float(loss), np.asarray(gradient, dtype=np.float64)

        def callback(theta: np.ndarray) -> None:
            history.append(objective(theta)[0])

        optimizer_scale = 1.0e12

        def optimizer_objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = objective(theta)
            return value * optimizer_scale, gradient * optimizer_scale

        initial = np.zeros((classes - 1) * columns + (classes - 1), dtype=np.float64)
        initial_loss, _ = objective(initial)
        history.append(initial_loss)
        theta = initial
        total_iterations = 0
        total_evaluations = 0
        result = None
        # L-BFGS can report machine-precision objective stagnation a few
        # iterations before the separately frozen gradient gate.  Deterministic
        # restarts from the last iterate preserve the optimizer, objective and
        # zero initialization while allowing the stricter gate to decide.
        while total_iterations < max_iterations:
            current_value, current_gradient = objective(theta)
            current_norm = float(np.max(np.abs(current_gradient)))
            if current_norm <= gradient_tolerance:
                break
            if current_norm <= 1.0e-7:
                direction = -current_gradient
                step = 1.0
                refined = False
                for _ in range(80):
                    candidate = theta + step * direction
                    candidate_value, candidate_gradient = objective(candidate)
                    if np.max(np.abs(candidate_gradient)) < current_norm:
                        theta = candidate
                        history.append(candidate_value)
                        refined = True
                        break
                    step *= 0.5
                total_iterations += 1
                if not refined:
                    break
                continue
            phase_limit = min(50, max_iterations - total_iterations)
            result = minimize(
                optimizer_objective,
                theta,
                method="L-BFGS-B",
                jac=True,
                callback=callback,
                options={
                    "maxiter": phase_limit,
                    "gtol": gradient_tolerance * optimizer_scale,
                    "ftol": 0.0,
                    "maxls": 200,
                    "maxcor": 10,
                },
            )
            total_iterations += int(result.nit)
            total_evaluations += int(result.nfev)
            theta = result.x
            _, restart_gradient = objective(theta)
            restart_norm = float(np.max(np.abs(restart_gradient)))
            if restart_norm <= gradient_tolerance:
                break
            if total_iterations >= max_iterations:
                break
            # A memory reset in L-BFGS starts from the steepest-descent
            # direction.  Perform that safeguarded restart step explicitly so
            # machine-precision line-search stagnation cannot burn the entire
            # frozen iteration budget without changing the iterate.
            base_value, base_gradient = objective(theta)
            direction = -base_gradient
            directional_derivative = float(np.dot(base_gradient, direction))
            step = 1.0
            accepted = False
            for _ in range(80):
                candidate = theta + step * direction
                candidate_value, candidate_gradient = objective(candidate)
                gradient_only_precision_refinement = (
                    restart_norm <= 1.0e-7
                    and np.max(np.abs(candidate_gradient)) < np.max(np.abs(base_gradient))
                )
                gradient_decreased_at_precision_floor = (
                    candidate_value == base_value
                    and np.max(np.abs(candidate_gradient)) < np.max(np.abs(base_gradient))
                )
                if (
                    candidate_value <= base_value + 1.0e-4 * step * directional_derivative
                    or gradient_decreased_at_precision_floor
                    or gradient_only_precision_refinement
                ):
                    theta = candidate
                    accepted = True
                    break
                step *= 0.5
            total_iterations += 1
            if not accepted:
                break
        if result is None:
            class Result:
                success = True
                message = "CONVERGED_BY_DETERMINISTIC_LBFGS_PRECISION_REFINEMENT"
            result = Result()
        result.x = theta
        final_loss, final_reduced_gradient = objective(theta)
        final_weights, final_intercepts = unpack(theta)
        final_logits = x.astype(np.longdouble) @ final_weights.astype(np.longdouble).T + final_intercepts.astype(np.longdouble)
        final_probabilities = _softmax(final_logits)
        final_diff = final_probabilities
        final_diff[np.arange(rows), y] -= 1.0
        final_diff /= rows
        full_weight_gradient = final_diff.T @ x.astype(np.longdouble) + l2_lambda * final_weights
        full_intercept_gradient = final_diff.sum(axis=0)
        final_gradient = np.asarray(np.concatenate([full_weight_gradient.ravel(), full_intercept_gradient]), dtype=np.float64)
        if not history or history[-1] != final_loss:
            history.append(final_loss)
        relative_change = (
            abs(history[-2] - history[-1]) / max(1.0, abs(history[-2]))
            if len(history) > 1
            # No optimizer step is required when the all-zero initialization
            # already satisfies the gradient branch.  Its deterministic
            # objective change is exactly zero, not an unavailable infinity.
            else 0.0
        )
        gradient_infinity_norm = float(np.max(np.abs(final_gradient)))
        finite_outputs = bool(
            np.isfinite(final_loss)
            and np.isfinite(final_gradient).all()
            and np.isfinite(final_weights).all()
            and np.isfinite(final_intercepts).all()
            and np.isfinite(final_probabilities).all()
        )
        converged = convergence_contract(
            optimizer_success=bool(result.success),
            iterations=total_iterations,
            iterations_max=max_iterations,
            gradient_infinity_norm=gradient_infinity_norm,
            gradient_infinity_norm_max=gradient_tolerance,
            absolute_gradient_guard_max=absolute_gradient_guard,
            objective_relative_change=relative_change,
            objective_relative_change_max=relative_objective_tolerance,
            finite_outputs=finite_outputs,
        )
        if not converged:
            raise RuntimeError(
                "PRIMARY_READER_NONCONVERGENCE:"
                f"success={result.success}:iterations={total_iterations}:"
                f"gradient_inf={gradient_infinity_norm:.17g}:relative_change={relative_change:.17g}:"
                f"message={result.message}"
            )
        weights, intercepts = unpack(result.x)
        return cls(
            standardizer=standardizer,
            coefficients=weights,
            intercepts=intercepts,
            convergence={
                "optimizer": "scipy_L-BFGS-B_deterministic_full_batch",
                "identifiability_parameterization": "exact_zero_sum_rows_and_intercepts",
                "uniform_objective_scale_for_numerical_resolution": optimizer_scale,
                "iterations": total_iterations,
                "function_evaluations": total_evaluations,
                "final_objective": final_loss,
                "gradient_infinity_norm": gradient_infinity_norm,
                "objective_relative_change": relative_change,
                "gradient_infinity_norm_max": gradient_tolerance,
                "absolute_gradient_guard_max": absolute_gradient_guard,
                "objective_relative_change_max": relative_objective_tolerance,
                "iterations_max": max_iterations,
                "optimizer_success_required": True,
                "finite_outputs_required": True,
                "stopping_logic": "optimizer_success AND finite_outputs AND absolute_gradient_guard AND (gradient_branch OR objective_relative_change_branch)",
                "gradient_branch_satisfied": gradient_infinity_norm <= gradient_tolerance,
                "objective_relative_change_branch_satisfied": relative_change <= relative_objective_tolerance,
                "status": "PASS",
            },
        )

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        x = self.standardizer.transform(features)
        return _softmax(x @ self.coefficients.T + self.intercepts)

    def predict(self, features: np.ndarray) -> list[str]:
        probabilities = self.probabilities(features)
        return [CLASS_ORDER[int(index)] for index in np.argmax(probabilities, axis=1)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": "COMMON_FIXED_MULTINOMIAL_LINEAR_READER",
            "class_order": list(CLASS_ORDER),
            "L2_lambda": 1.0,
            "intercept_penalty": 0.0,
            "initialization": "all_zero",
            "tie_break": "first_class_in_fixed_order",
            "standardizer": self.standardizer.as_dict(),
            "coefficients": [[float(value) for value in row] for row in self.coefficients],
            "intercepts": [float(value) for value in self.intercepts],
            "convergence": self.convergence,
        }
