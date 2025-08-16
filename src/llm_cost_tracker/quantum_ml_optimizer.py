"""
Quantum-Inspired Machine Learning Optimizer

This module provides ML-enhanced optimization for the quantum task planner:
- Reinforcement learning for task scheduling optimization
- Neural network-based execution time prediction
- Genetic algorithm with quantum operators
- Bayesian optimization for hyperparameter tuning
"""

import asyncio
import json
import logging
import math
import random
import statistics
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """State representation for ML optimization."""

    task_features: (
        np.ndarray
    )  # Task characteristics (priority, resources, dependencies)
    system_features: np.ndarray  # System state (load, resources available, time)
    schedule_features: np.ndarray  # Current schedule characteristics

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    state_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "task_features": self.task_features.tolist(),
            "system_features": self.system_features.tolist(),
            "schedule_features": self.schedule_features.tolist(),
            "timestamp": self.timestamp.isoformat(),
            "state_id": self.state_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationState":
        """Create state from dictionary."""
        return cls(
            task_features=np.array(data["task_features"]),
            system_features=np.array(data["system_features"]),
            schedule_features=np.array(data["schedule_features"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_id=data.get("state_id", ""),
        )


@dataclass
class OptimizationAction:
    """Action representation for ML optimization."""

    action_type: str  # "swap", "move", "reorder", "parallel"
    parameters: Dict[str, Any]  # Action-specific parameters
    expected_reward: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "expected_reward": self.expected_reward,
            "confidence": self.confidence,
        }


class QuantumReinforcementLearner:
    """
    Reinforcement learning agent for quantum task scheduling optimization.
    Uses Q-learning with quantum-inspired enhancements.
    """

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 20,
        learning_rate: float = 0.001,
        epsilon: float = 0.1,
        gamma: float = 0.95,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor

        # Q-table approximation using simple neural network
        self.q_network = self._create_q_network()
        self.target_network = self._create_q_network()

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32

        # Quantum-inspired enhancements
        self.superposition_factor = 0.2  # Amount of quantum superposition in actions
        self.entanglement_memory = deque(maxlen=100)  # Memory of correlated actions

        # Training metrics
        self.training_history = []
        self.total_steps = 0
        self.total_episodes = 0

    def _create_q_network(self) -> Dict[str, np.ndarray]:
        """Create simple neural network representation."""
        # Simple 2-layer network with weights stored as numpy arrays
        return {
            "w1": np.random.normal(0, 0.1, (self.state_dim, 64)),
            "b1": np.zeros(64),
            "w2": np.random.normal(0, 0.1, (64, 32)),
            "b2": np.zeros(32),
            "w3": np.random.normal(0, 0.1, (32, self.action_dim)),
            "b3": np.zeros(self.action_dim),
        }

    def _forward_pass(
        self, state: np.ndarray, network: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Forward pass through the network."""
        # Layer 1
        z1 = np.dot(state, network["w1"]) + network["b1"]
        a1 = np.tanh(z1)  # Activation

        # Layer 2
        z2 = np.dot(a1, network["w2"]) + network["b2"]
        a2 = np.tanh(z2)

        # Output layer
        q_values = np.dot(a2, network["w3"]) + network["b3"]

        return q_values

    def predict_q_values(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a given state."""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        return self._forward_pass(state, self.q_network)

    def select_action(
        self, state: np.ndarray, available_actions: List[OptimizationAction]
    ) -> OptimizationAction:
        """Select action using epsilon-greedy with quantum superposition."""
        self.total_steps += 1

        # Get Q-values for current state
        q_values = self.predict_q_values(state).flatten()

        # Quantum superposition: consider multiple actions simultaneously
        if random.random() < self.superposition_factor:
            # Create superposition of top actions
            top_k = min(3, len(available_actions))
            top_indices = np.argsort(q_values)[-top_k:]

            # Weight actions by their Q-values (quantum amplitudes)
            weights = np.exp(q_values[top_indices] / np.max(q_values[top_indices]))
            weights = weights / np.sum(weights)

            # Select action based on quantum probability
            selected_idx = np.random.choice(top_indices, p=weights)
        else:
            # Standard epsilon-greedy
            if random.random() < self.epsilon:
                selected_idx = random.randint(
                    0, min(len(available_actions) - 1, self.action_dim - 1)
                )
            else:
                selected_idx = np.argmax(q_values[: len(available_actions)])

        # Ensure we don't go out of bounds
        selected_idx = min(selected_idx, len(available_actions) - 1)

        selected_action = available_actions[selected_idx]
        selected_action.expected_reward = q_values[selected_idx]
        selected_action.confidence = 1.0 - self.epsilon

        return selected_action

    def store_experience(
        self,
        state: np.ndarray,
        action: OptimizationAction,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer."""
        experience = {
            "state": state.copy(),
            "action": action.to_dict(),
            "reward": reward,
            "next_state": next_state.copy(),
            "done": done,
        }

        self.replay_buffer.append(experience)

        # Store entanglement memory for quantum correlation
        self.entanglement_memory.append(
            {
                "action_type": action.action_type,
                "reward": reward,
                "state_hash": hash(state.tobytes()),
            }
        )

    def train_step(self):
        """Perform one training step using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        states = np.array([exp["state"] for exp in batch])
        rewards = np.array([exp["reward"] for exp in batch])
        next_states = np.array([exp["next_state"] for exp in batch])
        dones = np.array([exp["done"] for exp in batch])

        # Get current Q-values
        current_q = self._forward_pass(states, self.q_network)

        # Get next Q-values from target network
        next_q = self._forward_pass(next_states, self.target_network)

        # Calculate target Q-values
        target_q = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)

        # Simple gradient descent update (simplified)
        error = target_q.reshape(-1, 1) - current_q

        # Update weights (simplified backpropagation)
        self._update_weights(states, error)

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.999)

        # Record training metrics
        avg_loss = np.mean(np.square(error))
        self.training_history.append(
            {
                "step": self.total_steps,
                "loss": avg_loss,
                "epsilon": self.epsilon,
                "replay_buffer_size": len(self.replay_buffer),
            }
        )

        # Keep only recent history
        if len(self.training_history) > 1000:
            self.training_history = self.training_history[-1000:]

    def _update_weights(self, states: np.ndarray, error: np.ndarray):
        """Simple weight update (simplified for demonstration)."""
        # This is a very simplified version of backpropagation
        # In practice, you'd use proper gradient computation

        avg_error = np.mean(error, axis=0)

        # Update output layer
        self.q_network["w3"] += (
            self.learning_rate * np.outer(avg_error, avg_error) * 0.01
        )
        self.q_network["b3"] += self.learning_rate * avg_error * 0.1

    def update_target_network(self):
        """Update target network (soft update)."""
        tau = 0.005  # Soft update parameter

        for key in self.q_network:
            self.target_network[key] = (
                tau * self.q_network[key] + (1 - tau) * self.target_network[key]
            )

    def get_quantum_correlations(self) -> Dict[str, float]:
        """Analyze quantum correlations in action history."""
        correlations = defaultdict(list)

        for entry in self.entanglement_memory:
            correlations[entry["action_type"]].append(entry["reward"])

        # Calculate correlation strengths
        result = {}
        for action_type, rewards in correlations.items():
            if len(rewards) > 1:
                result[action_type] = {
                    "avg_reward": np.mean(rewards),
                    "reward_variance": np.var(rewards),
                    "count": len(rewards),
                }

        return result

    def save_model(self, filepath: str):
        """Save the model to file."""
        model_data = {
            "q_network": {k: v.tolist() for k, v in self.q_network.items()},
            "target_network": {k: v.tolist() for k, v in self.target_network.items()},
            "hyperparameters": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "learning_rate": self.learning_rate,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
            },
            "training_stats": {
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
                "training_history": self.training_history[-100:],  # Save recent history
            },
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        try:
            with open(filepath, "r") as f:
                model_data = json.load(f)

            # Restore networks
            self.q_network = {
                k: np.array(v) for k, v in model_data["q_network"].items()
            }
            self.target_network = {
                k: np.array(v) for k, v in model_data["target_network"].items()
            }

            # Restore hyperparameters
            params = model_data["hyperparameters"]
            self.state_dim = params["state_dim"]
            self.action_dim = params["action_dim"]
            self.learning_rate = params["learning_rate"]
            self.epsilon = params["epsilon"]
            self.gamma = params["gamma"]

            # Restore training stats
            stats = model_data["training_stats"]
            self.total_steps = stats["total_steps"]
            self.total_episodes = stats["total_episodes"]
            self.training_history = stats["training_history"]

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")


class QuantumNeuralPredictor:
    """
    Neural network for predicting task execution times and success probabilities.
    Uses quantum-inspired features for enhanced prediction accuracy.
    """

    def __init__(self, input_dim: int = 20):
        self.input_dim = input_dim
        self.hidden_dim = 64
        self.output_dim = 2  # [execution_time, success_probability]

        # Initialize network weights
        self.weights = {
            "w1": np.random.normal(0, 0.1, (input_dim, self.hidden_dim)),
            "b1": np.zeros(self.hidden_dim),
            "w2": np.random.normal(0, 0.1, (self.hidden_dim, 32)),
            "b2": np.zeros(32),
            "w3": np.random.normal(0, 0.1, (32, self.output_dim)),
            "b3": np.zeros(self.output_dim),
        }

        # Training data
        self.training_data = []
        self.prediction_cache = {}

        # Quantum features
        self.quantum_features_enabled = True
        self.uncertainty_estimation = True

    def extract_features(
        self, task_data: Dict[str, Any], system_state: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features for prediction."""
        features = []

        # Task features
        features.append(task_data.get("priority", 5.0) / 10.0)  # Normalized priority
        features.append(
            len(task_data.get("dependencies", [])) / 10.0
        )  # Normalized dependency count
        features.append(task_data.get("estimated_duration", 0) / 3600.0)  # Hours

        # Resource features
        resources = task_data.get("required_resources", {})
        features.append(resources.get("cpu_cores", 0) / 8.0)  # Normalized CPU
        features.append(resources.get("memory_gb", 0) / 16.0)  # Normalized memory
        features.append(resources.get("storage_gb", 0) / 100.0)  # Normalized storage

        # System state features
        features.append(system_state.get("cpu_utilization", 0.5))
        features.append(system_state.get("memory_utilization", 0.5))
        features.append(system_state.get("active_tasks", 0) / 20.0)  # Normalized

        # Temporal features
        current_hour = datetime.now().hour
        features.append(np.sin(2 * np.pi * current_hour / 24))  # Hour as sine
        features.append(np.cos(2 * np.pi * current_hour / 24))  # Hour as cosine

        # Quantum-inspired features
        if self.quantum_features_enabled:
            # Quantum superposition feature (task priority uncertainty)
            priority_uncertainty = 1.0 - abs(task_data.get("priority", 5.0) - 5.0) / 5.0
            features.append(priority_uncertainty)

            # Quantum entanglement feature (dependency correlation)
            entanglement_factor = len(task_data.get("entangled_tasks", [])) / 10.0
            features.append(entanglement_factor)

            # Quantum interference (resource conflicts)
            interference = len(task_data.get("interference_pattern", {})) / 10.0
            features.append(interference)

        # Pad or truncate to input_dim
        features = features[: self.input_dim]
        while len(features) < self.input_dim:
            features.append(0.0)

        return np.array(features)

    def forward_pass(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through network with uncertainty estimation."""
        # Layer 1
        z1 = np.dot(features, self.weights["w1"]) + self.weights["b1"]
        a1 = np.tanh(z1)

        # Layer 2
        z2 = np.dot(a1, self.weights["w2"]) + self.weights["b2"]
        a2 = np.tanh(z2)

        # Output layer
        output = np.dot(a2, self.weights["w3"]) + self.weights["b3"]

        # Apply output transformations
        execution_time = np.exp(output[0])  # Ensure positive execution time
        success_prob = 1.0 / (1.0 + np.exp(-output[1]))  # Sigmoid for probability

        predictions = np.array([execution_time, success_prob])

        # Uncertainty estimation using ensemble approach
        uncertainties = np.zeros(2)
        if self.uncertainty_estimation:
            # Simple dropout-like uncertainty estimation
            num_samples = 10
            samples = []

            for _ in range(num_samples):
                # Add noise to simulate dropout
                noise_scale = 0.1
                noisy_a1 = a1 + np.random.normal(0, noise_scale, a1.shape)
                noisy_a2 = np.tanh(
                    np.dot(noisy_a1, self.weights["w2"]) + self.weights["b2"]
                )
                noisy_output = np.dot(noisy_a2, self.weights["w3"]) + self.weights["b3"]

                noisy_exec_time = np.exp(noisy_output[0])
                noisy_success_prob = 1.0 / (1.0 + np.exp(-noisy_output[1]))

                samples.append([noisy_exec_time, noisy_success_prob])

            samples = np.array(samples)
            uncertainties = np.std(samples, axis=0)

        return predictions, uncertainties

    def predict(
        self, task_data: Dict[str, Any], system_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict execution time and success probability."""
        # Generate cache key
        cache_key = hash(
            json.dumps(task_data, sort_keys=True)
            + json.dumps(system_state, sort_keys=True)
        )

        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        features = self.extract_features(task_data, system_state)
        predictions, uncertainties = self.forward_pass(features)

        result = {
            "predicted_execution_time": float(predictions[0]),
            "predicted_success_probability": float(predictions[1]),
            "execution_time_uncertainty": float(uncertainties[0]),
            "success_probability_uncertainty": float(uncertainties[1]),
        }

        # Cache result
        self.prediction_cache[cache_key] = result

        # Limit cache size
        if len(self.prediction_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.prediction_cache.keys())[:100]
            for key in oldest_keys:
                del self.prediction_cache[key]

        return result

    def add_training_data(
        self,
        task_data: Dict[str, Any],
        system_state: Dict[str, Any],
        actual_execution_time: float,
        actual_success: bool,
    ):
        """Add training data point."""
        features = self.extract_features(task_data, system_state)
        targets = np.array([actual_execution_time, 1.0 if actual_success else 0.0])

        self.training_data.append(
            {
                "features": features,
                "targets": targets,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Limit training data size
        if len(self.training_data) > 5000:
            self.training_data = self.training_data[-5000:]

    def train(self, epochs: int = 100, learning_rate: float = 0.001):
        """Train the prediction model."""
        if len(self.training_data) < 10:
            logger.warning("Not enough training data for neural network training")
            return

        logger.info(
            f"Training neural predictor with {len(self.training_data)} samples for {epochs} epochs"
        )

        # Prepare training data
        X = np.array([entry["features"] for entry in self.training_data])
        y = np.array([entry["targets"] for entry in self.training_data])

        # Simple training loop with gradient descent
        for epoch in range(epochs):
            # Forward pass
            total_loss = 0.0

            for i in range(len(X)):
                features = X[i]
                targets = y[i]

                predictions, _ = self.forward_pass(features)
                loss = np.mean((predictions - targets) ** 2)
                total_loss += loss

                # Simple gradient update (very simplified)
                error = predictions - targets
                self._update_weights(features, error, learning_rate)

            if epoch % 20 == 0:
                avg_loss = total_loss / len(X)
                logger.debug(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

        logger.info("Neural predictor training completed")

    def _update_weights(self, features: np.ndarray, error: np.ndarray, lr: float):
        """Simple weight update (simplified backpropagation)."""
        # This is a very simplified version - in practice you'd compute proper gradients
        self.weights["b3"] -= lr * error * 0.1

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_data:
            return {"message": "No training data available"}

        execution_times = [entry["targets"][0] for entry in self.training_data]
        success_rates = [entry["targets"][1] for entry in self.training_data]

        return {
            "total_samples": len(self.training_data),
            "avg_execution_time": np.mean(execution_times),
            "execution_time_std": np.std(execution_times),
            "success_rate": np.mean(success_rates),
            "cache_size": len(self.prediction_cache),
            "quantum_features_enabled": self.quantum_features_enabled,
        }


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning of the quantum task planner.
    """

    def __init__(self, parameter_space: Dict[str, Tuple[float, float]]):
        self.parameter_space = parameter_space
        self.parameter_names = list(parameter_space.keys())
        self.bounds = np.array([parameter_space[name] for name in self.parameter_names])

        # Observations
        self.X_observed = []  # Parameter configurations
        self.y_observed = []  # Objective values

        # Gaussian Process (simplified)
        self.gp_mean = 0.0
        self.gp_variance = 1.0
        self.noise_variance = 0.01

        # Acquisition function parameters
        self.exploration_weight = 0.1
        self.best_observed = None

    def suggest_parameters(self) -> Dict[str, float]:
        """Suggest next parameters to evaluate using acquisition function."""
        if len(self.X_observed) < 2:
            # Random exploration for first few points
            params = {}
            for name, (min_val, max_val) in self.parameter_space.items():
                params[name] = random.uniform(min_val, max_val)
            return params

        # Find parameters that maximize acquisition function
        best_params = None
        best_acquisition = -float("inf")

        # Simple random search for acquisition function maximum
        for _ in range(1000):
            candidate_params = {}
            candidate_array = []

            for name, (min_val, max_val) in self.parameter_space.items():
                value = random.uniform(min_val, max_val)
                candidate_params[name] = value
                candidate_array.append(value)

            acquisition_value = self._acquisition_function(np.array(candidate_array))

            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_params = candidate_params

        return best_params

    def _acquisition_function(self, x: np.ndarray) -> float:
        """Upper Confidence Bound acquisition function."""
        # Simplified GP prediction
        mean, variance = self._gp_predict(x)

        # UCB acquisition
        std_dev = np.sqrt(variance)
        ucb = mean + self.exploration_weight * std_dev

        return ucb

    def _gp_predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Simplified Gaussian Process prediction."""
        if not self.X_observed:
            return self.gp_mean, self.gp_variance

        # Very simplified GP - just use nearest neighbor
        X_array = np.array(self.X_observed)
        distances = np.sum((X_array - x) ** 2, axis=1)
        nearest_idx = np.argmin(distances)

        # Simple kernel-based prediction
        nearest_y = self.y_observed[nearest_idx]
        distance_weight = np.exp(-distances[nearest_idx])

        predicted_mean = (
            distance_weight * nearest_y + (1 - distance_weight) * self.gp_mean
        )
        predicted_variance = self.gp_variance * (1 - distance_weight)

        return predicted_mean, predicted_variance

    def update(self, parameters: Dict[str, float], objective_value: float):
        """Update the optimizer with new observation."""
        param_array = [parameters[name] for name in self.parameter_names]

        self.X_observed.append(param_array)
        self.y_observed.append(objective_value)

        # Update best observed
        if self.best_observed is None or objective_value > self.best_observed:
            self.best_observed = objective_value

        # Update GP hyperparameters (simplified)
        if len(self.y_observed) > 1:
            self.gp_mean = np.mean(self.y_observed)
            self.gp_variance = np.var(self.y_observed)

        logger.debug(
            f"Bayesian optimizer updated with objective value: {objective_value:.4f}"
        )

    def get_best_parameters(self) -> Optional[Dict[str, float]]:
        """Get best parameters found so far."""
        if not self.y_observed:
            return None

        best_idx = np.argmax(self.y_observed)
        best_param_array = self.X_observed[best_idx]

        return {
            name: best_param_array[i] for i, name in enumerate(self.parameter_names)
        }

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        history = []

        for i, (params, obj_val) in enumerate(zip(self.X_observed, self.y_observed)):
            param_dict = {
                name: params[j] for j, name in enumerate(self.parameter_names)
            }
            history.append(
                {"iteration": i, "parameters": param_dict, "objective_value": obj_val}
            )

        return history


class QuantumMLOptimizer:
    """
    Main ML optimization system that integrates all ML components
    with the quantum task planner.
    """

    def __init__(self):
        self.rl_agent = QuantumReinforcementLearner()
        self.neural_predictor = QuantumNeuralPredictor()
        self.bayesian_optimizer = None

        # Optimization state
        self.optimization_enabled = True
        self.learning_enabled = True
        self.prediction_enabled = True

        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = deque(maxlen=1000)

        # Configuration
        self.training_frequency = 10  # Train every N episodes
        self.episode_count = 0

    def initialize_bayesian_optimization(
        self, parameter_space: Dict[str, Tuple[float, float]]
    ):
        """Initialize Bayesian optimization for hyperparameter tuning."""
        self.bayesian_optimizer = BayesianOptimizer(parameter_space)
        logger.info("Bayesian optimizer initialized")

    def get_schedule_optimization_actions(
        self,
        current_schedule: List[str],
        task_data: Dict[str, Any],
        system_state: Dict[str, Any],
    ) -> List[OptimizationAction]:
        """Generate possible optimization actions for current schedule."""
        actions = []

        if len(current_schedule) < 2:
            return actions

        # Generate swap actions
        for i in range(len(current_schedule)):
            for j in range(i + 1, len(current_schedule)):
                actions.append(
                    OptimizationAction(
                        action_type="swap", parameters={"pos1": i, "pos2": j}
                    )
                )

        # Generate move actions
        for i in range(len(current_schedule)):
            for new_pos in range(len(current_schedule)):
                if new_pos != i:
                    actions.append(
                        OptimizationAction(
                            action_type="move",
                            parameters={"from_pos": i, "to_pos": new_pos},
                        )
                    )

        # Generate reorder actions (for subsequences)
        if len(current_schedule) >= 3:
            actions.append(
                OptimizationAction(
                    action_type="reorder",
                    parameters={"start": 0, "end": min(3, len(current_schedule))},
                )
            )

        # Limit number of actions to prevent explosion
        return actions[:50]  # Limit to 50 actions

    def optimize_schedule(
        self,
        current_schedule: List[str],
        task_data: Dict[str, Any],
        system_state: Dict[str, Any],
    ) -> List[str]:
        """Optimize schedule using ML techniques."""
        if not self.optimization_enabled:
            return current_schedule

        # Extract state features
        state_features = self._extract_state_features(
            current_schedule, task_data, system_state
        )

        # Get possible actions
        available_actions = self.get_schedule_optimization_actions(
            current_schedule, task_data, system_state
        )

        if not available_actions:
            return current_schedule

        # Select action using RL agent
        selected_action = self.rl_agent.select_action(state_features, available_actions)

        # Apply action to schedule
        optimized_schedule = self._apply_action(current_schedule, selected_action)

        # Predict performance improvement
        if self.prediction_enabled:
            current_prediction = self._predict_schedule_performance(
                current_schedule, task_data, system_state
            )
            optimized_prediction = self._predict_schedule_performance(
                optimized_schedule, task_data, system_state
            )

            # Store prediction for later comparison with actual results
            self.performance_metrics.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": selected_action.to_dict(),
                    "predicted_improvement": optimized_prediction["total_score"]
                    - current_prediction["total_score"],
                    "current_schedule_score": current_prediction["total_score"],
                    "optimized_schedule_score": optimized_prediction["total_score"],
                }
            )

        return optimized_schedule

    def _extract_state_features(
        self,
        schedule: List[str],
        task_data: Dict[str, Any],
        system_state: Dict[str, Any],
    ) -> np.ndarray:
        """Extract state features for ML models."""
        features = []

        # Schedule features
        features.append(len(schedule) / 20.0)  # Normalized schedule length

        # Task complexity features
        if task_data:
            avg_priority = np.mean(
                [task_data.get(tid, {}).get("priority", 5.0) for tid in schedule]
            )
            features.append(avg_priority / 10.0)

            total_dependencies = sum(
                len(task_data.get(tid, {}).get("dependencies", [])) for tid in schedule
            )
            features.append(total_dependencies / 50.0)  # Normalized
        else:
            features.extend([0.5, 0.0])

        # System state features
        features.append(system_state.get("cpu_utilization", 0.5))
        features.append(system_state.get("memory_utilization", 0.5))
        features.append(system_state.get("active_tasks", 0) / 20.0)

        # Temporal features
        current_time = datetime.now()
        features.append(current_time.hour / 24.0)
        features.append(current_time.weekday() / 7.0)

        # Pad to fixed size (50 dimensions)
        while len(features) < 50:
            features.append(0.0)

        return np.array(features[:50])

    def _apply_action(
        self, schedule: List[str], action: OptimizationAction
    ) -> List[str]:
        """Apply optimization action to schedule."""
        new_schedule = schedule.copy()

        try:
            if action.action_type == "swap":
                pos1 = action.parameters["pos1"]
                pos2 = action.parameters["pos2"]
                if 0 <= pos1 < len(new_schedule) and 0 <= pos2 < len(new_schedule):
                    new_schedule[pos1], new_schedule[pos2] = (
                        new_schedule[pos2],
                        new_schedule[pos1],
                    )

            elif action.action_type == "move":
                from_pos = action.parameters["from_pos"]
                to_pos = action.parameters["to_pos"]
                if 0 <= from_pos < len(new_schedule) and 0 <= to_pos < len(
                    new_schedule
                ):
                    task = new_schedule.pop(from_pos)
                    new_schedule.insert(to_pos, task)

            elif action.action_type == "reorder":
                start = action.parameters.get("start", 0)
                end = action.parameters.get("end", len(new_schedule))
                if 0 <= start < end <= len(new_schedule):
                    subsequence = new_schedule[start:end]
                    random.shuffle(subsequence)
                    new_schedule[start:end] = subsequence

        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Failed to apply action {action.action_type}: {e}")
            return schedule

        return new_schedule

    def _predict_schedule_performance(
        self,
        schedule: List[str],
        task_data: Dict[str, Any],
        system_state: Dict[str, Any],
    ) -> Dict[str, float]:
        """Predict performance of a schedule using neural predictor."""
        if not self.prediction_enabled:
            return {"total_score": 0.0}

        total_score = 0.0
        total_execution_time = 0.0
        total_success_probability = 0.0

        for task_id in schedule:
            if task_id in task_data:
                task_info = task_data[task_id]
                prediction = self.neural_predictor.predict(task_info, system_state)

                # Calculate score based on prediction
                exec_time_score = 1.0 / (
                    1.0 + prediction["predicted_execution_time"] / 3600.0
                )  # Favor shorter times
                success_score = prediction["predicted_success_probability"]

                task_score = exec_time_score * success_score
                total_score += task_score

                total_execution_time += prediction["predicted_execution_time"]
                total_success_probability += prediction["predicted_success_probability"]

        if len(schedule) > 0:
            total_success_probability /= len(schedule)

        return {
            "total_score": total_score,
            "avg_execution_time": total_execution_time / max(len(schedule), 1),
            "avg_success_probability": total_success_probability,
        }

    def update_performance(
        self,
        schedule: List[str],
        actual_results: Dict[str, Any],
        system_state: Dict[str, Any],
    ):
        """Update ML models with actual performance results."""
        if not self.learning_enabled:
            return

        self.episode_count += 1

        # Calculate actual performance metrics
        total_execution_time = actual_results.get("total_execution_time", 0.0)
        success_rate = actual_results.get("success_rate", 0.0)
        total_tasks = actual_results.get("total_tasks", 1)

        # Update neural predictor with actual results
        if "task_results" in actual_results:
            for task_id, task_result in actual_results["task_results"].items():
                if task_id in actual_results.get("task_data", {}):
                    task_info = actual_results["task_data"][task_id]
                    actual_exec_time = task_result.get("execution_time", 0.0)
                    actual_success = task_result.get("success", False)

                    self.neural_predictor.add_training_data(
                        task_info, system_state, actual_exec_time, actual_success
                    )

        # Calculate reward for RL agent
        reward = self._calculate_reward(actual_results)

        # Store experience in RL agent (simplified)
        if len(self.optimization_history) > 0:
            last_state = self.optimization_history[-1].get("state_features")
            last_action = self.optimization_history[-1].get("action")
            current_state = self._extract_state_features(schedule, {}, system_state)

            if last_state is not None and last_action is not None:
                self.rl_agent.store_experience(
                    last_state, last_action, reward, current_state, done=True
                )

        # Train models periodically
        if self.episode_count % self.training_frequency == 0:
            self._train_models()

        # Record optimization history
        self.optimization_history.append(
            {
                "episode": self.episode_count,
                "timestamp": datetime.now().isoformat(),
                "schedule_length": len(schedule),
                "total_execution_time": total_execution_time,
                "success_rate": success_rate,
                "reward": reward,
                "system_state": system_state.copy(),
            }
        )

        # Limit history size
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]

    def _calculate_reward(self, actual_results: Dict[str, Any]) -> float:
        """Calculate reward signal for RL agent."""
        # Multi-objective reward function
        success_rate = actual_results.get("success_rate", 0.0)
        total_time = actual_results.get("total_execution_time", float("inf"))
        total_tasks = actual_results.get("total_tasks", 1)

        # Success rate component (0 to 1)
        success_reward = success_rate

        # Efficiency component (higher is better, normalized)
        time_per_task = total_time / max(total_tasks, 1)
        efficiency_reward = 1.0 / (1.0 + time_per_task / 3600.0)  # Normalize by 1 hour

        # Combined reward
        total_reward = 0.7 * success_reward + 0.3 * efficiency_reward

        return total_reward

    def _train_models(self):
        """Train ML models with collected data."""
        logger.info("Training ML models...")

        # Train RL agent
        for _ in range(10):  # Multiple training steps
            self.rl_agent.train_step()

        # Update target network
        self.rl_agent.update_target_network()

        # Train neural predictor
        self.neural_predictor.train(epochs=50, learning_rate=0.001)

        logger.info("ML model training completed")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        rl_correlations = self.rl_agent.get_quantum_correlations()
        predictor_stats = self.neural_predictor.get_training_stats()

        # Calculate recent performance trends
        recent_episodes = (
            self.optimization_history[-50:]
            if len(self.optimization_history) >= 50
            else self.optimization_history
        )

        recent_rewards = [ep["reward"] for ep in recent_episodes if "reward" in ep]
        recent_success_rates = [
            ep["success_rate"] for ep in recent_episodes if "success_rate" in ep
        ]

        stats = {
            "reinforcement_learning": {
                "total_episodes": self.rl_agent.total_episodes,
                "total_steps": self.rl_agent.total_steps,
                "current_epsilon": self.rl_agent.epsilon,
                "replay_buffer_size": len(self.rl_agent.replay_buffer),
                "quantum_correlations": rl_correlations,
                "recent_avg_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
            },
            "neural_prediction": predictor_stats,
            "optimization_performance": {
                "total_optimization_episodes": self.episode_count,
                "recent_avg_success_rate": (
                    np.mean(recent_success_rates) if recent_success_rates else 0.0
                ),
                "performance_metrics_count": len(self.performance_metrics),
                "optimization_enabled": self.optimization_enabled,
                "learning_enabled": self.learning_enabled,
            },
        }

        if self.bayesian_optimizer:
            bayesian_stats = {
                "total_evaluations": len(self.bayesian_optimizer.y_observed),
                "best_objective_value": self.bayesian_optimizer.best_observed,
                "best_parameters": self.bayesian_optimizer.get_best_parameters(),
            }
            stats["bayesian_optimization"] = bayesian_stats

        return stats

    def save_models(self, base_path: str):
        """Save all ML models."""
        import os

        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # Save RL agent
        rl_path = os.path.join(base_path, "rl_agent.json")
        self.rl_agent.save_model(rl_path)

        # Save optimization history
        history_path = os.path.join(base_path, "optimization_history.json")
        with open(history_path, "w") as f:
            json.dump(self.optimization_history, f, indent=2)

        logger.info(f"ML models saved to {base_path}")

    def load_models(self, base_path: str):
        """Load ML models from files."""
        import os

        try:
            # Load RL agent
            rl_path = os.path.join(base_path, "rl_agent.json")
            if os.path.exists(rl_path):
                self.rl_agent.load_model(rl_path)

            # Load optimization history
            history_path = os.path.join(base_path, "optimization_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    self.optimization_history = json.load(f)

            logger.info(f"ML models loaded from {base_path}")

        except Exception as e:
            logger.error(f"Failed to load ML models from {base_path}: {e}")


# Demo and testing functions
async def demo_quantum_ml_optimization():
    """Demonstrate quantum ML optimization capabilities."""
    logger.info("Starting Quantum ML Optimization Demo")

    # Initialize optimizer
    optimizer = QuantumMLOptimizer()

    # Initialize Bayesian optimization
    param_space = {
        "learning_rate": (0.0001, 0.01),
        "epsilon": (0.01, 0.3),
        "temperature": (10.0, 200.0),
    }
    optimizer.initialize_bayesian_optimization(param_space)

    # Simulate optimization episodes
    for episode in range(10):
        # Mock task data
        task_data = {
            f"task_{i}": {
                "priority": random.uniform(1, 10),
                "dependencies": [],
                "required_resources": {
                    "cpu_cores": random.uniform(0.5, 4.0),
                    "memory_gb": random.uniform(1.0, 8.0),
                },
                "estimated_duration": random.uniform(300, 3600),
            }
            for i in range(5)
        }

        # Mock system state
        system_state = {
            "cpu_utilization": random.uniform(0.2, 0.8),
            "memory_utilization": random.uniform(0.3, 0.7),
            "active_tasks": random.randint(0, 10),
        }

        # Current schedule
        current_schedule = [f"task_{i}" for i in range(5)]
        random.shuffle(current_schedule)

        # Optimize schedule
        optimized_schedule = optimizer.optimize_schedule(
            current_schedule, task_data, system_state
        )

        # Simulate actual results
        actual_results = {
            "total_execution_time": random.uniform(1000, 5000),
            "success_rate": random.uniform(0.7, 1.0),
            "total_tasks": len(optimized_schedule),
            "task_data": task_data,
        }

        # Update performance
        optimizer.update_performance(optimized_schedule, actual_results, system_state)

        logger.info(f"Episode {episode + 1} completed")

    # Get final statistics
    stats = optimizer.get_optimization_stats()
    logger.info(f"Final optimization stats: {json.dumps(stats, indent=2)}")

    return optimizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_quantum_ml_optimization())
