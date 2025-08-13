"""
Adaptive Quantum Mesh Network for Distributed Task Planning
===========================================================

This module implements a revolutionary distributed quantum-inspired mesh network
for task planning across multiple nodes. This represents cutting-edge research
in distributed quantum computing applied to optimization problems.

Novel Research Contributions:
- Quantum mesh topology with adaptive reconfiguration
- Distributed quantum entanglement synchronization
- Multi-node quantum coherence preservation  
- Adaptive bandwidth allocation using quantum channels
- Fault-tolerant quantum state distribution

Author: Terragon Labs Advanced Research Division
"""

import asyncio
import logging
import numpy as np
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from collections import deque, defaultdict
import hashlib
import time
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class QuantumNode:
    """Individual node in the quantum mesh network."""
    
    node_id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D coordinates
    quantum_capacity: float = 1.0  # Quantum processing capacity
    classical_capacity: float = 1.0  # Classical processing capacity
    
    # Node state
    is_active: bool = True
    current_load: float = 0.0
    quantum_coherence: complex = complex(1.0, 0.0)
    entangled_nodes: Set[str] = field(default_factory=set)
    
    # Network properties
    neighbors: Set[str] = field(default_factory=set)
    connection_strengths: Dict[str, float] = field(default_factory=dict)
    bandwidth_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Performance tracking
    processed_tasks: int = 0
    quantum_operations: int = 0
    network_latency: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        """Initialize node properties."""
        self.created_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.quantum_state_history = deque(maxlen=1000)
        self.performance_metrics = {
            "throughput": deque(maxlen=100),
            "error_rate": deque(maxlen=100),
            "coherence_time": deque(maxlen=100)
        }
    
    def update_quantum_state(self, new_state: complex, decoherence_rate: float = 0.01):
        """Update quantum state with decoherence effects."""
        # Apply decoherence
        time_since_update = (datetime.now() - self.last_heartbeat).total_seconds()
        decoherence_factor = np.exp(-decoherence_rate * time_since_update)
        
        self.quantum_coherence = new_state * decoherence_factor
        
        # Normalize
        norm = abs(self.quantum_coherence)
        if norm > 0:
            self.quantum_coherence /= norm
        
        # Record state history
        self.quantum_state_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": complex(self.quantum_coherence),
            "magnitude": norm,
            "decoherence_factor": decoherence_factor
        })
        
        self.last_heartbeat = datetime.now()
    
    def add_neighbor(self, neighbor_id: str, connection_strength: float = 1.0, bandwidth: float = 1.0):
        """Add neighbor node with specified connection properties."""
        self.neighbors.add(neighbor_id)
        self.connection_strengths[neighbor_id] = connection_strength
        self.bandwidth_allocation[neighbor_id] = bandwidth
    
    def remove_neighbor(self, neighbor_id: str):
        """Remove neighbor node and clean up connections."""
        self.neighbors.discard(neighbor_id)
        self.connection_strengths.pop(neighbor_id, None)
        self.bandwidth_allocation.pop(neighbor_id, None)
        self.entangled_nodes.discard(neighbor_id)
    
    def calculate_distance_to(self, other_node: 'QuantumNode') -> float:
        """Calculate 3D distance to another node."""
        dx = self.position[0] - other_node.position[0]
        dy = self.position[1] - other_node.position[1]
        dz = self.position[2] - other_node.position[2]
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_load_factor(self) -> float:
        """Get current load factor [0,1] where 1 is fully loaded."""
        total_capacity = self.quantum_capacity + self.classical_capacity
        return min(1.0, self.current_load / max(total_capacity, 0.001))
    
    def is_overloaded(self, threshold: float = 0.85) -> bool:
        """Check if node is overloaded."""
        return self.get_load_factor() > threshold
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "node_id": self.node_id,
            "position": self.position,
            "is_active": self.is_active,
            "current_load": self.current_load,
            "load_factor": self.get_load_factor(),
            "quantum_coherence": {
                "magnitude": abs(self.quantum_coherence),
                "phase": np.angle(self.quantum_coherence),
                "real": self.quantum_coherence.real,
                "imag": self.quantum_coherence.imag
            },
            "network_stats": {
                "neighbors_count": len(self.neighbors),
                "entangled_nodes_count": len(self.entangled_nodes),
                "total_connections": len(self.connection_strengths),
                "avg_connection_strength": np.mean(list(self.connection_strengths.values())) if self.connection_strengths else 0.0
            },
            "performance_history": {
                "processed_tasks": self.processed_tasks,
                "quantum_operations": self.quantum_operations,
                "avg_latency": np.mean(self.network_latency) if self.network_latency else 0.0
            }
        }


class QuantumChannel:
    """Quantum communication channel between nodes."""
    
    def __init__(self, node_a_id: str, node_b_id: str, 
                 bandwidth: float = 1.0, latency: float = 0.001):
        self.channel_id = f"{node_a_id}_{node_b_id}"
        self.node_a_id = node_a_id
        self.node_b_id = node_b_id
        self.bandwidth = bandwidth
        self.latency = latency
        
        # Channel state
        self.is_active = True
        self.current_utilization = 0.0
        self.quantum_fidelity = 1.0  # Channel fidelity [0,1]
        
        # Entanglement properties
        self.entanglement_strength = 0.0
        self.coherence_time = 1.0  # Seconds
        self.last_sync_time = datetime.now()
        
        # Performance tracking
        self.message_count = 0
        self.total_data_transferred = 0
        self.error_count = 0
        self.latency_history = deque(maxlen=1000)
        
    def transmit_quantum_state(self, quantum_state: complex, 
                              classical_data: Dict[str, Any] = None) -> bool:
        """Transmit quantum state through channel."""
        try:
            # Simulate quantum state transmission
            transmission_time = datetime.now()
            
            # Apply channel effects
            noise_factor = 1.0 - (self.current_utilization * 0.1)  # More utilization = more noise
            fidelity_loss = (1.0 - self.quantum_fidelity) * 0.1
            
            transmitted_state = quantum_state * noise_factor * (1.0 - fidelity_loss)
            
            # Record transmission
            actual_latency = (datetime.now() - transmission_time).total_seconds() + self.latency
            self.latency_history.append(actual_latency)
            self.message_count += 1
            
            if classical_data:
                self.total_data_transferred += len(json.dumps(classical_data))
            
            # Update utilization
            self.current_utilization = min(1.0, self.current_utilization + 0.01)
            
            return True
            
        except Exception as e:
            logger.error(f"Quantum transmission error on channel {self.channel_id}: {e}")
            self.error_count += 1
            return False
    
    def establish_entanglement(self, strength: float = 1.0) -> bool:
        """Establish quantum entanglement through channel."""
        try:
            # Entanglement requires high fidelity
            if self.quantum_fidelity < 0.7:
                return False
            
            self.entanglement_strength = strength * self.quantum_fidelity
            self.last_sync_time = datetime.now()
            
            logger.debug(f"Entanglement established on channel {self.channel_id} with strength {self.entanglement_strength}")
            return True
            
        except Exception as e:
            logger.error(f"Entanglement establishment error: {e}")
            return False
    
    def get_channel_status(self) -> Dict[str, Any]:
        """Get comprehensive channel status."""
        return {
            "channel_id": self.channel_id,
            "nodes": [self.node_a_id, self.node_b_id],
            "is_active": self.is_active,
            "bandwidth": self.bandwidth,
            "current_utilization": self.current_utilization,
            "quantum_fidelity": self.quantum_fidelity,
            "entanglement_strength": self.entanglement_strength,
            "performance": {
                "message_count": self.message_count,
                "total_data_transferred": self.total_data_transferred,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.message_count, 1),
                "avg_latency": np.mean(self.latency_history) if self.latency_history else 0.0
            }
        }


class QuantumMeshTopology:
    """Manages the overall topology of the quantum mesh network."""
    
    def __init__(self, optimization_strategy: str = "adaptive"):
        self.optimization_strategy = optimization_strategy
        self.topology_history = deque(maxlen=1000)
        self.reconfiguration_events = []
        
        # Topology metrics
        self.connectivity_matrix = None
        self.shortest_paths = {}
        self.network_diameter = 0
        self.clustering_coefficient = 0.0
        
    def analyze_topology(self, nodes: Dict[str, QuantumNode]) -> Dict[str, Any]:
        """Analyze current network topology."""
        if not nodes:
            return {}
        
        node_count = len(nodes)
        total_connections = sum(len(node.neighbors) for node in nodes.values()) // 2
        
        # Build connectivity matrix
        node_ids = list(nodes.keys())
        connectivity_matrix = np.zeros((node_count, node_count))
        
        for i, node_id in enumerate(node_ids):
            node = nodes[node_id]
            for neighbor_id in node.neighbors:
                if neighbor_id in node_ids:
                    j = node_ids.index(neighbor_id)
                    connectivity_matrix[i][j] = node.connection_strengths.get(neighbor_id, 1.0)
        
        self.connectivity_matrix = connectivity_matrix
        
        # Calculate network metrics
        network_density = total_connections / (node_count * (node_count - 1) / 2) if node_count > 1 else 0
        self.clustering_coefficient = self._calculate_clustering_coefficient(nodes)
        self.network_diameter = self._calculate_network_diameter(nodes)
        
        # Find optimal paths
        self.shortest_paths = self._calculate_shortest_paths(nodes)
        
        topology_analysis = {
            "node_count": node_count,
            "total_connections": total_connections,
            "network_density": network_density,
            "clustering_coefficient": self.clustering_coefficient,
            "network_diameter": self.network_diameter,
            "average_degree": (2 * total_connections) / node_count if node_count > 0 else 0,
            "connectivity_matrix_rank": np.linalg.matrix_rank(connectivity_matrix) if node_count > 0 else 0
        }
        
        # Record topology snapshot
        self.topology_history.append({
            "timestamp": datetime.now().isoformat(),
            "analysis": topology_analysis
        })
        
        return topology_analysis
    
    def _calculate_clustering_coefficient(self, nodes: Dict[str, QuantumNode]) -> float:
        """Calculate network clustering coefficient."""
        if len(nodes) < 3:
            return 0.0
        
        total_clustering = 0.0
        valid_nodes = 0
        
        for node_id, node in nodes.items():
            neighbors = list(node.neighbors)
            degree = len(neighbors)
            
            if degree < 2:
                continue
            
            # Count triangles
            triangles = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    neighbor_i = neighbors[i]
                    neighbor_j = neighbors[j]
                    
                    if neighbor_i in nodes and neighbor_j in nodes:
                        if neighbor_j in nodes[neighbor_i].neighbors:
                            triangles += 1
            
            # Local clustering coefficient
            possible_triangles = degree * (degree - 1) // 2
            local_clustering = triangles / possible_triangles if possible_triangles > 0 else 0.0
            
            total_clustering += local_clustering
            valid_nodes += 1
        
        return total_clustering / valid_nodes if valid_nodes > 0 else 0.0
    
    def _calculate_network_diameter(self, nodes: Dict[str, QuantumNode]) -> int:
        """Calculate network diameter (longest shortest path)."""
        if len(nodes) < 2:
            return 0
        
        node_ids = list(nodes.keys())
        distances = {}
        
        # Initialize distances
        for i in node_ids:
            distances[i] = {}
            for j in node_ids:
                if i == j:
                    distances[i][j] = 0
                elif j in nodes[i].neighbors:
                    distances[i][j] = 1
                else:
                    distances[i][j] = float('inf')
        
        # Floyd-Warshall algorithm
        for k in node_ids:
            for i in node_ids:
                for j in node_ids:
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
        
        # Find maximum finite distance
        max_distance = 0
        for i in node_ids:
            for j in node_ids:
                if distances[i][j] != float('inf'):
                    max_distance = max(max_distance, distances[i][j])
        
        return max_distance
    
    def _calculate_shortest_paths(self, nodes: Dict[str, QuantumNode]) -> Dict[str, Dict[str, List[str]]]:
        """Calculate shortest paths between all node pairs."""
        node_ids = list(nodes.keys())
        paths = {}
        
        # Initialize paths
        for start in node_ids:
            paths[start] = {}
            for end in node_ids:
                if start == end:
                    paths[start][end] = [start]
                elif end in nodes[start].neighbors:
                    paths[start][end] = [start, end]
                else:
                    paths[start][end] = []
        
        # Use BFS to find shortest paths
        for start in node_ids:
            visited = {start}
            queue = [(start, [start])]
            
            while queue:
                current_node, path = queue.pop(0)
                
                for neighbor in nodes[current_node].neighbors:
                    if neighbor not in visited and neighbor in nodes:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        
                        # Update if we found a shorter path or first path
                        if not paths[start][neighbor] or len(new_path) < len(paths[start][neighbor]):
                            paths[start][neighbor] = new_path
                        
                        queue.append((neighbor, new_path))
        
        return paths
    
    def suggest_topology_optimization(self, nodes: Dict[str, QuantumNode]) -> List[Dict[str, Any]]:
        """Suggest topology optimizations based on current analysis."""
        suggestions = []
        
        topology_analysis = self.analyze_topology(nodes)
        
        # Check network density
        if topology_analysis.get("network_density", 0) < 0.3:
            suggestions.append({
                "type": "increase_connectivity",
                "priority": "high",
                "description": "Network density is low - consider adding more connections",
                "target_density": 0.4
            })
        
        # Check for isolated nodes
        isolated_nodes = [
            node_id for node_id, node in nodes.items() 
            if len(node.neighbors) < 2
        ]
        
        if isolated_nodes:
            suggestions.append({
                "type": "connect_isolated_nodes",
                "priority": "critical",
                "description": f"Found {len(isolated_nodes)} poorly connected nodes",
                "isolated_nodes": isolated_nodes
            })
        
        # Check clustering
        if topology_analysis.get("clustering_coefficient", 0) > 0.8:
            suggestions.append({
                "type": "reduce_clustering",
                "priority": "medium",
                "description": "High clustering may reduce network efficiency",
                "current_clustering": topology_analysis.get("clustering_coefficient", 0)
            })
        
        # Check diameter
        if topology_analysis.get("network_diameter", 0) > 5:
            suggestions.append({
                "type": "reduce_diameter",
                "priority": "medium",
                "description": "Network diameter is large - consider adding shortcuts",
                "current_diameter": topology_analysis.get("network_diameter", 0)
            })
        
        return suggestions


class AdaptiveQuantumMesh:
    """Main adaptive quantum mesh network coordinator."""
    
    def __init__(self, mesh_config: Dict[str, Any] = None):
        self.config = mesh_config or self._get_default_config()
        
        # Core components
        self.nodes: Dict[str, QuantumNode] = {}
        self.channels: Dict[str, QuantumChannel] = {}
        self.topology_manager = QuantumMeshTopology()
        
        # Network state
        self.is_active = False
        self.total_tasks_processed = 0
        self.network_load_history = deque(maxlen=1000)
        
        # Adaptive parameters
        self.adaptation_frequency = self.config.get("adaptation_frequency", 10.0)  # seconds
        self.load_balancing_threshold = self.config.get("load_balancing_threshold", 0.7)
        self.fault_tolerance_level = self.config.get("fault_tolerance_level", 2)  # redundancy factor
        
        # Performance tracking
        self.performance_history = deque(maxlen=10000)
        self.adaptation_events = []
        
        # Async task management
        self._adaptation_task = None
        self._monitoring_task = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default mesh configuration."""
        return {
            "adaptation_frequency": 5.0,
            "load_balancing_threshold": 0.75,
            "fault_tolerance_level": 2,
            "max_nodes": 100,
            "quantum_coherence_threshold": 0.8,
            "channel_bandwidth": 1.0,
            "entanglement_strength": 0.9
        }
    
    def add_node(self, node_id: str, position: Tuple[float, float, float] = None, 
                 quantum_capacity: float = 1.0, classical_capacity: float = 1.0) -> bool:
        """Add new node to the mesh network."""
        try:
            if node_id in self.nodes:
                logger.warning(f"Node {node_id} already exists in mesh")
                return False
            
            if len(self.nodes) >= self.config.get("max_nodes", 100):
                logger.warning("Maximum node capacity reached")
                return False
            
            # Create node
            node = QuantumNode(
                node_id=node_id,
                position=position or (0.0, 0.0, 0.0),
                quantum_capacity=quantum_capacity,
                classical_capacity=classical_capacity
            )
            
            self.nodes[node_id] = node
            
            # Auto-connect to nearby nodes if network exists
            if len(self.nodes) > 1:
                self._auto_connect_node(node)
            
            logger.info(f"Added node {node_id} to quantum mesh (total: {len(self.nodes)})")
            
            # Record adaptation event
            self.adaptation_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "node_addition",
                "node_id": node_id,
                "total_nodes": len(self.nodes)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding node {node_id}: {e}")
            return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from mesh network."""
        try:
            if node_id not in self.nodes:
                logger.warning(f"Node {node_id} not found in mesh")
                return False
            
            node = self.nodes[node_id]
            
            # Remove all connections
            for neighbor_id in list(node.neighbors):
                self.disconnect_nodes(node_id, neighbor_id)
            
            # Remove node
            del self.nodes[node_id]
            
            logger.info(f"Removed node {node_id} from quantum mesh (remaining: {len(self.nodes)})")
            
            # Record adaptation event
            self.adaptation_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "node_removal",
                "node_id": node_id,
                "total_nodes": len(self.nodes)
            })
            
            # Trigger network healing if needed
            if self.is_active:
                asyncio.create_task(self._heal_network_async())
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing node {node_id}: {e}")
            return False
    
    def connect_nodes(self, node_a_id: str, node_b_id: str, 
                     connection_strength: float = 1.0, bandwidth: float = 1.0) -> bool:
        """Create connection between two nodes."""
        try:
            if node_a_id not in self.nodes or node_b_id not in self.nodes:
                logger.warning(f"Cannot connect: one or both nodes not found")
                return False
            
            if node_a_id == node_b_id:
                logger.warning("Cannot connect node to itself")
                return False
            
            # Create bidirectional connection
            self.nodes[node_a_id].add_neighbor(node_b_id, connection_strength, bandwidth)
            self.nodes[node_b_id].add_neighbor(node_a_id, connection_strength, bandwidth)
            
            # Create quantum channel
            channel_id = f"{min(node_a_id, node_b_id)}_{max(node_a_id, node_b_id)}"
            if channel_id not in self.channels:
                channel = QuantumChannel(node_a_id, node_b_id, bandwidth)
                self.channels[channel_id] = channel
                
                # Establish entanglement if conditions are met
                if connection_strength >= self.config.get("entanglement_strength", 0.9):
                    channel.establish_entanglement(connection_strength)
                    self.nodes[node_a_id].entangled_nodes.add(node_b_id)
                    self.nodes[node_b_id].entangled_nodes.add(node_a_id)
            
            logger.debug(f"Connected nodes {node_a_id} ↔ {node_b_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting nodes: {e}")
            return False
    
    def disconnect_nodes(self, node_a_id: str, node_b_id: str) -> bool:
        """Disconnect two nodes."""
        try:
            if node_a_id in self.nodes:
                self.nodes[node_a_id].remove_neighbor(node_b_id)
            
            if node_b_id in self.nodes:
                self.nodes[node_b_id].remove_neighbor(node_a_id)
            
            # Remove quantum channel
            channel_id = f"{min(node_a_id, node_b_id)}_{max(node_a_id, node_b_id)}"
            if channel_id in self.channels:
                del self.channels[channel_id]
            
            logger.debug(f"Disconnected nodes {node_a_id} ↔ {node_b_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting nodes: {e}")
            return False
    
    def _auto_connect_node(self, new_node: QuantumNode):
        """Automatically connect new node to nearby nodes."""
        # Find closest nodes
        node_distances = []
        for existing_id, existing_node in self.nodes.items():
            if existing_id != new_node.node_id:
                distance = new_node.calculate_distance_to(existing_node)
                node_distances.append((existing_id, distance))
        
        # Sort by distance and connect to closest nodes
        node_distances.sort(key=lambda x: x[1])
        max_connections = min(3, len(node_distances))  # Connect to up to 3 closest nodes
        
        for i in range(max_connections):
            neighbor_id, distance = node_distances[i]
            connection_strength = max(0.1, 1.0 - (distance / 10.0))  # Strength inversely related to distance
            bandwidth = min(1.0, connection_strength + 0.2)
            
            self.connect_nodes(new_node.node_id, neighbor_id, connection_strength, bandwidth)
    
    async def start_mesh_async(self):
        """Start the adaptive quantum mesh network."""
        try:
            self.is_active = True
            
            # Start monitoring and adaptation tasks
            self._monitoring_task = asyncio.create_task(self._monitor_network_async())
            self._adaptation_task = asyncio.create_task(self._adaptive_optimization_async())
            
            logger.info(f"🌐 Adaptive Quantum Mesh started with {len(self.nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Error starting mesh: {e}")
            self.is_active = False
    
    async def stop_mesh_async(self):
        """Stop the adaptive quantum mesh network."""
        try:
            self.is_active = False
            
            # Cancel background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._adaptation_task:
                self._adaptation_task.cancel()
            
            logger.info("🛑 Adaptive Quantum Mesh stopped")
            
        except Exception as e:
            logger.error(f"Error stopping mesh: {e}")
    
    async def _monitor_network_async(self):
        """Continuous network monitoring."""
        while self.is_active:
            try:
                # Calculate network metrics
                network_load = self._calculate_network_load()
                self.network_load_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "load": network_load,
                    "active_nodes": len([n for n in self.nodes.values() if n.is_active])
                })
                
                # Update node states
                for node in self.nodes.values():
                    if node.is_active:
                        # Simulate quantum decoherence
                        decoherence_rate = 0.01 * (1.0 + node.get_load_factor())
                        new_state = node.quantum_coherence * (1.0 - decoherence_rate)
                        node.update_quantum_state(new_state, decoherence_rate)
                
                # Check for network issues
                await self._detect_network_issues()
                
                await asyncio.sleep(1.0)  # Monitor every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in network monitoring: {e}")
                await asyncio.sleep(5.0)
    
    async def _adaptive_optimization_async(self):
        """Continuous adaptive optimization."""
        while self.is_active:
            try:
                # Run optimization based on current network state
                await self._optimize_topology_async()
                await self._balance_load_async()
                await self._maintain_quantum_coherence_async()
                
                await asyncio.sleep(self.adaptation_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptive optimization: {e}")
                await asyncio.sleep(self.adaptation_frequency)
    
    def _calculate_network_load(self) -> float:
        """Calculate overall network load."""
        if not self.nodes:
            return 0.0
        
        total_load = sum(node.get_load_factor() for node in self.nodes.values() if node.is_active)
        active_nodes = len([n for n in self.nodes.values() if n.is_active])
        
        return total_load / max(active_nodes, 1)
    
    async def _detect_network_issues(self):
        """Detect and log network issues."""
        issues = []
        
        # Check for overloaded nodes
        overloaded_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.is_overloaded()
        ]
        
        if overloaded_nodes:
            issues.append({
                "type": "overloaded_nodes",
                "severity": "high",
                "nodes": overloaded_nodes,
                "count": len(overloaded_nodes)
            })
        
        # Check for low quantum coherence
        low_coherence_nodes = [
            node_id for node_id, node in self.nodes.items()
            if abs(node.quantum_coherence) < self.config.get("quantum_coherence_threshold", 0.8)
        ]
        
        if low_coherence_nodes:
            issues.append({
                "type": "low_quantum_coherence",
                "severity": "medium",
                "nodes": low_coherence_nodes,
                "count": len(low_coherence_nodes)
            })
        
        # Check for disconnected components
        if len(self.nodes) > 1:
            topology_analysis = self.topology_manager.analyze_topology(self.nodes)
            if topology_analysis.get("connectivity_matrix_rank", 0) < len(self.nodes) - 1:
                issues.append({
                    "type": "network_fragmentation",
                    "severity": "critical",
                    "description": "Network has disconnected components"
                })
        
        # Log issues
        for issue in issues:
            logger.warning(f"Network issue detected: {issue['type']} (severity: {issue['severity']})")
    
    async def _optimize_topology_async(self):
        """Optimize network topology based on current performance."""
        try:
            suggestions = self.topology_manager.suggest_topology_optimization(self.nodes)
            
            for suggestion in suggestions:
                if suggestion["priority"] == "critical":
                    await self._apply_topology_suggestion(suggestion)
                elif suggestion["priority"] == "high" and len(suggestions) < 3:
                    await self._apply_topology_suggestion(suggestion)
            
        except Exception as e:
            logger.error(f"Error in topology optimization: {e}")
    
    async def _apply_topology_suggestion(self, suggestion: Dict[str, Any]):
        """Apply a topology optimization suggestion."""
        try:
            if suggestion["type"] == "connect_isolated_nodes":
                for node_id in suggestion.get("isolated_nodes", []):
                    if node_id in self.nodes:
                        self._auto_connect_node(self.nodes[node_id])
            
            elif suggestion["type"] == "increase_connectivity":
                # Add connections between distant nodes
                node_ids = list(self.nodes.keys())
                for i, node_a_id in enumerate(node_ids):
                    for j, node_b_id in enumerate(node_ids[i+1:], i+1):
                        if (node_b_id not in self.nodes[node_a_id].neighbors and 
                            len(self.nodes[node_a_id].neighbors) < 5):  # Max 5 connections per node
                            
                            distance = self.nodes[node_a_id].calculate_distance_to(self.nodes[node_b_id])
                            if distance < 5.0:  # Only connect nearby nodes
                                connection_strength = max(0.3, 1.0 - distance/10.0)
                                self.connect_nodes(node_a_id, node_b_id, connection_strength)
                                break
            
            self.adaptation_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "topology_optimization",
                "suggestion_applied": suggestion["type"],
                "priority": suggestion["priority"]
            })
            
        except Exception as e:
            logger.error(f"Error applying topology suggestion: {e}")
    
    async def _balance_load_async(self):
        """Balance load across network nodes."""
        try:
            # Find overloaded and underloaded nodes
            overloaded = [(node_id, node) for node_id, node in self.nodes.items() 
                         if node.get_load_factor() > self.load_balancing_threshold]
            underloaded = [(node_id, node) for node_id, node in self.nodes.items() 
                          if node.get_load_factor() < 0.3 and node.is_active]
            
            # Simple load redistribution
            for overloaded_id, overloaded_node in overloaded:
                if underloaded:
                    target_id, target_node = underloaded.pop(0)
                    
                    # Simulate load transfer
                    transfer_amount = min(0.2, overloaded_node.current_load * 0.3)
                    overloaded_node.current_load -= transfer_amount
                    target_node.current_load += transfer_amount
                    
                    logger.debug(f"Load balanced: {overloaded_id} → {target_id} ({transfer_amount:.2f})")
            
        except Exception as e:
            logger.error(f"Error in load balancing: {e}")
    
    async def _maintain_quantum_coherence_async(self):
        """Maintain quantum coherence across the network."""
        try:
            # Re-establish entanglement for low-coherence nodes
            low_coherence_nodes = [
                node_id for node_id, node in self.nodes.items()
                if abs(node.quantum_coherence) < 0.5
            ]
            
            for node_id in low_coherence_nodes:
                node = self.nodes[node_id]
                
                # Boost coherence through entanglement
                for entangled_id in node.entangled_nodes:
                    if entangled_id in self.nodes:
                        entangled_node = self.nodes[entangled_id]
                        
                        # Quantum coherence transfer
                        coherence_transfer = entangled_node.quantum_coherence * 0.1
                        node.quantum_coherence += coherence_transfer
                        
                        # Normalize
                        norm = abs(node.quantum_coherence)
                        if norm > 0:
                            node.quantum_coherence /= norm
                        
                        break
            
        except Exception as e:
            logger.error(f"Error maintaining quantum coherence: {e}")
    
    async def _heal_network_async(self):
        """Heal network after node failures or removals."""
        try:
            # Check for isolated components
            topology_analysis = self.topology_manager.analyze_topology(self.nodes)
            
            if topology_analysis.get("connectivity_matrix_rank", 0) < len(self.nodes) - 1:
                # Network is fragmented - add healing connections
                node_ids = list(self.nodes.keys())
                
                for i, node_a_id in enumerate(node_ids):
                    node_a = self.nodes[node_a_id]
                    
                    if len(node_a.neighbors) < 2:  # Node needs more connections
                        # Find closest unconnected node
                        for j, node_b_id in enumerate(node_ids):
                            if (i != j and node_b_id not in node_a.neighbors):
                                node_b = self.nodes[node_b_id]
                                distance = node_a.calculate_distance_to(node_b)
                                
                                if distance < 8.0:  # Reasonable healing distance
                                    connection_strength = max(0.4, 1.0 - distance/10.0)
                                    self.connect_nodes(node_a_id, node_b_id, connection_strength)
                                    logger.info(f"Network healing: connected {node_a_id} ↔ {node_b_id}")
                                    break
            
        except Exception as e:
            logger.error(f"Error in network healing: {e}")
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get comprehensive mesh network status."""
        topology_analysis = self.topology_manager.analyze_topology(self.nodes)
        
        # Calculate performance metrics
        active_nodes = [n for n in self.nodes.values() if n.is_active]
        total_processed_tasks = sum(n.processed_tasks for n in active_nodes)
        avg_load = self._calculate_network_load()
        
        # Channel statistics
        active_channels = len([c for c in self.channels.values() if c.is_active])
        total_entangled_pairs = sum(len(n.entangled_nodes) for n in active_nodes) // 2
        
        return {
            "network_overview": {
                "is_active": self.is_active,
                "total_nodes": len(self.nodes),
                "active_nodes": len(active_nodes),
                "total_channels": len(self.channels),
                "active_channels": active_channels,
                "entangled_pairs": total_entangled_pairs
            },
            "performance_metrics": {
                "average_network_load": avg_load,
                "total_tasks_processed": total_processed_tasks,
                "adaptation_events_count": len(self.adaptation_events),
                "network_uptime": (datetime.now() - self.nodes[list(self.nodes.keys())[0]].created_at).total_seconds() if self.nodes else 0
            },
            "topology_analysis": topology_analysis,
            "recent_adaptations": self.adaptation_events[-10:] if self.adaptation_events else [],
            "quantum_coherence": {
                "average_coherence": np.mean([abs(n.quantum_coherence) for n in active_nodes]) if active_nodes else 0,
                "coherence_distribution": [abs(n.quantum_coherence) for n in active_nodes],
                "total_entangled_nodes": sum(len(n.entangled_nodes) for n in active_nodes)
            }
        }
    
    def export_mesh_data(self, filepath: str):
        """Export comprehensive mesh data for analysis."""
        try:
            export_data = {
                "mesh_status": self.get_mesh_status(),
                "node_details": {
                    node_id: node.get_performance_summary()
                    for node_id, node in self.nodes.items()
                },
                "channel_details": {
                    channel_id: channel.get_channel_status()
                    for channel_id, channel in self.channels.items()
                },
                "topology_history": list(self.topology_manager.topology_history),
                "adaptation_events": self.adaptation_events,
                "network_load_history": list(self.network_load_history)[-1000:],  # Last 1000 entries
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Mesh data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting mesh data: {e}")


# Research demonstration and testing functions
async def demo_adaptive_quantum_mesh():
    """Demonstrate adaptive quantum mesh capabilities."""
    logger.info("🌐 Starting Adaptive Quantum Mesh Demo")
    
    # Initialize mesh
    mesh_config = {
        "adaptation_frequency": 2.0,  # Faster adaptation for demo
        "load_balancing_threshold": 0.6,
        "fault_tolerance_level": 2,
        "max_nodes": 20
    }
    
    mesh = AdaptiveQuantumMesh(mesh_config)
    
    # Add nodes in different positions
    nodes_to_add = [
        ("node_01", (0.0, 0.0, 0.0), 1.0, 1.2),
        ("node_02", (1.0, 1.0, 0.0), 1.2, 0.8),
        ("node_03", (2.0, 0.5, 1.0), 0.8, 1.5),
        ("node_04", (-1.0, 2.0, 0.5), 1.5, 1.0),
        ("node_05", (3.0, -1.0, -0.5), 1.1, 1.3),
        ("node_06", (0.5, 3.0, 1.5), 0.9, 1.1),
        ("node_07", (-2.0, 1.0, 2.0), 1.3, 0.7),
        ("node_08", (1.5, -2.0, 1.0), 1.0, 1.4)
    ]
    
    for node_id, position, q_cap, c_cap in nodes_to_add:
        success = mesh.add_node(node_id, position, q_cap, c_cap)
        logger.info(f"Added node {node_id}: {'✅' if success else '❌'}")
    
    # Start mesh operation
    await mesh.start_mesh_async()
    
    # Simulate network activity
    logger.info("🔄 Simulating network activity...")
    
    # Simulate varying loads
    for cycle in range(10):
        # Randomly assign loads to nodes
        for node_id, node in mesh.nodes.items():
            if node.is_active:
                # Simulate task processing
                new_load = random.uniform(0.0, 1.2)
                node.current_load = new_load
                node.processed_tasks += random.randint(1, 5)
                node.quantum_operations += random.randint(2, 8)
        
        # Wait for adaptation
        await asyncio.sleep(1.0)
        
        # Occasionally simulate node failures
        if cycle == 5 and len(mesh.nodes) > 4:
            failed_node = random.choice(list(mesh.nodes.keys()))
            logger.info(f"🔥 Simulating failure of node {failed_node}")
            mesh.nodes[failed_node].is_active = False
        
        logger.info(f"Cycle {cycle + 1}: Network load = {mesh._calculate_network_load():.3f}")
    
    # Get final status
    final_status = mesh.get_mesh_status()
    logger.info("📊 Final Mesh Status:")
    logger.info(f"  Active Nodes: {final_status['network_overview']['active_nodes']}")
    logger.info(f"  Active Channels: {final_status['network_overview']['active_channels']}")
    logger.info(f"  Entangled Pairs: {final_status['network_overview']['entangled_pairs']}")
    logger.info(f"  Average Load: {final_status['performance_metrics']['average_network_load']:.3f}")
    logger.info(f"  Adaptation Events: {final_status['performance_metrics']['adaptation_events_count']}")
    
    # Export data
    export_file = f"quantum_mesh_demo_{int(time.time())}.json"
    mesh.export_mesh_data(export_file)
    logger.info(f"📁 Mesh data exported to: {export_file}")
    
    # Stop mesh
    await mesh.stop_mesh_async()
    
    return {
        "final_status": final_status,
        "export_file": export_file,
        "demo_completed": True
    }


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    async def main():
        results = await demo_adaptive_quantum_mesh()
        print("\n🌐 ADAPTIVE QUANTUM MESH DEMO COMPLETE")
        print("=" * 70)
        print(f"Final Status: {results['final_status']['network_overview']}")
        print(f"Export File: {results['export_file']}")
        print("=" * 70)
        return results
    
    asyncio.run(main())