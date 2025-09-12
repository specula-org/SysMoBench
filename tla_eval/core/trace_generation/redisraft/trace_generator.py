#!/usr/bin/env python3
"""
RedisRaft Trace Generator

This script spawns multiple Redis instances with RedisRaft module loaded,
simulates client operations and fault injection, and captures trace logs
for TLA+ model checking.
"""

import os
import sys
import json
import time
import signal
import random
import socket
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import redis
from redis.exceptions import ConnectionError, TimeoutError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RedisRaftConfig:
    """Configuration for RedisRaft trace generation"""
    node_count: int = 3
    duration_seconds: int = 60
    client_qps: float = 10.0
    fault_rate: float = 0.1
    output_file: str = "trace.ndjson"
    random_seed: Optional[int] = None
    redis_port_start: int = 7000
    raft_port_start: int = 8000
    redis_executable: str = "redis-server"
    redisraft_module: str = None
    working_dir: str = "/tmp/redisraft_trace"
    enable_debug_trace: bool = True
    trace_filter: str = "all"  # all, election, logsync, coarse


class RedisRaftNode:
    """Represents a single RedisRaft node"""
    
    def __init__(self, node_id: int, config: RedisRaftConfig):
        self.node_id = node_id
        self.config = config
        self.redis_port = config.redis_port_start + node_id
        self.raft_port = config.raft_port_start + node_id
        self.process = None
        self.client = None
        self.log_file = None
        self.working_dir = Path(config.working_dir) / f"node_{node_id}"
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.isolated = False
        
    def start(self, join_addr: Optional[str] = None):
        """Start the Redis server with RedisRaft module"""
        
        # Find RedisRaft module if not specified
        if not self.config.redisraft_module:
            # Try to find the compiled module
            module_paths = [
                Path(__file__).parent.parent.parent.parent.parent / "data/repositories/redisraft/redisraft.so",
                Path("/usr/local/lib/redisraft.so"),
                Path("./redisraft.so"),
            ]
            for path in module_paths:
                if path.exists():
                    self.config.redisraft_module = str(path)
                    break
            
            if not self.config.redisraft_module:
                raise FileNotFoundError("RedisRaft module not found. Please build it first or specify path.")
        
        # Create Redis configuration
        redis_conf = self.working_dir / "redis.conf"
        with open(redis_conf, 'w') as f:
            f.write(f"""
port {self.redis_port}
dir {self.working_dir}
logfile redis.log
loglevel debug
save ""
appendonly no
""")
        
        # Prepare command
        cmd = [
            self.config.redis_executable,
            str(redis_conf),
            "--loadmodule", self.config.redisraft_module,
            f"raft-log-filename=raftlog-{self.node_id}.db",
            f"addr=127.0.0.1:{self.raft_port}",
            f"id={self.node_id}",
        ]
        
        # Enable debug tracing if requested
        if self.config.enable_debug_trace:
            cmd.append("loglevel=debug")
            cmd.append("trace=31")  # Enable all trace flags
        
        # Open log file for capturing output
        self.log_file = open(self.working_dir / "redis.log", 'w')
        
        # Start Redis server
        logger.info(f"Starting node {self.node_id} on ports {self.redis_port}/{self.raft_port}")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            cwd=str(self.working_dir)
        )
        
        # Wait for Redis to start
        time.sleep(1)
        
        # Connect Redis client
        self.client = redis.Redis(host='127.0.0.1', port=self.redis_port, decode_responses=True)
        
        # Wait for connection
        for _ in range(10):
            try:
                self.client.ping()
                break
            except ConnectionError:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"Failed to connect to Redis node {self.node_id}")
        
        # Join cluster if not the first node
        if join_addr:
            self.join_cluster(join_addr)
    
    def join_cluster(self, leader_addr: str):
        """Join an existing cluster"""
        try:
            result = self.client.execute_command('RAFT.CLUSTER', 'JOIN', leader_addr)
            logger.info(f"Node {self.node_id} joined cluster at {leader_addr}: {result}")
        except Exception as e:
            logger.error(f"Node {self.node_id} failed to join cluster: {e}")
    
    def propose_value(self, key: str, value: str):
        """Propose a key-value pair to the cluster"""
        try:
            self.client.set(key, value)
            return True
        except Exception as e:
            logger.debug(f"Node {self.node_id} failed to propose: {e}")
            return False
    
    def isolate(self):
        """Simulate network partition by dropping messages"""
        if not self.isolated:
            logger.info(f"Isolating node {self.node_id}")
            # We can't easily simulate network partition without modifying Redis
            # Instead, we'll pause the process
            if self.process:
                self.process.send_signal(signal.SIGSTOP)
            self.isolated = True
    
    def recover(self):
        """Recover from network partition"""
        if self.isolated:
            logger.info(f"Recovering node {self.node_id}")
            if self.process:
                self.process.send_signal(signal.SIGCONT)
            self.isolated = False
    
    def stop(self):
        """Stop the Redis server"""
        if self.client:
            try:
                self.client.close()
            except:
                pass
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        if self.log_file:
            self.log_file.close()
    
    def get_info(self) -> Dict[str, Any]:
        """Get Raft info from the node"""
        try:
            info = self.client.execute_command('RAFT.INFO')
            return self._parse_info(info)
        except Exception as e:
            logger.debug(f"Failed to get info from node {self.node_id}: {e}")
            return {}
    
    def _parse_info(self, info_str: str) -> Dict[str, Any]:
        """Parse RAFT.INFO output"""
        info = {}
        for line in info_str.split('\r\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key] = value
        return info


class RedisRaftCluster:
    """Manages a cluster of RedisRaft nodes"""
    
    def __init__(self, config: RedisRaftConfig):
        self.config = config
        self.nodes: List[RedisRaftNode] = []
        self.random = random.Random(config.random_seed)
        
    def start(self):
        """Start all nodes in the cluster"""
        logger.info(f"Starting RedisRaft cluster with {self.config.node_count} nodes")
        
        # Start first node
        first_node = RedisRaftNode(1, self.config)
        first_node.start()
        self.nodes.append(first_node)
        
        # Wait for first node to be ready
        time.sleep(2)
        
        # Get first node's address for others to join
        first_addr = f"127.0.0.1:{first_node.raft_port}"
        
        # Start remaining nodes and join cluster
        for i in range(2, self.config.node_count + 1):
            node = RedisRaftNode(i, self.config)
            node.start(join_addr=first_addr)
            self.nodes.append(node)
            time.sleep(1)
        
        logger.info("Cluster started successfully")
    
    def stop(self):
        """Stop all nodes in the cluster"""
        logger.info("Stopping cluster")
        for node in self.nodes:
            node.stop()
    
    def propose_random(self):
        """Make a random proposal to the cluster"""
        node = self.random.choice(self.nodes)
        if not node.isolated:
            key = f"key_{self.random.randint(1, 100)}"
            value = f"value_{int(time.time() * 1000000)}"
            success = node.propose_value(key, value)
            logger.debug(f"Node {node.node_id} proposed {key}={value}: {success}")
    
    def inject_fault(self):
        """Inject a random fault into the cluster"""
        fault_type = self.random.choice(['isolate', 'recover', 'restart'])
        node = self.random.choice(self.nodes)
        
        if fault_type == 'isolate' and not node.isolated:
            node.isolate()
        elif fault_type == 'recover' and node.isolated:
            node.recover()
        elif fault_type == 'restart':
            logger.info(f"Restarting node {node.node_id}")
            node.stop()
            time.sleep(1)
            # Restart and rejoin
            first_addr = f"127.0.0.1:{self.nodes[0].raft_port}"
            node.start(join_addr=first_addr if node.node_id != 1 else None)
    
    def collect_info(self) -> List[Dict[str, Any]]:
        """Collect Raft info from all nodes"""
        info = []
        for node in self.nodes:
            node_info = node.get_info()
            if node_info:
                node_info['node_id'] = node.node_id
                node_info['isolated'] = node.isolated
                info.append(node_info)
        return info


class TraceGenerator:
    """Generates traces from a RedisRaft cluster"""
    
    def __init__(self, config: RedisRaftConfig):
        self.config = config
        self.cluster = RedisRaftCluster(config)
        self.trace_events = []
        self.start_time = None
    
    def generate(self):
        """Generate traces by running the cluster"""
        logger.info(f"Starting trace generation for {self.config.duration_seconds} seconds")
        
        # Start cluster
        self.cluster.start()
        
        # Let cluster stabilize
        time.sleep(3)
        
        self.start_time = time.time()
        end_time = self.start_time + self.config.duration_seconds
        
        # Calculate operation intervals
        client_interval = 1.0 / self.config.client_qps if self.config.client_qps > 0 else 1.0
        
        next_client_op = time.time()
        next_fault_op = time.time() + self.cluster.random.uniform(5, 10)
        next_info_collect = time.time() + 1
        
        operation_count = 0
        fault_count = 0
        
        try:
            while time.time() < end_time:
                now = time.time()
                
                # Client operations
                if now >= next_client_op:
                    self.cluster.propose_random()
                    operation_count += 1
                    next_client_op = now + client_interval * (0.5 + self.cluster.random.random())
                
                # Fault injection
                if self.config.fault_rate > 0 and now >= next_fault_op:
                    if self.cluster.random.random() < self.config.fault_rate:
                        self.cluster.inject_fault()
                        fault_count += 1
                    next_fault_op = now + self.cluster.random.uniform(5, 15)
                
                # Collect cluster info periodically
                if now >= next_info_collect:
                    info = self.cluster.collect_info()
                    self.record_cluster_state(info)
                    next_info_collect = now + 1
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            logger.info("Trace generation interrupted")
        
        finally:
            # Stop cluster
            self.cluster.stop()
            
            # Parse logs and generate trace
            self.parse_logs()
            
            # Write trace file
            self.write_trace()
            
            logger.info(f"Trace generation completed: {operation_count} operations, {fault_count} faults")
    
    def record_cluster_state(self, info: List[Dict[str, Any]]):
        """Record cluster state as trace events"""
        timestamp = time.time() - self.start_time
        for node_info in info:
            event = {
                "timestamp": timestamp,
                "node_id": node_info.get('node_id'),
                "role": node_info.get('role', 'unknown'),
                "term": int(node_info.get('current_term', 0)),
                "commit_index": int(node_info.get('commit_index', 0)),
                "last_applied": int(node_info.get('last_applied_index', 0)),
                "isolated": node_info.get('isolated', False),
            }
            self.trace_events.append(event)
    
    def parse_logs(self):
        """Parse Redis logs to extract Raft events"""
        logger.info("Parsing logs to extract trace events")
        
        for node in self.cluster.nodes:
            log_file = node.working_dir / "redis.log"
            if log_file.exists():
                self.parse_node_log(node.node_id, log_file)
    
    def parse_node_log(self, node_id: int, log_file: Path):
        """Parse a single node's log file"""
        # This would parse the Redis log output and extract Raft events
        # For now, we'll just note that logs exist
        logger.debug(f"Parsing log for node {node_id}: {log_file}")
        
        # TODO: Implement actual log parsing based on RedisRaft log format
        # This would extract events like:
        # - State transitions (follower -> candidate -> leader)
        # - Vote requests and responses
        # - AppendEntries messages
        # - Log commits
    
    def write_trace(self):
        """Write trace events to output file"""
        logger.info(f"Writing {len(self.trace_events)} trace events to {self.config.output_file}")
        
        # Sort events by timestamp
        self.trace_events.sort(key=lambda x: x['timestamp'])
        
        # Write as NDJSON
        with open(self.config.output_file, 'w') as f:
            for event in self.trace_events:
                json.dump(event, f)
                f.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Generate RedisRaft traces for TLA+ model checking')
    parser.add_argument('--nodes', type=int, default=3, help='Number of nodes in cluster')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    parser.add_argument('--qps', type=float, default=10.0, help='Client operations per second')
    parser.add_argument('--fault-rate', type=float, default=0.1, help='Fault injection rate')
    parser.add_argument('--output', default='trace.ndjson', help='Output trace file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--redis-exe', default='redis-server', help='Path to redis-server')
    parser.add_argument('--module', help='Path to redisraft.so module')
    parser.add_argument('--work-dir', default='/tmp/redisraft_trace', help='Working directory')
    parser.add_argument('--filter', choices=['all', 'election', 'logsync', 'coarse'], 
                       default='all', help='Trace filter type')
    
    args = parser.parse_args()
    
    config = RedisRaftConfig(
        node_count=args.nodes,
        duration_seconds=args.duration,
        client_qps=args.qps,
        fault_rate=args.fault_rate,
        output_file=args.output,
        random_seed=args.seed,
        redis_executable=args.redis_exe,
        redisraft_module=args.module,
        working_dir=args.work_dir,
        trace_filter=args.filter,
    )
    
    generator = TraceGenerator(config)
    generator.generate()


if __name__ == '__main__':
    main()