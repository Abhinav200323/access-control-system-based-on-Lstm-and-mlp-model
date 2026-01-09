"""
Live Wireshark (tshark) capture and real-time feature extraction for Streamlit.

This module uses tshark (Wireshark's command-line tool) to capture live network traffic
and convert it to KDD features in real-time for display in the frontend.
"""

from __future__ import annotations

import subprocess
import threading
import time
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# KDD99 feature names (first 41 features)
KDD99_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Protocol mappings
PROTOCOL_MAP = {
    '1': 'icmp', '6': 'tcp', '17': 'udp', 'icmp': 'icmp', 'tcp': 'tcp', 'udp': 'udp'
}

# Service port mappings
SERVICE_MAP = {
    '21': 'ftp', '22': 'ssh', '23': 'telnet', '25': 'smtp', '53': 'domain',
    '80': 'http', '110': 'pop_3', '143': 'imap4', '443': 'https', '993': 'imaps', '995': 'pop3s',
    '20': 'ftp_data', '69': 'tftp', '123': 'ntp', '161': 'snmp', '179': 'bgp',
    '389': 'ldap', '636': 'ldaps', '1433': 'sql_server', '3306': 'mysql', '5432': 'postgresql'
}


def check_tshark_available() -> bool:
    """Check if tshark is available on the system."""
    try:
        result = subprocess.run(
            ['tshark', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_available_interfaces() -> List[str]:
    """Get list of available network interfaces from tshark."""
    try:
        result = subprocess.run(
            ['tshark', '-D'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            interfaces = []
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('tshark:'):
                    # Format: "1. en0"
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        interfaces.append(parts[1].strip())
            return interfaces
    except Exception:
        pass
    return []


class WiresharkLiveCapture:
    """
    Live network capture using tshark with real-time KDD feature extraction.
    """
    
    def __init__(self, interface: Optional[str] = None, bpf_filter: Optional[str] = None, max_flows: int = 1000):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.max_flows = max_flows
        self.is_capturing = False
        self.capture_process = None
        self.capture_thread = None
        
        # Data structures
        self.flows: Dict[str, Dict] = {}
        self.flow_history: deque = deque(maxlen=max_flows)
        self.host_stats: Dict[str, Dict] = defaultdict(lambda: {
            'connections': 0,
            'services': set(),
            'ports': set(),
            'errors': 0
        })
        
        # Per-user sequences for behavioral LSTM
        self.sequence_length: int = 20
        self.user_sequences: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.sequence_length)
        )
        
        self.lock = threading.Lock()
        self.packet_buffer: List[Dict] = []
    
    def _parse_tshark_json(self, json_line: str) -> Optional[Dict]:
        """Parse a single JSON line from tshark output."""
        try:
            data = json.loads(json_line.strip())
            if '_source' not in data or 'layers' not in data['_source']:
                return None
            
            layers = data['_source']['layers']
            
            # Extract IP layer
            if 'ip' not in layers:
                return None
            
            ip_layer = layers['ip']
            src_ip = ip_layer.get('ip.src', '')
            dst_ip = ip_layer.get('ip.dst', '')
            protocol = ip_layer.get('ip.proto', '')
            
            # Extract transport layer
            src_port = '0'
            dst_port = '0'
            service = 'other'
            flag = 'OTH'
            
            if 'tcp' in layers:
                tcp = layers['tcp']
                src_port = tcp.get('tcp.srcport', '0')
                dst_port = tcp.get('tcp.dstport', '0')
                service = SERVICE_MAP.get(dst_port, 'other')
                
                # TCP flags
                flags = tcp.get('tcp.flags', {})
                flag_str = ''
                if isinstance(flags, dict):
                    if flags.get('tcp.flags.syn', '0') == '1':
                        flag_str += 'S'
                    if flags.get('tcp.flags.ack', '0') == '1':
                        flag_str += 'A'
                    if flags.get('tcp.flags.rst', '0') == '1':
                        flag_str += 'R'
                    if flags.get('tcp.flags.fin', '0') == '1':
                        flag_str += 'F'
                    if flags.get('tcp.flags.push', '0') == '1':
                        flag_str += 'P'
                flag = flag_str if flag_str else 'OTH'
                
            elif 'udp' in layers:
                udp = layers['udp']
                src_port = udp.get('udp.srcport', '0')
                dst_port = udp.get('udp.dstport', '0')
                service = SERVICE_MAP.get(dst_port, 'other')
                flag = 'OTH'
            
            # Extract frame info
            frame = layers.get('frame', {})
            frame_len = int(frame.get('frame.len', '0'))
            timestamp = float(frame.get('frame.time_epoch', time.time()))
            
            return {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': PROTOCOL_MAP.get(protocol, 'other'),
                'service': service,
                'flag': flag,
                'length': frame_len,
                'timestamp': timestamp
            }
        except Exception:
            return None
    
    def _extract_flow_features(self, packet_info: Dict) -> Optional[Dict[str, Any]]:
        """Extract KDD features from packet info and update flow statistics."""
        if not packet_info:
            return None
        
        src_ip = packet_info['src_ip']
        dst_ip = packet_info['dst_ip']
        src_port = int(packet_info.get('src_port', 0))
        dst_port = int(packet_info.get('dst_port', 0))
        protocol = packet_info['protocol']
        service = packet_info['service']
        flag = packet_info['flag']
        length = packet_info['length']
        timestamp = packet_info['timestamp']
        
        # Create flow key
        if src_ip < dst_ip or (src_ip == dst_ip and src_port < dst_port):
            flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        with self.lock:
            # Initialize or update flow
            if flow_key not in self.flows:
                self.flows[flow_key] = {
                    'start_time': timestamp,
                    'packet_count': 0,
                    'src_bytes': 0,
                    'dst_bytes': 0,
                    'services': set(),
                    'flags': [],
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'protocol': protocol,
                    'packets': []
                }
            
            flow = self.flows[flow_key]
            flow['packet_count'] += 1
            flow['packets'].append(packet_info)
            
            # Update bytes
            if packet_info['src_ip'] == src_ip:
                flow['src_bytes'] += length
            else:
                flow['dst_bytes'] += length
            
            flow['services'].add(service)
            flow['flags'].append(flag)
            flow['duration'] = timestamp - flow['start_time']
            
            # Update host statistics
            self.host_stats[dst_ip]['connections'] += 1
            self.host_stats[dst_ip]['services'].add(service)
            self.host_stats[dst_ip]['ports'].add(dst_port)
            
            # Create KDD feature vector (first 41 features only)
            features = {col: 0 for col in KDD99_COLUMNS[:41]}
            
            features['duration'] = flow['duration']
            features['protocol_type'] = protocol
            features['service'] = service
            features['flag'] = flag
            features['src_bytes'] = flow['src_bytes']
            features['dst_bytes'] = flow['dst_bytes']
            features['land'] = 1 if src_ip == dst_ip else 0
            features['count'] = flow['packet_count']
            features['srv_count'] = len(flow['services'])
            features['same_srv_rate'] = 1.0 / max(flow['packet_count'], 1)
            features['diff_srv_rate'] = 1.0 - features['same_srv_rate']
            
            # Destination host features
            features['dst_host_count'] = self.host_stats[dst_ip]['connections']
            features['dst_host_srv_count'] = len(self.host_stats[dst_ip]['services'])
            features['dst_host_same_srv_rate'] = 1.0 / max(self.host_stats[dst_ip]['connections'], 1)
            features['dst_host_diff_srv_rate'] = 1.0 - features['dst_host_same_srv_rate']
            
            # Error rates (simplified)
            rst_count = sum(1 for f in flow['flags'] if 'R' in f)
            features['rerror_rate'] = rst_count / max(flow['packet_count'], 1)
            features['serror_rate'] = sum(1 for f in flow['flags'] if 'S' in f) / max(flow['packet_count'], 1)
            features['srv_serror_rate'] = features['serror_rate']
            features['srv_rerror_rate'] = features['rerror_rate']
            features['dst_host_serror_rate'] = features['serror_rate']
            features['dst_host_srv_serror_rate'] = features['serror_rate']
            features['dst_host_rerror_rate'] = features['rerror_rate']
            features['dst_host_srv_rerror_rate'] = features['rerror_rate']
            
            # Add metadata
            features['src_ip'] = src_ip
            features['dst_ip'] = dst_ip
            features['src_port'] = src_port
            features['dst_port'] = dst_port
            user_id = str(src_ip)  # Use src_ip as user_id
            features['user_id'] = user_id
            features['timestamp'] = timestamp
            
            # Update per-user sequence history for behavioral LSTM
            # Build vector in the canonical KDD feature order (first 41 features)
            vec = [features.get(col, 0) for col in KDD99_COLUMNS[:41]]
            self.user_sequences[user_id].append(vec)
            
            # Move completed flows to history
            if flow['packet_count'] >= 10 or flow['duration'] > 60:  # Flow complete conditions
                if flow_key in self.flows:
                    self.flow_history.append(features.copy())
                    del self.flows[flow_key]
            
            return features
    
    def _capture_loop(self):
        """Main capture loop running in background thread."""
        # Build tshark command
        cmd = ['tshark', '-i', self.interface or 'any', '-T', 'json', '-e', 'frame.number',
               '-e', 'frame.time_epoch', '-e', 'frame.len', '-e', 'ip.src', '-e', 'ip.dst',
               '-e', 'ip.proto', '-e', 'tcp.srcport', '-e', 'tcp.dstport', '-e', 'tcp.flags',
               '-e', 'udp.srcport', '-e', 'udp.dstport']
        
        if self.bpf_filter:
            cmd.extend(['-f', self.bpf_filter])
        
        try:
            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            for line in self.capture_process.stdout:
                if not self.is_capturing:
                    break
                
                packet_info = self._parse_tshark_json(line)
                if packet_info:
                    features = self._extract_flow_features(packet_info)
                    if features:
                        with self.lock:
                            self.packet_buffer.append(features)
                            # Keep buffer size manageable
                            if len(self.packet_buffer) > self.max_flows:
                                self.packet_buffer = self.packet_buffer[-self.max_flows:]
        except Exception as e:
            print(f"Capture error: {e}")
        finally:
            if self.capture_process:
                self.capture_process.terminate()
                self.capture_process.wait()
    
    def start(self, interface: Optional[str] = None, bpf_filter: Optional[str] = None):
        """Start live capture."""
        if self.is_capturing:
            return
        
        self.interface = interface or self.interface
        self.bpf_filter = bpf_filter or self.bpf_filter
        
        self.is_capturing = True
        self.packet_buffer = []
        self.flows.clear()
        self.flow_history.clear()
        self.host_stats.clear()
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop(self):
        """Stop live capture."""
        self.is_capturing = False
        if self.capture_process:
            self.capture_process.terminate()
            self.capture_process.wait()
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
    
    def dataframe(self) -> pd.DataFrame:
        """Get current captured data as DataFrame."""
        with self.lock:
            # Combine buffer and history
            all_data = list(self.packet_buffer) + list(self.flow_history)
            
            if not all_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_data)
            
            # Ensure all KDD columns exist (first 41 features only)
            for col in KDD99_COLUMNS[:41]:
                if col not in df.columns:
                    df[col] = 0
            
            return df
    
    def get_user_sequences(self, min_len: int = 5) -> Dict[str, np.ndarray]:
        """Get per-user sequences for behavioral LSTM."""
        with self.lock:
            user_seqs = {}
            for user_id, seq_deque in self.user_sequences.items():
                if len(seq_deque) >= min_len:
                    # Convert deque to numpy array
                    seq_array = np.array(list(seq_deque))
                    user_seqs[user_id] = seq_array
            return user_seqs


def get_wireshark_capture(interface: Optional[str] = None, bpf_filter: Optional[str] = None) -> WiresharkLiveCapture:
    """Factory function to get Wireshark live capture instance."""
    if not check_tshark_available():
        raise RuntimeError(
            "tshark (Wireshark command-line tool) is not available. "
            "Please install Wireshark: https://www.wireshark.org/download.html"
        )
    return WiresharkLiveCapture(interface=interface, bpf_filter=bpf_filter)

