"""
Scapy-based packet capture and feature extraction for attack detection.
This module provides real-time network packet analysis capabilities.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

try:
    from scapy.all import *
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Install with: pip install scapy")

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

class PacketFeaturizer:
    """
    Real-time packet capture and feature extraction for attack detection.
    """
    
    def __init__(self, window_size: int = 100, max_flows: int = 1000):
        self.window_size = window_size
        self.max_flows = max_flows
        self.is_capturing = False
        self.capture_thread = None

        # Data structures for feature computation
        self.flows: Dict[str, Dict] = {}  # Active flows
        self.flow_history: deque = deque(maxlen=max_flows)  # Recent flows
        self.host_stats: Dict[str, Dict] = defaultdict(lambda: {
            'connections': 0, 'services': set(), 'ports': set(), 'errors': 0
        })

        # Per-user behavioural sequences (for behavioural LSTM)
        # Key: user_id (currently src_ip); Value: deque of last N KDD feature vectors
        self.sequence_length: int = 20
        self.user_sequences: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.sequence_length)
        )
        
        # Protocol mappings
        self.protocol_map = {
            1: 'icmp', 6: 'tcp', 17: 'udp'
        }
        
        # Service port mappings (simplified)
        self.service_map = {
            21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'domain',
            80: 'http', 110: 'pop_3', 143: 'imap4', 443: 'https', 993: 'imaps', 995: 'pop3s'
        }
        
        # Flag mappings for TCP (KDD format uses string flags)
        # No numeric mapping needed - we'll build string flags
        
        self.lock = threading.Lock()
    
    def _extract_basic_features(self, packet, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract basic features from a single packet."""
        features = {}
        
        # Initialize all features to 0
        for col in KDD99_COLUMNS[:41]:
            features[col] = 0
        
        # Basic packet info
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol_num = packet[IP].proto
            # Track user identity (for behavioural analysis)
            user_id = user_id or src_ip

            features['protocol_type'] = self.protocol_map.get(protocol_num, 'other')
            features['src_bytes'] = len(packet)
            features['dst_bytes'] = len(packet)
            features['land'] = 1 if src_ip == dst_ip else 0
            
            # TCP features
            if TCP in packet:
                tcp = packet[TCP]
                features['service'] = self.service_map.get(tcp.dport, 'other')
                
                # TCP flags (KDD format: string like 'SF', 'S', 'A', etc.)
                flags = tcp.flags
                flag_str = ''
                if flags & 0x02: flag_str += 'S'  # SYN
                if flags & 0x10: flag_str += 'A'  # ACK
                if flags & 0x04: flag_str += 'R'  # RST
                if flags & 0x01: flag_str += 'F'  # FIN
                if flags & 0x08: flag_str += 'P'  # PSH
                if flags & 0x20: flag_str += 'U'  # URG
                features['flag'] = flag_str if flag_str else 'OTH'
                
                # Connection features
                features['urgent'] = tcp.urgptr if hasattr(tcp, 'urgptr') else 0
                features['hot'] = 1 if tcp.flags & 0x20 else 0  # URG flag
                
            elif UDP in packet:
                features['service'] = self.service_map.get(packet[UDP].dport, 'other')
                features['flag'] = 'OTH'  # UDP has no flags
            elif ICMP in packet:
                features['service'] = 'other'
                features['flag'] = 'OTH'
            else:
                features['service'] = 'other'
                features['flag'] = 'OTH'
        else:
            features['protocol_type'] = 'other'
            features['service'] = 'other'
            features['flag'] = 0

        # Attach user_id if known (not part of KDD features, but useful in DataFrame)
        if user_id:
            features['user_id'] = user_id

        return features
    
    def _update_flow_features(self, packet) -> Dict[str, Any]:
        """Update flow-based features from packet."""
        if not (IP in packet and (TCP in packet or UDP in packet)):
            return self._extract_basic_features(packet)
        
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            protocol = 'tcp'
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            protocol = 'udp'
        else:
            return self._extract_basic_features(packet)
        
        # Create flow identifier
        flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        reverse_flow_id = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        # Use the canonical flow ID (smaller IP first)
        canonical_flow_id = min(flow_id, reverse_flow_id)
        
        # Use src_ip as the primary "user id" for behavioural tracking
        user_id = src_ip
        features = self._extract_basic_features(packet, user_id=user_id)
        
        with self.lock:
            # Update flow statistics
            if canonical_flow_id not in self.flows:
                self.flows[canonical_flow_id] = {
                    'start_time': time.time(),
                    'packet_count': 0,
                    'src_bytes': 0,
                    'dst_bytes': 0,
                    'services': set(),
                    'flags': [],
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'protocol': protocol
                }
            
            flow = self.flows[canonical_flow_id]
            current_time = time.time()
            
            # Update flow data
            flow['packet_count'] += 1
            flow['src_bytes'] += features['src_bytes']
            flow['dst_bytes'] += features['dst_bytes']
            flow['services'].add(features['service'])
            flow['flags'].append(features['flag'])
            flow['duration'] = current_time - flow['start_time']
            
            # Update host statistics
            self.host_stats[src_ip]['connections'] += 1
            self.host_stats[src_ip]['services'].add(features['service'])
            self.host_stats[src_ip]['ports'].add(src_port)
            
            self.host_stats[dst_ip]['connections'] += 1
            self.host_stats[dst_ip]['services'].add(features['service'])
            self.host_stats[dst_ip]['ports'].add(dst_port)
            
            # Calculate derived features
            features['duration'] = flow['duration']
            features['count'] = flow['packet_count']
            features['srv_count'] = len(flow['services'])
            features['same_srv_rate'] = len(flow['services']) / max(flow['packet_count'], 1)
            features['diff_srv_rate'] = 1 - features['same_srv_rate']
            
            # Host-based features (simplified)
            features['dst_host_count'] = self.host_stats[dst_ip]['connections']
            features['dst_host_srv_count'] = len(self.host_stats[dst_ip]['services'])
            features['dst_host_same_srv_rate'] = len(self.host_stats[dst_ip]['services']) / max(self.host_stats[dst_ip]['connections'], 1)
            features['dst_host_diff_srv_rate'] = 1 - features['dst_host_same_srv_rate']
            
            # Error rates (simplified)
            features['serror_rate'] = 0.0  # Would need to track SYN errors
            features['srv_serror_rate'] = 0.0
            features['rerror_rate'] = 0.0  # Would need to track RST errors
            features['srv_rerror_rate'] = 0.0
            features['dst_host_serror_rate'] = 0.0
            features['dst_host_srv_serror_rate'] = 0.0
            features['dst_host_rerror_rate'] = 0.0
            features['dst_host_srv_rerror_rate'] = 0.0

            # Other features (set to default values)
            features['wrong_fragment'] = 0
            features['num_failed_logins'] = 0
            features['logged_in'] = 1 if flow['packet_count'] > 1 else 0
            features['num_compromised'] = 0
            features['root_shell'] = 0
            features['su_attempted'] = 0
            features['num_root'] = 0
            features['num_file_creations'] = 0
            features['num_shells'] = 0
            features['num_access_files'] = 0
            features['num_outbound_cmds'] = 0
            features['is_host_login'] = 0
            features['is_guest_login'] = 0
            features['srv_diff_host_rate'] = 0.0
            features['dst_host_same_src_port_rate'] = 0.0
            features['dst_host_srv_diff_host_rate'] = 0.0

            # Update per-user sequence history for behavioural LSTM
            if user_id:
                # Build vector in the canonical KDD feature order
                vec = [features[col] for col in KDD99_COLUMNS[:41]]
                self.user_sequences[user_id].append(vec)

            # Add flow to history if it's complete (simplified: after 10 packets or 30 seconds)
            if flow['packet_count'] >= 10 or flow['duration'] > 30:
                self.flow_history.append(flow.copy())
                if len(self.flow_history) > self.max_flows:
                    self.flow_history.popleft()
                # Remove from active flows
                del self.flows[canonical_flow_id]
        
        return features
    
    def _packet_handler(self, packet):
        """Handle incoming packets."""
        try:
            features = self._update_flow_features(packet)
            # Store the latest packet features for real-time analysis
            with self.lock:
                self.latest_features = features
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def start(self, iface: Optional[str] = None, bpf_filter: Optional[str] = None):
        """Start packet capture."""
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for packet capture. Install with: pip install scapy")
        
        if self.is_capturing:
            return
        
        self.is_capturing = True
        
        def capture_loop():
            try:
                sniff(
                    iface=iface,
                    filter=bpf_filter,
                    prn=self._packet_handler,
                    stop_filter=lambda x: not self.is_capturing,
                    store=0  # Don't store packets in memory
                )
            except PermissionError as e:
                error_msg = str(e)
                if "bpf" in error_msg.lower() or "permission denied" in error_msg.lower():
                    print(f"\n❌ Permission Error: {error_msg}")
                    print("\n📱 macOS Network Permissions Required:")
                    print("   Option 1 (Recommended): Grant network permissions")
                    print("      1. Open System Settings → Privacy & Security → Network")
                    print("      2. Enable Terminal (or your terminal app)")
                    print("      3. Restart Streamlit")
                    print("\n   Option 2: Use sudo (alternative)")
                    print("      sudo env PATH=\"$PATH\" streamlit run streamlit_app.py")
                    print("\n   Option 3: Use Wireshark method (no permissions needed)")
                    print("      Select 'Wireshark (tshark)' in the Live Capture tab")
                    print("      Install with: brew install wireshark")
                else:
                    print(f"Capture error: {error_msg}")
                self.is_capturing = False
            except Exception as e:
                error_msg = str(e)
                if "bpf" in error_msg.lower() or "permission" in error_msg.lower():
                    print(f"\n❌ Permission Error: {error_msg}")
                    print("\n📱 macOS: Grant network permissions or use sudo")
                    print("   See macOS_SETUP.md for instructions")
                else:
                    print(f"Capture error: {error_msg}")
                self.is_capturing = False
        
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop(self):
        """Stop packet capture."""
        self.is_capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
    
    def dataframe(self) -> pd.DataFrame:
        """Get current flows as a DataFrame."""
        with self.lock:
            if hasattr(self, 'latest_features'):
                # Return single row with latest packet features
                return pd.DataFrame([self.latest_features])
            elif self.flow_history:
                # Return recent flows
                return pd.DataFrame(list(self.flow_history)[-self.window_size:])
            else:
                # Return empty DataFrame with correct columns
                return pd.DataFrame(columns=KDD99_COLUMNS[:41])

    def get_user_sequences(self, min_len: int = 5) -> Dict[str, np.ndarray]:
        """
        Return a mapping of user_id -> sequence matrix (timesteps, features)
        for users that have at least `min_len` steps of history.
        """
        with self.lock:
            out: Dict[str, np.ndarray] = {}
            for user_id, buf in self.user_sequences.items():
                if len(buf) >= min_len:
                    out[user_id] = np.asarray(list(buf), dtype=float)
            return out

def get_featurizer() -> PacketFeaturizer:
    """Get a new packet featurizer instance."""
    return PacketFeaturizer()

if __name__ == "__main__":
    # Test the featurizer
    featurizer = get_featurizer()
    print("Packet featurizer created successfully!")
    print(f"Required columns: {len(KDD99_COLUMNS[:41])}")
    print("Features:", KDD99_COLUMNS[:41][:10], "...")
