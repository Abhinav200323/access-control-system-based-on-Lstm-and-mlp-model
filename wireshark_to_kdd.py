"""
Convert Wireshark PCAP files to KDD-style CSV format for training behavioral LSTM.

This module processes PCAP files captured from Wireshark and extracts KDD99 features
for each network flow, enabling training on real network traffic data.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP
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

# Protocol mappings
PROTOCOL_MAP = {
    1: 'icmp', 6: 'tcp', 17: 'udp'
}

# Service port mappings
SERVICE_MAP = {
    21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'domain',
    80: 'http', 110: 'pop_3', 143: 'imap4', 443: 'https', 993: 'imaps', 995: 'pop3s',
    20: 'ftp_data', 69: 'tftp', 123: 'ntp', 161: 'snmp', 179: 'bgp',
    389: 'ldap', 636: 'ldaps', 1433: 'sql_server', 3306: 'mysql', 5432: 'postgresql'
}

# TCP flag mappings
TCP_FLAG_MAP = {
    'F': 1, 'S': 2, 'R': 4, 'P': 8, 'A': 16, 'U': 32, 'E': 64, 'C': 128
}


def get_tcp_flags(tcp_pkt) -> str:
    """Extract TCP flags as string."""
    flags = []
    if tcp_pkt.flags & 0x01: flags.append('F')  # FIN
    if tcp_pkt.flags & 0x02: flags.append('S')  # SYN
    if tcp_pkt.flags & 0x04: flags.append('R')  # RST
    if tcp_pkt.flags & 0x08: flags.append('P')  # PSH
    if tcp_pkt.flags & 0x10: flags.append('A')  # ACK
    if tcp_pkt.flags & 0x20: flags.append('U')  # URG
    if tcp_pkt.flags & 0x40: flags.append('E')  # ECE
    if tcp_pkt.flags & 0x80: flags.append('C')  # CWR
    return ''.join(flags) if flags else 'OTH'


def get_service_name(port: int, protocol: str) -> str:
    """Get service name from port number."""
    return SERVICE_MAP.get(port, 'other')


def extract_flow_features(
    packets: List,
    flow_key: str,
    flow_packets: List,
    host_stats: Dict,
    time_window: float = 2.0
) -> Dict[str, Any]:
    """
    Extract KDD features from a network flow (connection).
    
    Args:
        packets: All packets in the capture
        flow_key: Unique identifier for this flow (e.g., "src_ip:src_port-dst_ip:dst_port")
        flow_packets: Packets belonging to this flow
        host_stats: Statistics about hosts (for destination host features)
        time_window: Time window for computing statistical features
    
    Returns:
        Dictionary with KDD features
    """
    if not flow_packets:
        return None
    
    features = {col: 0 for col in KDD99_COLUMNS}
    
    # Get first and last packet timestamps
    first_pkt = flow_packets[0]
    last_pkt = flow_packets[-1]
    duration = float(last_pkt.time - first_pkt.time) if len(flow_packets) > 1 else 0.0
    features['duration'] = duration
    
    # Extract IP layer info
    if IP not in first_pkt:
        return None
    
    src_ip = first_pkt[IP].src
    dst_ip = first_pkt[IP].dst
    protocol_num = first_pkt[IP].proto
    features['protocol_type'] = PROTOCOL_MAP.get(protocol_num, 'other')
    features['land'] = 1 if src_ip == dst_ip else 0
    
    # Extract transport layer info
    src_port = 0
    dst_port = 0
    service = 'other'
    flag = 'OTH'
    
    if TCP in first_pkt:
        tcp = first_pkt[TCP]
        src_port = tcp.sport
        dst_port = tcp.dport
        service = get_service_name(dst_port, 'tcp')
        flag = get_tcp_flags(tcp)
        features['urgent'] = tcp.urgptr if hasattr(tcp, 'urgptr') else 0
    elif UDP in first_pkt:
        udp = first_pkt[UDP]
        src_port = udp.sport
        dst_port = udp.dport
        service = get_service_name(dst_port, 'udp')
        flag = 'OTH'
    elif ICMP in first_pkt:
        icmp = first_pkt[ICMP]
        service = 'eco_i'  # ICMP echo
        flag = 'OTH'
    
    features['service'] = service
    features['flag'] = flag
    
    # Compute byte statistics
    src_bytes = sum(len(pkt) for pkt in flow_packets if IP in pkt and pkt[IP].src == src_ip)
    dst_bytes = sum(len(pkt) for pkt in flow_packets if IP in pkt and pkt[IP].dst == src_ip)
    features['src_bytes'] = src_bytes
    features['dst_bytes'] = dst_bytes
    
    # Count packets in this flow
    features['count'] = len(flow_packets)
    
    # Compute same service count (flows with same service in time window)
    same_service_count = 0
    for pkt in packets:
        if IP in pkt and abs(float(pkt.time - first_pkt.time)) <= time_window:
            if TCP in pkt and pkt[TCP].dport == dst_port:
                same_service_count += 1
            elif UDP in pkt and pkt[UDP].dport == dst_port:
                same_service_count += 1
    features['srv_count'] = same_service_count
    
    # Error rates (simplified - count RST flags for TCP)
    serror_count = 0
    rerror_count = 0
    if TCP in first_pkt:
        for pkt in flow_packets:
            if TCP in pkt:
                flags = get_tcp_flags(pkt[TCP])
                if 'R' in flags or 'S' in flags:  # SYN or RST
                    serror_count += 1
                if 'R' in flags:
                    rerror_count += 1
    
    features['serror_rate'] = serror_count / max(len(flow_packets), 1)
    features['srv_serror_rate'] = features['serror_rate']  # Simplified
    features['rerror_rate'] = rerror_count / max(len(flow_packets), 1)
    features['srv_rerror_rate'] = features['rerror_rate']  # Simplified
    
    # Same service rate
    features['same_srv_rate'] = 1.0 if same_service_count > 0 else 0.0
    
    # Different service rate (simplified)
    features['diff_srv_rate'] = 1.0 - features['same_srv_rate']
    
    # Destination host statistics
    if dst_ip not in host_stats:
        host_stats[dst_ip] = {
            'connections': 0,
            'services': set(),
            'ports': set(),
            'errors': 0
        }
    
    host_stats[dst_ip]['connections'] += 1
    host_stats[dst_ip]['services'].add(service)
    host_stats[dst_ip]['ports'].add(dst_port)
    if rerror_count > 0:
        host_stats[dst_ip]['errors'] += 1
    
    # Destination host features
    features['dst_host_count'] = host_stats[dst_ip]['connections']
    features['dst_host_srv_count'] = len(host_stats[dst_ip]['services'])
    features['dst_host_same_srv_rate'] = 1.0 / max(host_stats[dst_ip]['connections'], 1)
    features['dst_host_diff_srv_rate'] = 1.0 - features['dst_host_same_srv_rate']
    features['dst_host_same_src_port_rate'] = 1.0 / max(len(host_stats[dst_ip]['ports']), 1)
    features['dst_host_srv_diff_host_rate'] = 0.0  # Simplified
    features['dst_host_serror_rate'] = host_stats[dst_ip]['errors'] / max(host_stats[dst_ip]['connections'], 1)
    features['dst_host_srv_serror_rate'] = features['dst_host_serror_rate']
    features['dst_host_rerror_rate'] = features['rerror_rate']  # Simplified
    features['dst_host_srv_rerror_rate'] = features['dst_host_rerror_rate']
    
    # Features that are typically 0 for network traffic (host-based features)
    features['wrong_fragment'] = 0
    features['hot'] = 0
    features['num_failed_logins'] = 0
    features['logged_in'] = 0
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
    features['srv_diff_host_rate'] = 0.0  # Simplified
    
    # Add metadata
    features['src_ip'] = src_ip
    features['dst_ip'] = dst_ip
    features['src_port'] = src_port
    features['dst_port'] = dst_port
    features['timestamp'] = float(first_pkt.time)
    
    return features


def pcap_to_kdd_csv(
    pcap_path: str,
    output_csv: str,
    time_window: float = 2.0,
    min_packets_per_flow: int = 1
) -> str:
    """
    Convert Wireshark PCAP file to KDD-style CSV.
    
    Args:
        pcap_path: Path to input PCAP file
        output_csv: Path to output CSV file
        time_window: Time window for grouping packets into flows (seconds)
        min_packets_per_flow: Minimum packets required to create a flow
    
    Returns:
        Path to output CSV file
    """
    if not SCAPY_AVAILABLE:
        raise ImportError("Scapy is required. Install with: pip install scapy")
    
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
    
    print(f"Reading PCAP file: {pcap_path}")
    packets = rdpcap(pcap_path)
    print(f"Loaded {len(packets)} packets")
    
    # Group packets into flows
    # Flow key: "src_ip:src_port-dst_ip:dst_port-protocol"
    flows: Dict[str, List] = defaultdict(list)
    
    for pkt in packets:
        if IP not in pkt:
            continue
        
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        protocol = PROTOCOL_MAP.get(pkt[IP].proto, 'other')
        
        src_port = 0
        dst_port = 0
        
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        
        # Create bidirectional flow key (smaller IP:port first)
        if src_ip < dst_ip or (src_ip == dst_ip and src_port < dst_port):
            flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        flows[flow_key].append(pkt)
    
    print(f"Identified {len(flows)} flows")
    
    # Sort packets in each flow by timestamp
    for flow_key in flows:
        flows[flow_key].sort(key=lambda p: float(p.time))
    
    # Extract features for each flow
    host_stats = defaultdict(lambda: {
        'connections': 0,
        'services': set(),
        'ports': set(),
        'errors': 0
    })
    
    flow_features = []
    for flow_key, flow_packets in flows.items():
        if len(flow_packets) < min_packets_per_flow:
            continue
        
        features = extract_flow_features(
            packets=list(packets),
            flow_key=flow_key,
            flow_packets=flow_packets,
            host_stats=host_stats,
            time_window=time_window
        )
        
        if features:
            flow_features.append(features)
    
    if not flow_features:
        raise ValueError("No valid flows extracted from PCAP file")
    
    # Create DataFrame
    df = pd.DataFrame(flow_features)
    
    # Reorder columns: KDD features first, then metadata
    kdd_cols = [col for col in KDD99_COLUMNS if col in df.columns]
    meta_cols = [col for col in df.columns if col not in KDD99_COLUMNS]
    df = df[kdd_cols + meta_cols]
    
    # Add label column (default to 'normal' - user can update later)
    if 'label' not in df.columns:
        df['label'] = 'normal'
    
    # Add user_id column (using src_ip for behavioral analysis)
    if 'user_id' not in df.columns and 'src_ip' in df.columns:
        df['user_id'] = df['src_ip'].astype(str)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} flows to {output_csv}")
    print(f"Columns: {list(df.columns)}")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Convert Wireshark PCAP file to KDD-style CSV for training"
    )
    parser.add_argument(
        "--pcap",
        type=str,
        required=True,
        help="Path to input PCAP file (from Wireshark)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wireshark_capture.csv",
        help="Path to output CSV file (default: wireshark_capture.csv)"
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=2.0,
        help="Time window for flow grouping in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--min_packets",
        type=int,
        default=1,
        help="Minimum packets per flow (default: 1)"
    )
    args = parser.parse_args()
    
    try:
        output_path = pcap_to_kdd_csv(
            pcap_path=args.pcap,
            output_csv=args.output,
            time_window=args.time_window,
            min_packets_per_flow=args.min_packets
        )
        print(f"\n✅ Successfully converted PCAP to CSV: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Review the CSV file and update 'label' column if needed")
        print(f"  2. Build sequences: python build_beh_sequences.py --csv {output_path}")
        print(f"  3. Train behavioral LSTM: python train_beh_lstm.py --x X_seq.npy --y y_seq.npy")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

