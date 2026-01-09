import pandas as pd


def main() -> None:
    cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
        "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
    ]

    # Load full KDDTest+ txt (no header)
    df = pd.read_csv(r".\archive\KDDTest+.txt", header=None, names=cols)

    # Keep only normal (benign) traffic
    normal_df = df[df.label.str.contains("normal", case=False, na=False)].copy()

    # Write out full normal-only dataset (no sampling)
    output_path = r".\KDDTest_plus_normal.csv"
    normal_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(normal_df)} rows (normal only)")


if __name__ == "__main__":
    main()


