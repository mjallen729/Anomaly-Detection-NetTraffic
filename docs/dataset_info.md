## In-Use
### KDDTrain+.CSV 
The full NSL-KDD train set including attack-type labels and difficulty level in CSV format.

### KDDTest+.TXT 
The full NSL-KDD test set including attack-type labels and difficulty level in CSV format

## Column dictionary
**duration** - Length of the connection in seconds.

**protocol_type** - Type of protocol used (tcp, udp, or icmp).

**service** - Network service on the destination (e.g., http, ftp, smtp).

**flag** - Status of the connection (e.g., SF for normal, REJ for rejected).

**src_bytes** - Number of data bytes sent from source to destination (important for detecting data exfiltration).

**dst_bytes** - Number of data bytes sent from destination to source (important for detecting data exfiltration).

**land** - 1 if connection is from/to the same host/port, 0 otherwise (potential attack indicator).

**wrong_fragment** - Number of wrong fragments in the connection.

**urgent** - Number of urgent packets in the connection.

**hot** - Number of "hot" indicators (accessing system files, creating programs, etc.).

**num_failed_logins** - Number of failed login attempts (important for detecting brute force attacks).

**logged_in** - 1 if successfully logged in, 0 otherwise.

**num_compromised** - Number of compromised conditions detected.

**root_shell** - 1 if root shell is obtained, 0 otherwise (critical attack indicator).

**su_attempted** - 1 if "su root" command attempted, 0 otherwise.

**num_root** - Number of root accesses.

**num_file_creations** - Number of file creation operations.

**num_shells** - Number of shell prompts.

**num_access_files** - Number of operations on access control files.

**num_outbound_cmds** - Number of outbound commands in an FTP session.

**is_host_login** - 1 if the login belongs to the host list, 0 otherwise.

**is_guest_login** - 1 if the login is a guest login, 0 otherwise.

**count** - Number of connections to the same host as the current connection in the past 2 seconds (traffic pattern feature).

**srv_count** - Number of connections to the same service in the past 2 seconds (traffic pattern feature).

**serror_rate** - Percentage of connections with SYN errors (important for detecting DOS attacks).

**srv_serror_rate** - Percentage of connections with SYN errors for the same service.

**rerror_rate** - Percentage of connections with REJ errors.

**srv_rerror_rate** - Percentage of connections with REJ errors for the same service.

**same_srv_rate** - Percentage of connections to the same service.

**diff_srv_rate** - Percentage of connections to different services.

**srv_diff_host_rate** - Percentage of connections to different hosts for the same service.

**dst_host_count** - Number of connections having the same destination host in the past 100 connections.

**dst_host_srv_count** - Number of connections having the same destination host and service.

**dst_host_same_srv_rate** - Percentage of connections to the same service on the destination host.

**dst_host_diff_srv_rate** - Percentage of connections to different services on the destination host.

**dst_host_same_src_port_rate** - Percentage of connections to the same source port on the destination host.

**dst_host_srv_diff_host_rate** - Percentage of connections to different hosts for the same service on the destination host.

**dst_host_serror_rate** - Percentage of connections with SYN errors on the destination host.

**dst_host_srv_serror_rate** - Percentage of connections with SYN errors for the same service on the destination host.

**dst_host_rerror_rate** - Percentage of connections with REJ errors on the destination host.

**dst_host_srv_rerror_rate** - Percentage of connections with REJ errors for the same service on the destination host.

**🎯 attack_type** - **LABEL COLUMN** - The type of network attack (e.g., "normal", "neptune", "smurf", "portsweep") or normal traffic

**difficulty_level** - Difficulty level of detecting the attack instance (only present in some NSL-KDD files, can be removed if your file has 42 columns instead of 43).