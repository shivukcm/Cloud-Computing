# Subnet Allocation and Stateless Network Access Control (NACL)

## 1. Subnet Allocation Table

The provided base address block is `10.20.0.0/16`. Subnets are allocated strictly in order from the largest requirement to the smallest, placed contiguously starting at `10.20.0.0`, using the smallest prefix that fits each requirement ($2^{32 - \text{prefix}} - 2 \ge \text{Hosts Needed}$).

| Subnet | Hosts Needed | Network Address | Prefix Length | Usable Host Range | Broadcast Address | Usable Hosts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **app** | 1,000 | `10.20.0.0` | `/22` | `10.20.0.1` – `10.20.3.254` | `10.20.3.255` | 1,022 |
| **public** | 250 | `10.20.4.0` | `/24` | `10.20.4.1` – `10.20.4.254` | `10.20.4.255` | 254 |
| **data** | 100 | `10.20.5.0` | `/25` | `10.20.5.1` – `10.20.5.126` | `10.20.5.127` | 126 |
| **management** | 20 | `10.20.5.128` | `/27` | `10.20.5.129` – `10.20.5.158` | `10.20.5.159` | 30 |

---

## 2. Mathematical Working & Derivations

### Base Network
* CIDR: `10.20.0.0/16`
* Total Address Space: $2^{32 - 16} = 2^{16} = 65,536$ addresses (`10.20.0.0` to `10.20.255.255`).

### Subnet 1: `app` (1,000 hosts required)
* We require $2^h - 2 \ge 1,000$, where $h$ is host bits.
  * For $h = 9$: $2^9 - 2 = 512 - 2 = 510 < 1000$ (insufficient).
  * For $h = 10$: $2^{10} - 2 = 1024 - 2 = 1,022 \ge 1000$ (smallest valid prefix).
* Prefix length: $32 - 10 = 22$ (`/22`).
* Address block size: $2^{10} = 1,024$ IP addresses (spans 4 standard `/24` blocks).
* Start IP: `10.20.0.0`
* Network Address: `10.20.0.0/22`
* Usable Host Range: `10.20.0.1` to `10.20.3.254`
* Broadcast Address: `10.20.3.255`
* Usable Hosts: $1,024 - 2 = 1,022$.
* Next contiguous available IP: `10.20.4.0`.

### Subnet 2: `public` (250 hosts required)
* We require $2^h - 2 \ge 250$.
  * For $h = 7$: $2^7 - 2 = 128 - 2 = 126 < 250$ (insufficient).
  * For $h = 8$: $2^8 - 2 = 256 - 2 = 254 \ge 250$ (smallest valid prefix).
* Prefix length: $32 - 8 = 24$ (`/24`).
* Address block size: $2^8 = 256$ IP addresses.
* Start IP: `10.20.4.0`
* Network Address: `10.20.4.0/24`
* Usable Host Range: `10.20.4.1` to `10.20.4.254`
* Broadcast Address: `10.20.4.255`
* Usable Hosts: $256 - 2 = 254$.
* Next contiguous available IP: `10.20.5.0`.

### Subnet 3: `data` (100 hosts required)
* We require $2^h - 2 \ge 100$.
  * For $h = 6$: $2^6 - 2 = 64 - 2 = 62 < 100$ (insufficient).
  * For $h = 7$: $2^7 - 2 = 128 - 2 = 126 \ge 100$ (smallest valid prefix).
* Prefix length: $32 - 7 = 25$ (`/25`).
* Address block size: $2^7 = 128$ IP addresses.
* Start IP: `10.20.5.0`
* Network Address: `10.20.5.0/25`
* Usable Host Range: `10.20.5.1` to `10.20.5.126`
* Broadcast Address: `10.20.5.127`
* Usable Hosts: $128 - 2 = 126$.
* Next contiguous available IP: `10.20.5.128`.

### Subnet 4: `management` (20 hosts required)
* We require $2^h - 2 \ge 20$.
  * For $h = 4$: $2^4 - 2 = 16 - 2 = 14 < 20$ (insufficient).
  * For $h = 5$: $2^5 - 2 = 32 - 2 = 30 \ge 20$ (smallest valid prefix).
* Prefix length: $32 - 5 = 27$ (`/27`).
* Address block size: $2^5 = 32$ IP addresses.
* Start IP: `10.20.5.128`
* Network Address: `10.20.5.128/27`
* Usable Host Range: `10.20.5.129` to `10.20.5.158`
* Broadcast Address: `10.20.5.159`
* Usable Hosts: $32 - 2 = 30$.
* Remaining unallocated address space in `10.20.0.0/16`: `10.20.5.160` to `10.20.255.255`.

---

## 3. Stateless NACL Reasoning & Architecture

### Why a Stateless NACL Needs Separate Rules for Reply Traffic
Unlike stateful firewalls (such as AWS Security Groups or iptables with `conntrack`), a **Network Access Control List (NACL) is completely stateless**. It evaluates each packet independently in isolation without tracking session states, TCP three-way handshakes, or connection tables.

When an external client initiates an inbound HTTP connection to port 80 or 443, the inbound packet is allowed if an inbound rule permits it. However, when the service sends back the HTTP response, that response packet is evaluated by the **outbound NACL table**. Because the NACL has no memory of the originating inbound request, it does not automatically permit the return packet. Therefore, explicit outbound rules must be defined to allow traffic returning to client ephemeral ports (`1024`–`65535`). Similarly, when the load balancer forwards requests outbound to the backend `app` subnet on port 8000, explicit inbound rules must be configured to permit the app instances' return traffic arriving on the load balancer's ephemeral ports (`1024`–`65535`). Without explicit bidirectional rules, the connection breaks immediately.

### Why Rule Order Matters for the `198.51.100.0/24` Block
NACL evaluation follows a **first-match wins** strategy, processing rules sequentially from the lowest rule number to the highest rule number. Once a packet matches a rule's criteria (protocol, port range, CIDR), the designated action (`allow` or `deny`) is applied immediately, and subsequent rules are skipped.

In this design, we must block all traffic from the malicious range `198.51.100.0/24`, even on legitimate web ports such as 80 and 443:
1. **Rule 100** explicitly denies all traffic from `198.51.100.0/24`.
2. **Rules 110 and 120** allow TCP ports 80 and 443 from anywhere (`0.0.0.0/0`).

Because `198.51.100.0/24` is a subset of `0.0.0.0/0`, placing Rule 100 before Rules 110 and 120 ensures that any packet originating from the abusive block is matched and dropped at Rule 100. If the order were reversed (e.g., allow `0.0.0.0/0` on port 443 at Rule 100, and deny `198.51.100.0/24` at Rule 120), malicious traffic directed at port 443 would match the broad allow rule first, rendering the deny rule completely ineffective.
