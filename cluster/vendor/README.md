# Vendored metrics-server manifest

Unmodified v0.8.1 release asset from
https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.1/components.yaml

SHA-256: `4a672c4891902573a3ff753cece5de1bf1f55dd053403dfec39df9d1636b7ff1`.

The renderer pins the container digest and adds `--kubelet-insecure-tls`
for kind node certificates. This flag is specific to this local test cluster.
Upstream is Apache-2.0; see LICENSE.metrics-server.
