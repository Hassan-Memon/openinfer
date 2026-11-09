# Deploy to Oracle Cloud Always Free (ARM, llama.cpp)

Oracle's Always Free tier includes up to **4 Ampere A1 OCPUs and 24 GB RAM** (one VM or split
across several), 200 GB block storage and 10 TB egress per month, at no charge and with no time
limit. It is the only free tier where a real 1B to 8B model is comfortable, which makes it the
reference deployment for the `private` alias: your data, your VM, no vendor GPU.

Ampere A1 is ARM64 (aarch64) with NEON and dot-product instructions. llama.cpp is well optimised
for it; expect roughly 10 to 15 tokens/s on a 1.5B Q4_K_M model and 3 to 5 tokens/s on 7B Q4_K_M
with all 4 cores.

## 1. Create the VM

1. Oracle Cloud console, Compute, Instances, Create.
2. Image: **Ubuntu 24.04 (aarch64)**. Shape: **VM.Standard.A1.Flex**, 4 OCPU, 24 GB.
3. Add your SSH public key. Create.
4. Networking: in the VCN's security list add an ingress rule for TCP 443 (and 80 for the ACME challenge). Also open the OS firewall:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

Capacity note: A1 shapes are popular and "Out of host capacity" errors are common. Retry at a
different hour or in another availability domain; a small script that retries the create call
is a common workaround.

## 2. Install

```bash
sudo apt update && sudo apt install -y build-essential cmake git
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/Hassan-Memon/openinfer && cd openinfer
uv sync --extra llamacpp --extra metrics      # compiles llama.cpp for aarch64 (5 to 10 min)
uv run openinfer hardware                     # should show NEON / asimd and "cpu" recommendation
uv run openinfer download-model               # Qwen2.5-0.5B to start; swap for a bigger GGUF later
```

For a stronger model, for example Qwen2.5-3B-Instruct Q4_K_M (about 2 GB):

```bash
uv run openinfer download-model --repo Qwen/Qwen2.5-3B-Instruct-GGUF --filename qwen2.5-3b-instruct-q4_k_m.gguf
```

## 3. Configure

`openinfer.yaml`:

```yaml
backends:
  llamacpp:
    type: llamacpp
    enabled: true
    model_path: models/qwen2.5-3b-instruct-q4_k_m.gguf
    models: [qwen2.5-3b-instruct]
    n_ctx: 8192
    n_threads: 4
    n_gpu_layers: 0
aliases:
  private: [llamacpp/qwen2.5-3b-instruct]
  fast: [groq/llama-3.1-8b-instant, llamacpp/qwen2.5-3b-instruct]
```

`.env`: `OPENINFER_API_KEY=<random>`, optional cloud keys.

## 4. Run as a service

`/etc/systemd/system/openinfer.service`:

```ini
[Unit]
Description=OpenInfer
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/openinfer
EnvironmentFile=/home/ubuntu/openinfer/.env
ExecStart=/home/ubuntu/.local/bin/uv run openinfer serve --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now openinfer
```

## 5. TLS with Caddy

```bash
sudo apt install -y caddy
echo 'infer.example.com {
    reverse_proxy 127.0.0.1:8000
}' | sudo tee /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy obtains and renews a Let's Encrypt certificate. Point a DNS A record at the VM's public IP
first. Without a domain, use `:443` with `tls internal` for a self-signed cert, or keep it behind
Tailscale/WireGuard, which is a good fit for "private" anyway.

## 6. Verify

```bash
curl https://infer.example.com/v1/hardware -H "Authorization: Bearer $KEY" | jq .cpu.features
uv run openinfer bench --backend llamacpp --runs 3
```

## Docker alternative

The repo's `Dockerfile` builds for arm64 as is (`docker compose up`), but it does not include
the llamacpp extra. Add `--extra llamacpp` to both `uv sync` lines and `cmake` plus
`build-essential` to the builder stage, then `docker compose --profile ollama up` if you prefer
Ollama's model management over GGUF files.
