# Set up uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Install Caddy 
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# Install node
curl -o- https://fnm.vercel.app/install | bash

omz reload
fnm install --lts