# Use Debian Trixie slim as base for Rust development
FROM debian:trixie-slim

# Install system dependencies needed for Rust development and VS Code devcontainer
# - curl: for downloading Rust installer and general use
# - git: version control
# - build-essential: C compiler and build tools (needed for some Rust crates)
# - pkg-config: helps Rust find system libraries
# - ca-certificates: SSL certificate validation
# - gnupg: GPG key management
# - libssl-dev: OpenSSL development files (commonly needed by Rust projects)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    ca-certificates \
    gnupg \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup (official Rust installer)
# - Install to /usr/local/cargo and /usr/local/rustup for system-wide access
# - Use default stable toolchain
# - Add cargo bin to PATH
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile default \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# Install Node.js LTS (needed for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
# Using --no-fund and --no-audit flags to reduce installation noise
RUN npm install -g @anthropic-ai/claude-code --no-fund --no-audit

# Set the working directory
WORKDIR /workspace

# Verify installations
RUN rustc --version && \
    cargo --version && \
    claude --version

# Pre-create cargo registry directory with proper permissions
# This helps avoid permission issues when first running cargo commands
RUN mkdir -p $CARGO_HOME/registry && chmod -R a+w $CARGO_HOME

# Keep container running for devcontainer usage
CMD ["sleep", "infinity"]