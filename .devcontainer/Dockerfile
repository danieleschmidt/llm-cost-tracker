# Development container for LLM Cost Tracker
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Set environment variables
ENV POETRY_HOME="/opt/poetry" \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VENV_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH=/workspace/src \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    curl \
    wget \
    git \
    # Database clients
    postgresql-client \
    redis-tools \
    # Network tools
    netcat \
    telnet \
    # Development tools
    vim \
    htop \
    tree \
    jq \
    # Fonts for better terminal experience
    fonts-powerline \
    # Security tools
    gnupg2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Node.js and npm (for semantic-release and other tools)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install global npm packages for semantic release
RUN npm install -g \
    @semantic-release/changelog \
    @semantic-release/commit-analyzer \
    @semantic-release/exec \
    @semantic-release/git \
    @semantic-release/github \
    @semantic-release/release-notes-generator \
    conventional-changelog-conventionalcommits \
    semantic-release

# Install development tools
RUN pip install --user \
    # SBOM generation tools
    cyclonedx-bom \
    # Security scanning
    bandit \
    safety \
    # Code quality
    pre-commit \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    # Jupyter for notebooks
    jupyter \
    jupyterlab

# Install Syft and Grype for SBOM generation and vulnerability scanning
RUN curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin \
    && curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Install Cosign for artifact signing
RUN curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64" \
    && mv cosign-linux-amd64 /usr/local/bin/cosign \
    && chmod +x /usr/local/bin/cosign

# Configure shell
RUN chsh -s /bin/zsh vscode

# Set up workspace directory
WORKDIR /workspace

# Create directories for development
RUN mkdir -p /workspace/{logs,sbom,temp,data} \
    && chown -R vscode:vscode /workspace

# Switch to vscode user
USER vscode

# Install Oh My Zsh for better shell experience
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Configure Git for the development environment
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global user.name "VS Code Dev Container" \
    && git config --global user.email "dev@localhost"

# Copy shell configuration files
COPY --chown=vscode:vscode .devcontainer/zshrc /home/vscode/.zshrc
COPY --chown=vscode:vscode .devcontainer/bashrc /home/vscode/.bashrc

# Create useful aliases
RUN echo 'alias ll="ls -alF"' >> /home/vscode/.zshrc \
    && echo 'alias la="ls -A"' >> /home/vscode/.zshrc \
    && echo 'alias l="ls -CF"' >> /home/vscode/.zshrc \
    && echo 'alias ..="cd .."' >> /home/vscode/.zshrc \
    && echo 'alias ...="cd ../.."' >> /home/vscode/.zshrc \
    && echo 'alias grep="grep --color=auto"' >> /home/vscode/.zshrc \
    && echo 'alias fgrep="fgrep --color=auto"' >> /home/vscode/.zshrc \
    && echo 'alias egrep="egrep --color=auto"' >> /home/vscode/.zshrc

# Set up development shortcuts
RUN echo 'alias start-dev="make dev"' >> /home/vscode/.zshrc \
    && echo 'alias run-tests="make test"' >> /home/vscode/.zshrc \
    && echo 'alias check-quality="make quality"' >> /home/vscode/.zshrc \
    && echo 'alias build-app="make build"' >> /home/vscode/.zshrc \
    && echo 'alias clean-up="make clean"' >> /home/vscode/.zshrc

# Expose common ports
EXPOSE 8000 3000 5432 9090 6379 8080 8501

# Default command
CMD ["/bin/zsh"]