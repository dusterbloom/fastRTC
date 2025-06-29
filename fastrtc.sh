#!/bin/bash
# FastRTC - Universal Deployment Script
# ====================================
# Single command for all deployment modes: dev, docker, prod
# Usage: ./fastrtc.sh [dev|docker|prod] [options]

set -e

# Script constants
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="FastRTC"
readonly VERSION="1.0.0"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Process tracking
BACKEND_PID=""
FRONTEND_PID=""
DOCKER_RUNNING=false

# Command line flag tracking
CMDLINE_THREADING_PIPELINE=""
CMDLINE_THREADING_FALLBACK=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Show usage information
show_usage() {
    cat << EOF
$PROJECT_NAME v$VERSION - Universal Deployment Script

USAGE:
    ./fastrtc.sh <mode> [options]

MODES:
    dev         Local development (Python backend + Node frontend)
    docker      Docker testing (Linux/macOS)
    prod        Production deployment (Docker with external IP)

OPTIONS:
    --log-level LEVEL    Set logging level (DEBUG, INFO, WARNING, ERROR)
    --threading          Enable threading pipeline (experimental)
    --no-fallback        Disable fallback to async pipeline
    --help, -h           Show this help message

EXAMPLES:
    ./fastrtc.sh dev                           # Start local development
    ./fastrtc.sh dev --log-level INFO          # Development with INFO logging
    ./fastrtc.sh dev --threading               # Development with threading pipeline
    ./fastrtc.sh dev --threading --no-fallback # Threading without async fallback
    ./fastrtc.sh docker                        # Start Docker services
    EXTERNAL_IP=1.2.3.4 ./fastrtc.sh prod     # Production with external IP
    OLLAMA_URL=http://custom:11434 ./fastrtc.sh dev  # Custom service URL

ENVIRONMENT:
    The script loads environment variables from:
    - .env.development (dev mode)
    - .env.docker (docker mode)  
    - .env.production (prod mode)
    - .env.local (user overrides, optional)

LOG LEVELS:
    DEBUG       Show all technical details, model loading, etc.
    INFO        Show essential status and prominent conversation display
    WARNING     Show only warnings and errors
    ERROR       Show only errors

SHORTCUTS:
    Ctrl+C      Graceful shutdown
    kill -TERM  Graceful shutdown with cleanup

EOF
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported OS: $OSTYPE"
        log_info "This script supports Linux and macOS only"
        exit 1
    fi
    log_info "Detected OS: $OS"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies for development mode
check_dev_dependencies() {
    log_step "Checking development dependencies..."
    
    if ! command_exists python3; then
        log_error "Python 3 is required for development mode"
        log_info "Please install Python 3.8+ from https://python.org/downloads"
        exit 1
    fi
    
    if ! command_exists node; then
        log_error "Node.js is required for development mode"
        log_info "Please install Node.js 18+ from https://nodejs.org"
        exit 1
    fi
    
    if ! command_exists npm; then
        log_error "npm is required for development mode"
        log_info "npm comes with Node.js. Please reinstall Node.js"
        exit 1
    fi
    
    log_success "Development dependencies verified"
}

# Check dependencies for Docker mode
check_docker_dependencies() {
    log_step "Checking Docker dependencies..."
    
    if ! command_exists docker; then
        log_error "Docker is required for Docker mode"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        log_error "Docker Compose is required for Docker mode"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        log_info "Please start Docker and try again"
        exit 1
    fi
    
    log_success "Docker dependencies verified"
}

# Create default environment files if missing
create_default_env_files() {
    local env_files=(".env.development" ".env.docker" ".env.production")
    
    for env_file in "${env_files[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$env_file" ]]; then
            log_info "Creating default $env_file..."
            cat > "$SCRIPT_DIR/$env_file" << EOF
# FastRTC Environment Configuration
# Generated automatically - modify as needed

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Backend settings
BACKEND_HOST=localhost
BACKEND_PORT=8000

# Frontend settings  
FRONTEND_HOST=localhost
FRONTEND_PORT=3001

# LLM settings
OLLAMA_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b

# Redis settings (optional)
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# Threading pipeline settings (experimental)
USE_THREADING_PIPELINE=false
THREADING_FALLBACK_TO_ASYNC=true
THREADING_MAX_QUEUE_SIZE=100

# Add any custom environment variables below
EOF
            log_success "Created $env_file with default values"
        fi
    done
}

# Load environment variables from file
load_env_file() {
    local env_file="$1"
    
    if [[ -f "$env_file" ]]; then
        log_info "Loading environment from: $env_file"
        # Export variables from env file
        set -a
        source "$env_file"
        set +a
    else
        log_warn "Environment file not found: $env_file"
    fi
}

# Setup environment for the specified mode
setup_environment() {
    local mode="$1"
    
    log_step "Setting up environment for $mode mode..."
    
    # Create default environment files if missing
    create_default_env_files
    
    # Load mode-specific environment file
    case "$mode" in
        "dev")
            load_env_file "$SCRIPT_DIR/.env.development"
            ;;
        "docker")
            load_env_file "$SCRIPT_DIR/.env.docker"
            ;;
        "prod")
            load_env_file "$SCRIPT_DIR/.env.production"
            ;;
    esac
    
    # Load user overrides if present
    load_env_file "$SCRIPT_DIR/.env.local"
    
    log_success "Environment setup complete"
}

# Get external IP address
get_external_ip() {
    local ip=""
    
    # Try multiple services
    ip=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || \
         curl -s --max-time 5 ipinfo.io/ip 2>/dev/null || \
         curl -s --max-time 5 icanhazip.com 2>/dev/null || \
         echo "")
    
    if [[ -n "$ip" ]]; then
        echo "$ip"
    else
        log_warn "Could not detect external IP"
        echo "localhost"
    fi
}

# Health check for services
health_check() {
    local service="$1"
    local url="$2"
    local max_attempts=999999999 # Effectively infinite attempts
    local attempt=0
    
    log_step "Waiting for $service to be ready..."
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s --max-time 3 "$url" >/dev/null 2>&1; then
            log_success "$service is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo ""
    log_error "$service failed to start after $((max_attempts * 2)) seconds"
    return 1
}

# Cleanup function
cleanup() {
    log_step "Cleaning up..."
    
    # Kill background processes
    if [[ -n "$BACKEND_PID" ]]; then
        log_info "Stopping backend (PID: $BACKEND_PID)"
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
    
    if [[ -n "$FRONTEND_PID" ]]; then
        log_info "Stopping frontend (PID: $FRONTEND_PID)"
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    
    # Stop Docker if running
    if [[ "$DOCKER_RUNNING" == "true" ]]; then
        log_info "Stopping Docker services..."
        docker-compose down >/dev/null 2>&1 || true
    fi
    
    log_success "Cleanup complete"
}

# Signal handlers
trap cleanup EXIT INT TERM

# Development mode implementation
run_development() {
    log_step "Starting development mode..."
    
    check_dev_dependencies
    setup_environment "dev"
    
    # Re-apply command line overrides after loading environment files
    # This ensures command line flags take precedence over environment files
    echo "DEBUG: CMDLINE_THREADING_PIPELINE=$CMDLINE_THREADING_PIPELINE"
    echo "DEBUG: Current USE_THREADING_PIPELINE=$USE_THREADING_PIPELINE"
    if [[ "$CMDLINE_THREADING_PIPELINE" == "true" ]]; then
        export USE_THREADING_PIPELINE=true
        echo "DEBUG: Set USE_THREADING_PIPELINE=true"
        log_info "Command line override: Threading pipeline enabled"
    fi
    if [[ "$CMDLINE_THREADING_FALLBACK" == "false" ]]; then
        export THREADING_FALLBACK_TO_ASYNC=false
        log_info "Command line override: Threading fallback disabled"
    fi
    echo "DEBUG: Final USE_THREADING_PIPELINE=$USE_THREADING_PIPELINE"
    
    # Check if virtual environment exists and activate it
    if [[ -d "$SCRIPT_DIR/backend/venv" ]]; then
        log_info "Activating Python virtual environment..."
        source "$SCRIPT_DIR/backend/venv/bin/activate"
        log_success "Virtual environment activated: $SCRIPT_DIR/backend/venv"
        log_info "Python path: $(which python3)"
        
        # Check if requirements are installed
        log_info "Checking required packages installation..."
        if python3 -c "import uvicorn, fastapi, numpy, torch" 2>/dev/null; then
            log_success "Core packages are installed"
        else
            log_warn "Some packages missing from virtual environment"
            log_info "Installing requirements..."
            cd "$SCRIPT_DIR/backend"
            pip install -r requirements.txt
            pip install --no-deps fastrtc[vad,stt,tts]==0.0.28 
            cd "$SCRIPT_DIR"
            log_success "Requirements installed"
        fi
    else
        log_warn "No virtual environment found at backend/venv"
        log_info "Creating virtual environment..."
        cd "$SCRIPT_DIR/backend"
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        cd "$SCRIPT_DIR"
        log_success "Virtual environment created and requirements installed"
    fi
    
    # Start backend
    log_info "Starting backend server..."
    cd "$SCRIPT_DIR/backend"
    
    # Build command with threading arguments if set
    cmd="python3 start_deferred.py"
    if [[ "${USE_THREADING_PIPELINE:-false}" == "true" ]]; then
        cmd="$cmd --threading"
        log_info "Threading pipeline enabled"
    fi
    if [[ "${THREADING_FALLBACK_TO_ASYNC:-true}" == "false" ]]; then
        cmd="$cmd --no-fallback"
        log_info "Threading fallback disabled"
    fi
    
    # Debug: Show environment variable values
    log_info "Debug: USE_THREADING_PIPELINE=${USE_THREADING_PIPELINE:-false}"
    log_info "Debug: THREADING_FALLBACK_TO_ASYNC=${THREADING_FALLBACK_TO_ASYNC:-false}"
    
    log_info "Starting backend with: $cmd"
    $cmd &
    BACKEND_PID=$!
    cd "$SCRIPT_DIR"
    
    # Wait for backend to be ready
    if ! health_check "Backend" "http://localhost:8000/health"; then
        log_error "Backend failed to start"
        exit 1
    fi
    
    # Check and install frontend dependencies
    log_info "Checking frontend dependencies..."
    cd "$SCRIPT_DIR/frontend/react-vite"
    if [[ ! -d "node_modules" ]]; then
        log_info "Installing frontend dependencies..."
        npm install
        log_success "Frontend dependencies installed"
    fi
    
    # Start frontend
    log_info "Starting frontend server..."
    # Set port explicitly to avoid conflicts
    PORT=3001 npm run dev &
    FRONTEND_PID=$!
    cd "$SCRIPT_DIR"
    
    # Wait for frontend to be ready
    if ! health_check "Frontend" "http://localhost:3001"; then
        log_error "Frontend failed to start"
        exit 1
    fi
    
    # Wait for all backend components to fully initialize
    log_step "Waiting for all components to fully initialize..."
    sleep 3  # Give VAD and other components time to warm up
    
    # Final readiness check
    if curl -s --max-time 3 "http://localhost:8000/health" | grep -q "healthy" 2>/dev/null; then
        log_success "All components fully initialized!"
    else
        log_warn "Components may still be initializing in background"
    fi
    
    log_success "Development mode started successfully!"
    echo ""
    echo "🚀 FastRTC Development Mode"
    echo "📱 Frontend: http://localhost:3001"
    echo "🔧 Backend API: http://localhost:8000"
    echo "❤️  Health Check: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop..."
    
    # Wait for processes
    wait
}

# Docker mode implementation
run_docker() {
    log_step "Starting Docker mode..."
    
    check_docker_dependencies
    setup_environment "docker"
    
    # Stop any existing containers
    log_info "Stopping existing containers..."
    docker-compose down >/dev/null 2>&1 || true
    
    # Start Docker services
    log_info "Starting Docker services..."
    docker-compose up -d
    DOCKER_RUNNING=true
    
    # Wait for backend to be ready
    if ! health_check "Backend" "http://localhost:8000/health"; then
        log_error "Backend failed to start"
        exit 1
    fi
    
    # Wait for frontend to be ready
    if ! health_check "Frontend" "http://localhost:3001"; then
        log_error "Frontend failed to start"
        exit 1
    fi
    
    log_success "Docker mode started successfully!"
    echo ""
    echo "🐳 FastRTC Docker Mode"
    echo "📱 Frontend: http://localhost:3001"
    echo "🔧 Backend API: http://localhost:8000"
    echo "❤️  Health Check: http://localhost:8000/health"
    echo ""
    echo "View logs: docker-compose logs -f"
    echo "Press Ctrl+C to stop..."
    
    # Follow logs
    docker-compose logs -f
}

# Production mode implementation
run_production() {
    log_step "Starting production mode..."
    
    check_docker_dependencies
    setup_environment "prod"
    
    # Get external IP if not provided
    if [[ -z "$EXTERNAL_IP" ]]; then
        EXTERNAL_IP=$(get_external_ip)
        log_info "Detected external IP: $EXTERNAL_IP"
    else
        log_info "Using provided external IP: $EXTERNAL_IP"
    fi
    
    # Export for Docker Compose
    export EXTERNAL_IP
    
    # Generate TURN secret if not provided
    if [[ -z "$TURN_AUTH_SECRET" ]]; then
        TURN_AUTH_SECRET=$(openssl rand -hex 16)
        log_info "Generated TURN auth secret"
    fi
    export TURN_AUTH_SECRET
    
    # Stop any existing containers
    log_info "Stopping existing containers..."
    docker-compose down >/dev/null 2>&1 || true
    
    # Start Docker services
    log_info "Starting production services..."
    docker-compose up -d
    DOCKER_RUNNING=true
    
    # Wait for backend to be ready
    if ! health_check "Backend" "http://localhost:8000/health"; then
        log_error "Backend failed to start"
        exit 1
    fi
    
    # Wait for frontend to be ready
    if ! health_check "Frontend" "http://localhost:3001"; then
        log_error "Frontend failed to start"
        exit 1
    fi
    
    log_success "Production mode started successfully!"
    echo ""
    echo "🚀 FastRTC Production Mode"
    echo "🌐 External IP: $EXTERNAL_IP"
    echo "📱 Frontend: http://$EXTERNAL_IP:3001"
    echo "🔧 Backend API: http://$EXTERNAL_IP:8000"
    echo "🧊 STUN Server: stun:$EXTERNAL_IP:3478"
    echo "🔄 TURN Server: turn:$EXTERNAL_IP:3478"
    echo ""
    echo "View logs: docker-compose logs -f"
    echo "Press Ctrl+C to stop..."
    
    # Follow logs
    docker-compose logs -f
}

# Parse command line arguments
parse_arguments() {
    local mode=""
    local log_level=""
    
    # Command line flags are already declared globally
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            dev|development|docker|prod|production)
                if [[ -n "$mode" ]]; then
                    log_error "Multiple modes specified: $mode and $1"
                    exit 1
                fi
                mode="$1"
                shift
                ;;
            --log-level)
                if [[ -n "$2" ]] && [[ "$2" =~ ^(DEBUG|INFO|WARNING|ERROR)$ ]]; then
                    log_level="$2"
                    shift 2
                else
                    log_error "Invalid log level. Use: DEBUG, INFO, WARNING, or ERROR"
                    exit 1
                fi
                ;;
            --threading)
                CMDLINE_THREADING_PIPELINE="true"
                export USE_THREADING_PIPELINE=true
                shift
                ;;
            --no-fallback)
                CMDLINE_THREADING_FALLBACK="false"
                export THREADING_FALLBACK_TO_ASYNC=false
                shift
                ;;
            -h|--help|help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$mode" ]]; then
        log_error "No mode specified"
        show_usage
        exit 1
    fi
    
    # Export log level for backend to use (but don't log yet)
    if [[ -n "$log_level" ]]; then
        export LOG_LEVEL="$log_level"
    fi
    
    echo "$mode"
}

# Main function
main() {
    # Show header
    echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║             FastRTC v$VERSION             ║${NC}"
    echo -e "${CYAN}║        Universal Deployment Script     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
    echo ""
    
    # Check arguments
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    # Parse arguments and get mode
    local mode=$(parse_arguments "$@")
    
    # Show log level if set
    if [[ -n "$LOG_LEVEL" ]]; then
        log_info "Log level set to: $LOG_LEVEL"
    fi
    
    # Detect OS
    detect_os
    
    # Run the specified mode
    case "$mode" in
        "dev"|"development")
            run_development
            ;;
        "docker")
            run_docker
            ;;
        "prod"|"production")
            run_production
            ;;
        *)
            log_error "Unknown mode: $mode"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"