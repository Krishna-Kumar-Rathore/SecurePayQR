#!/bin/bash
# SecurePayQR Deployment Scripts
# Automated deployment for development and production environments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check if script is run with appropriate permissions
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root user. Consider using a non-root user for security."
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        error "Insufficient disk space. At least 10GB required."
    fi
    
    # Check memory (minimum 4GB)
    available_memory=$(free -m | awk 'NR==2{print $2}')
    required_memory=4096  # 4GB in MB
    
    if [ "$available_memory" -lt "$required_memory" ]; then
        warn "Low memory detected. At least 4GB RAM recommended for optimal performance."
    fi
    
    log "System requirements check passed!"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create required directories
    mkdir -p {data,models,outputs,logs,ssl,static}
    mkdir -p {config,scripts,src,tests,notebooks}
    mkdir -p {monitoring/{prometheus,grafana/{dashboards,datasources}},nginx}
    
    # Set proper permissions
    chmod 755 data models outputs logs
    chmod 600 config/* 2>/dev/null || true
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log "Creating .env file..."
        cat > .env << EOF
# SecurePayQR Environment Configuration
ENVIRONMENT=production
POSTGRES_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 64)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
WANDB_API_KEY=

# Optional: GPU support
CUDA_VISIBLE_DEVICES=0

# Optional: Custom model path
MODEL_PATH=/app/models/securepayqr_model.onnx
EOF
        log "Environment file created. Please review and update .env as needed."
    fi
    
    log "Environment setup completed!"
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'securepayqr-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

    # Grafana datasource configuration
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Grafana dashboard configuration
    mkdir -p monitoring/grafana/dashboards
    cat > monitoring/grafana/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    log "Monitoring configuration completed!"
}

# Setup Nginx configuration
setup_nginx() {
    log "Setting up Nginx configuration..."
    
    cat > nginx/nginx.conf << EOF
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    log_format main '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                    '\$status \$body_bytes_sent "\$http_referer" '
                    '"\$http_user_agent" "\$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    include /etc/nginx/conf.d/*.conf;
}
EOF

    cat > nginx/default.conf << EOF
upstream api_backend {
    server api:8000;
}

server {
    listen 80;
    server_name localhost;
    
    # Redirect HTTP to HTTPS in production
    # return 301 https://\$server_name\$request_uri;
    
    client_max_body_size 10M;
    
    # API routes
    location /api/ {
        proxy_pass http://api_backend/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias /var/www/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Frontend (if serving from Nginx)
    location / {
        try_files \$uri \$uri/ /index.html;
        root /var/www/static;
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}

# HTTPS configuration (uncomment for production)
# server {
#     listen 443 ssl http2;
#     server_name localhost;
#     
#     ssl_certificate /etc/nginx/ssl/cert.pem;
#     ssl_certificate_key /etc/nginx/ssl/key.pem;
#     ssl_session_timeout 1d;
#     ssl_session_cache shared:MozTLS:10m;
#     ssl_session_tickets off;
#     
#     ssl_protocols TLSv1.2 TLSv1.3;
#     ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
#     ssl_prefer_server_ciphers off;
#     
#     # HSTS
#     add_header Strict-Transport-Security "max-age=63072000" always;
#     
#     # Same location blocks as HTTP server
# }
EOF

    log "Nginx configuration completed!"
}

# Initialize database
init_database() {
    log "Initializing database..."
    
    cat > scripts/init_db.sql << EOF
-- SecurePayQR Database Initialization

-- Create schemas
CREATE SCHEMA IF NOT EXISTS securepayqr;

-- Users table
CREATE TABLE IF NOT EXISTS securepayqr.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detection logs table
CREATE TABLE IF NOT EXISTS securepayqr.detection_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES securepayqr.users(id),
    qr_content TEXT,
    is_tampered BOOLEAN NOT NULL,
    confidence FLOAT NOT NULL,
    processing_time_ms FLOAT NOT NULL,
    model_version VARCHAR(20),
    client_ip INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API usage statistics
CREATE TABLE IF NOT EXISTS securepayqr.api_stats (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms FLOAT NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS securepayqr.model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_detection_logs_user_id ON securepayqr.detection_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_detection_logs_created_at ON securepayqr.detection_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_api_stats_endpoint ON securepayqr.api_stats(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_stats_created_at ON securepayqr.api_stats(created_at);

-- Create a default admin user (password: admin123)
INSERT INTO securepayqr.users (username, email, password_hash) 
VALUES ('admin', 'admin@securepayqr.com', '\$2b\$12\$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LdMnzk9DzO1Ld/u6u')
ON CONFLICT (username) DO NOTHING;

COMMIT;
EOF

    log "Database initialization script created!"
}

# Deploy development environment
deploy_dev() {
    log "Deploying development environment..."
    
    check_requirements
    setup_environment
    setup_monitoring
    setup_nginx
    init_database
    
    # Build and start services
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    
    log "Development environment deployed!"
    log "Services available at:"
    log "  - API: http://localhost:8000"
    log "  - API Docs: http://localhost:8000/docs"
    log "  - Jupyter: http://localhost:8888"
    log "  - Grafana: http://localhost:3000 (admin/admin123)"
    log "  - Prometheus: http://localhost:9090"
    log "  - pgAdmin: http://localhost:5050 (admin@securepayqr.com/admin123)"
}

# Deploy production environment
deploy_prod() {
    log "Deploying production environment..."
    
    check_requirements
    setup_environment
    setup_monitoring
    setup_nginx
    init_database
    
    # Pull latest images and deploy
    docker-compose pull
    docker-compose up -d --build
    
    # Wait for services to be healthy
    log "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "Production deployment successful!"
        log "API is healthy and ready to serve requests"
    else
        error "Production deployment failed - API health check failed"
    fi
    
    log "Production environment deployed!"
    log "Services available at:"
    log "  - API: http://localhost:8000"
    log "  - Grafana: http://localhost:3000"
    log "  - Prometheus: http://localhost:9090"
}

# Backup data
backup() {
    log "Creating backup..."
    
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup MongoDB database
    log "Backing up MongoDB database..."
    docker-compose exec -T mongodb mongodump --db securepayqr --out /tmp/backup
    docker-compose exec -T mongodb tar -czf /tmp/mongodb_backup.tar.gz -C /tmp/backup .
    docker cp $(docker-compose ps -q mongodb):/tmp/mongodb_backup.tar.gz "$backup_dir/mongodb_backup.tar.gz"
    
    # Clean up temporary files in container
    docker-compose exec -T mongodb rm -rf /tmp/backup /tmp/mongodb_backup.tar.gz
    
    # Backup models
    cp -r models "$backup_dir/" 2>/dev/null || true
    
    # Backup configuration
    cp -r config "$backup_dir/" 2>/dev/null || true
    cp .env "$backup_dir/" 2>/dev/null || true
    
    # Create archive
    tar -czf "$backup_dir.tar.gz" -C backups "$(basename "$backup_dir")"
    rm -rf "$backup_dir"
    
    log "Backup created: $backup_dir.tar.gz"
}

# Restore from backup
restore() {
    backup_file=$1
    if [ -z "$backup_file" ]; then
        error "Please specify backup file to restore from"
    fi
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
    fi
    
    log "Restoring from backup: $backup_file"
    
    # Extract backup
    temp_dir=$(mktemp -d)
    tar -xzf "$backup_file" -C "$temp_dir"
    
    # Stop services
    docker-compose down
    
    # Restore MongoDB database
    if [ -f "$temp_dir"/*/mongodb_backup.tar.gz ]; then
        log "Restoring MongoDB database..."
        docker-compose up -d mongodb
        sleep 15
        
        # Copy backup to container and restore
        docker cp "$temp_dir"/*/mongodb_backup.tar.gz $(docker-compose ps -q mongodb):/tmp/
        docker-compose exec -T mongodb sh -c "cd /tmp && tar -xzf mongodb_backup.tar.gz"
        docker-compose exec -T mongodb mongorestore --db securepayqr --drop /tmp/securepayqr
        docker-compose exec -T mongodb rm -rf /tmp/mongodb_backup.tar.gz /tmp/securepayqr
    fi
    
    # Restore other files
    cp -r "$temp_dir"/*/models . 2>/dev/null || true
    cp -r "$temp_dir"/*/config . 2>/dev/null || true
    cp "$temp_dir"/*/.env . 2>/dev/null || true
    
    # Restart services
    docker-compose up -d
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "Restore completed!"
}

# Update deployment
update() {
    log "Updating deployment..."
    
    # Pull latest code (assumes git repository)
    if [ -d ".git" ]; then
        git pull
    fi
    
    # Backup before update
    backup
    
    # Update containers
    docker-compose pull
    docker-compose up -d --build
    
    log "Update completed!"
}

# Clean up resources
cleanup() {
    log "Cleaning up resources..."
    
    # Stop and remove containers
    docker-compose down -v
    
    # Remove images
    docker system prune -f
    
    # Clean up volumes (WARNING: This will delete all data)
    read -p "Do you want to remove all data volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
        log "All volumes removed"
    fi
    
    log "Cleanup completed!"
}

# Show logs
logs() {
    service=${1:-}
    if [ -n "$service" ]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Show status
status() {
    log "Service Status:"
    docker-compose ps
    
    echo
    log "System Resources:"
    docker stats --no-stream
    
    echo
    log "Health Checks:"
    curl -s http://localhost:8000/health || echo "API: DOWN"
    curl -s http://localhost:3000/api/health || echo "Grafana: DOWN"
}

# Main script logic
case "${1:-}" in
    "dev")
        deploy_dev
        ;;
    "prod")
        deploy_prod
        ;;
    "backup")
        backup
        ;;
    "restore")
        restore "$2"
        ;;
    "update")
        update
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        logs "$2"
        ;;
    "status")
        status
        ;;
    *)
        echo "SecurePayQR Deployment Script"
        echo "Usage: $0 {dev|prod|backup|restore|update|cleanup|logs|status}"
        echo
        echo "Commands:"
        echo "  dev       - Deploy development environment"
        echo "  prod      - Deploy production environment"
        echo "  backup    - Create backup of data and configuration"
        echo "  restore   - Restore from backup file"
        echo "  update    - Update deployment with latest code"
        echo "  cleanup   - Clean up containers and resources"
        echo "  logs      - Show service logs (optional: specify service name)"
        echo "  status    - Show service status and health"
        echo
        echo "Examples:"
        echo "  $0 dev                              # Deploy development environment"
        echo "  $0 prod                             # Deploy production environment"
        echo "  $0 restore backups/20240101.tar.gz # Restore from specific backup"
        echo "  $0 logs api                         # Show API service logs"
        exit 1
        ;;
esac