// frontend/README.md
# SecurePayQR Frontend

Modern React.js frontend for the SecurePayQR fraud detection system.

## Features

- 📷 Real-time camera QR code scanning
- 🔍 AI-powered fraud detection interface
- 📊 Comprehensive analytics dashboard
- 📱 Responsive design for all devices
- ⚡ Fast performance with optimized components

## Getting Started

### Prerequisites

- Node.js 16+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

### Environment Setup

Create a `.env` file:

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENABLE_MOCK=true
```

### Project Structure

```
src/
├── components/       # React components
├── context/         # React context providers
├── hooks/           # Custom hooks
├── utils/           # Utility functions
├── App.js          # Main app component
└── index.js        # Entry point
```

### Available Scripts

- `npm start` - Development server
- `npm run build` - Production build
- `npm test` - Run tests
- `npm run lint` - ESLint check
- `npm run lint:fix` - Fix ESLint issues

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t securepayqr-frontend .

# Run container
docker run -p 3000:3000 securepayqr-frontend
```

### Production Build

```bash
npm run build
serve -s build
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request