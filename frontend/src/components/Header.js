// frontend/src/components/Header.js - FIXED VERSION
import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useModel } from '../context/ModelContext';

const Header = () => {
  const { modelStatus } = useModel();
  const navigate = useNavigate();
  const location = useLocation();

  const navigation = [
    { name: 'Scanner', path: '/scanner', icon: '🔍' },
    { name: 'Results', path: '/results', icon: '📊' },
    { name: 'Analytics', path: '/analytics', icon: '📈' },
    { name: 'About', path: '/about', icon: 'ℹ️' }
  ];

  const handleNavigation = (path) => {
    navigate(path);
  };

  const isCurrentPage = (path) => {
    // Handle root path and scanner path as the same
    if (path === '/scanner' && (location.pathname === '/' || location.pathname === '/scanner')) {
      return true;
    }
    return location.pathname === path;
  };

  return (
    <header className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center cursor-pointer" onClick={() => handleNavigation('/')}>
            <div className="text-2xl mr-3">🔒</div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">SecurePayQR</h1>
              <p className="text-sm text-gray-600">AI-Powered QR Code Fraud Detection</p>
            </div>
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-6">
            {navigation.map((item) => (
              <button
                key={item.path}
                onClick={() => handleNavigation(item.path)}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isCurrentPage(item.path)
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.name}
              </button>
            ))}
          </nav>

          {/* Mobile Navigation */}
          <div className="md:hidden">
            <select
              value={location.pathname}
              onChange={(e) => handleNavigation(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            >
              {navigation.map((item) => (
                <option key={item.path} value={item.path}>
                  {item.icon} {item.name}
                </option>
              ))}
            </select>
          </div>
          
          <div className="flex items-center space-x-4">
            <ModelStatusIndicator status={modelStatus} />
          </div>
        </div>
      </div>
    </header>
  );
};

const ModelStatusIndicator = ({ status }) => {
  const getStatusInfo = () => {
    switch (status) {
      case 'ready':
        return { color: 'green', text: 'Model Ready', icon: '✅' };
      case 'loading':
        return { color: 'yellow', text: 'Loading Model...', icon: '⏳' };
      case 'error':
        return { color: 'red', text: 'Model Error', icon: '❌' };
      default:
        return { color: 'gray', text: 'Initializing...', icon: '⚪' };
    }
  };

  const { color, text, icon } = getStatusInfo();

  return (
    <div className={`flex items-center text-${color}-600`}>
      <div className={`w-2 h-2 bg-${color}-500 rounded-full mr-2 ${status === 'loading' ? 'animate-pulse' : ''}`}></div>
      <span className="text-sm">{icon} {text}</span>
    </div>
  );
};

export default Header;