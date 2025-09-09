// frontend/src/components/Results.js
import React, { useState } from 'react';
import { useDetection } from '../context/DetectionContext';

const Results = () => {
  const { detectionHistory, statistics, clearHistory } = useDetection();
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('timestamp');

  const filteredResults = detectionHistory.filter(result => {
    if (filter === 'valid') return !result.detection.is_tampered;
    if (filter === 'tampered') return result.detection.is_tampered;
    return true;
  });

  const sortedResults = [...filteredResults].sort((a, b) => {
    switch (sortBy) {
      case 'confidence':
        return b.detection.confidence - a.detection.confidence;
      case 'processing_time':
        return (a.detection.processing_time_ms || 0) - (b.detection.processing_time_ms || 0);
      default:
        return new Date(b.timestamp) - new Date(a.timestamp);
    }
  });

  return (
    <div className="space-y-6">
      {/* Statistics Summary */}
      <StatisticsCards statistics={statistics} />
      
      {/* Results Table */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">Detection Results</h2>
            <div className="flex items-center space-x-4">
              <FilterDropdown filter={filter} setFilter={setFilter} />
              <SortDropdown sortBy={sortBy} setSortBy={setSortBy} />
              <button
                onClick={clearHistory}
                className="px-4 py-2 text-red-600 border border-red-300 rounded-lg hover:bg-red-50"
              >
                Clear History
              </button>
            </div>
          </div>
        </div>
        
        <ResultsTable results={sortedResults} />
      </div>
    </div>
  );
};

const StatisticsCards = ({ statistics }) => {
  const cards = [
    { title: 'Total Scans', value: statistics.totalScans, icon: '🔍', color: 'blue' },
    { title: 'Valid QR Codes', value: statistics.validScans, icon: '✅', color: 'green' },
    { title: 'Tampered QR Codes', value: statistics.tamperedScans, icon: '❌', color: 'red' },
    { title: 'Avg. Confidence', value: `${(statistics.averageConfidence * 100).toFixed(1)}%`, icon: '📊', color: 'purple' }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, index) => (
        <div key={index} className="bg-white rounded-lg p-6 shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">{card.icon}</div>
            <div>
              <p className="text-sm text-gray-600">{card.title}</p>
              <p className={`text-2xl font-bold text-${card.color}-600`}>{card.value}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

const FilterDropdown = ({ filter, setFilter }) => (
  <select
    value={filter}
    onChange={(e) => setFilter(e.target.value)}
    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
  >
    <option value="all">All Results</option>
    <option value="valid">Valid Only</option>
    <option value="tampered">Tampered Only</option>
  </select>
);

const SortDropdown = ({ sortBy, setSortBy }) => (
  <select
    value={sortBy}
    onChange={(e) => setSortBy(e.target.value)}
    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
  >
    <option value="timestamp">Sort by Time</option>
    <option value="confidence">Sort by Confidence</option>
    <option value="processing_time">Sort by Speed</option>
  </select>
);

const ResultsTable = ({ results }) => {
  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-4xl mb-4">📊</div>
        <p className="text-gray-500">No detection results found</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">QR Content</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Processing Time</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {results.map((result) => (
            <ResultRow key={result.id} result={result} />
          ))}
        </tbody>
      </table>
    </div>
  );
};

const ResultRow = ({ result }) => {
  const isValid = !result.detection.is_tampered;
  
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="flex items-center">
          <span className="text-lg mr-2">{isValid ? '✅' : '❌'}</span>
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${
            isValid ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {isValid ? 'Valid' : 'Tampered'}
          </span>
        </div>
      </td>
      <td className="px-6 py-4">
        <div className="text-sm text-gray-900 max-w-xs truncate">
          {result.qrData.text}
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="text-sm text-gray-900">
          {(result.detection.confidence * 100).toFixed(1)}%
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="text-sm text-gray-900">
          {(result.detection.processing_time_ms || 0).toFixed(0)}ms
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="text-sm text-gray-900">
          {new Date(result.timestamp).toLocaleString()}
        </div>
      </td>
    </tr>
  );
};

export default Results;