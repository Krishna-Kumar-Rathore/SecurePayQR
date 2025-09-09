// SecurePayQR MongoDB Initialization Script
// This script sets up the database, collections, and indexes

// Connect to the securepayqr database
db = db.getSiblingDB('securepayqr');

// Create collections with validators
print('Creating collections with schema validation...');

// Users collection with validation
db.createCollection('users', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['username', 'email', 'password_hash', 'is_active', 'created_at'],
            properties: {
                username: {
                    bsonType: 'string',
                    minLength: 3,
                    maxLength: 50,
                    description: 'Username must be a string between 3-50 characters'
                },
                email: {
                    bsonType: 'string',
                    pattern: '^[^@]+@[^@]+\\.[^@]+$',
                    description: 'Email must be a valid email address'
                },
                password_hash: {
                    bsonType: 'string',
                    description: 'Password hash is required'
                },
                is_active: {
                    bsonType: 'bool',
                    description: 'Active status must be a boolean'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp is required'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Update timestamp'
                }
            }
        }
    }
});

// Detection logs collection with validation
db.createCollection('detection_logs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['is_tampered', 'confidence', 'probabilities', 'processing_time_ms', 'model_version', 'client_ip', 'timestamp'],
            properties: {
                user_id: {
                    bsonType: 'string',
                    description: 'User ID if authenticated'
                },
                qr_content: {
                    bsonType: 'string',
                    description: 'QR code content'
                },
                is_tampered: {
                    bsonType: 'bool',
                    description: 'Tampering detection result'
                },
                confidence: {
                    bsonType: 'double',
                    minimum: 0,
                    maximum: 1,
                    description: 'Confidence score between 0 and 1'
                },
                probabilities: {
                    bsonType: 'object',
                    required: ['valid', 'tampered'],
                    properties: {
                        valid: { bsonType: 'double', minimum: 0, maximum: 1 },
                        tampered: { bsonType: 'double', minimum: 0, maximum: 1 }
                    },
                    description: 'Probability scores for each class'
                },
                processing_time_ms: {
                    bsonType: 'double',
                    minimum: 0,
                    description: 'Processing time in milliseconds'
                },
                model_version: {
                    bsonType: 'string',
                    description: 'Model version used for detection'
                },
                client_ip: {
                    bsonType: 'string',
                    description: 'Client IP address'
                },
                user_agent: {
                    bsonType: 'string',
                    description: 'Client user agent'
                },
                timestamp: {
                    bsonType: 'date',
                    description: 'Detection timestamp'
                },
                features: {
                    bsonType: 'object',
                    description: 'Optional model features'
                }
            }
        }
    }
});

// API stats collection with validation
db.createCollection('api_stats', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['endpoint', 'method', 'status_code', 'response_time_ms', 'timestamp'],
            properties: {
                endpoint: {
                    bsonType: 'string',
                    description: 'API endpoint path'
                },
                method: {
                    bsonType: 'string',
                    enum: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
                    description: 'HTTP method'
                },
                status_code: {
                    bsonType: 'int',
                    minimum: 100,
                    maximum: 599,
                    description: 'HTTP status code'
                },
                response_time_ms: {
                    bsonType: 'double',
                    minimum: 0,
                    description: 'Response time in milliseconds'
                },
                request_size_bytes: {
                    bsonType: 'int',
                    minimum: 0,
                    description: 'Request size in bytes'
                },
                response_size_bytes: {
                    bsonType: 'int',
                    minimum: 0,
                    description: 'Response size in bytes'
                },
                timestamp: {
                    bsonType: 'date',
                    description: 'Request timestamp'
                }
            }
        }
    }
});

// Model metrics collection
db.createCollection('model_metrics', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['model_version', 'evaluation_date'],
            properties: {
                model_version: {
                    bsonType: 'string',
                    description: 'Model version identifier'
                },
                accuracy: {
                    bsonType: 'double',
                    minimum: 0,
                    maximum: 1,
                    description: 'Model accuracy score'
                },
                precision: {
                    bsonType: 'double',
                    minimum: 0,
                    maximum: 1,
                    description: 'Model precision score'
                },
                recall: {
                    bsonType: 'double',
                    minimum: 0,
                    maximum: 1,
                    description: 'Model recall score'
                },
                f1_score: {
                    bsonType: 'double',
                    minimum: 0,
                    maximum: 1,
                    description: 'Model F1 score'
                },
                roc_auc: {
                    bsonType: 'double',
                    minimum: 0,
                    maximum: 1,
                    description: 'ROC AUC score'
                },
                evaluation_date: {
                    bsonType: 'date',
                    description: 'Evaluation timestamp'
                }
            }
        }
    }
});

print('Collections created successfully!');

// Create indexes for better performance
print('Creating indexes...');

// Users collection indexes
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "created_at": -1 });
db.users.createIndex({ "is_active": 1 });

// Detection logs indexes
db.detection_logs.createIndex({ "user_id": 1 });
db.detection_logs.createIndex({ "timestamp": -1 });
db.detection_logs.createIndex({ "is_tampered": 1 });
db.detection_logs.createIndex({ "confidence": -1 });
db.detection_logs.createIndex({ "model_version": 1 });
db.detection_logs.createIndex({ "client_ip": 1 });

// Compound indexes for common queries
db.detection_logs.createIndex({ "user_id": 1, "timestamp": -1 });
db.detection_logs.createIndex({ "is_tampered": 1, "timestamp": -1 });
db.detection_logs.createIndex({ "model_version": 1, "timestamp": -1 });

// API stats indexes
db.api_stats.createIndex({ "endpoint": 1 });
db.api_stats.createIndex({ "timestamp": -1 });
db.api_stats.createIndex({ "status_code": 1 });
db.api_stats.createIndex({ "response_time_ms": -1 });

// Compound indexes for analytics
db.api_stats.createIndex({ "endpoint": 1, "timestamp": -1 });
db.api_stats.createIndex({ "method": 1, "status_code": 1 });

// Model metrics indexes
db.model_metrics.createIndex({ "model_version": 1 });
db.model_metrics.createIndex({ "evaluation_date": -1 });
db.model_metrics.createIndex({ "accuracy": -1 });

// TTL indexes for automatic cleanup (optional)
// Uncomment to enable automatic deletion of old records

// Delete detection logs older than 1 year
// db.detection_logs.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 31536000 });

// Delete API stats older than 90 days
// db.api_stats.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 7776000 });

print('Indexes created successfully!');

// Insert default admin user
print('Creating default admin user...');

const adminUser = {
    username: 'admin',
    email: 'admin@securepayqr.com',
    // Password: admin123 (bcrypt hash)
    password_hash: '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LdMnzk9DzO1Ld/u6u',
    is_active: true,
    created_at: new Date(),
    updated_at: new Date(),
    role: 'admin'
};

try {
    db.users.insertOne(adminUser);
    print('Default admin user created successfully!');
    print('Username: admin');
    print('Password: admin123');
    print('Email: admin@securepayqr.com');
} catch (error) {
    if (error.code === 11000) {
        print('Admin user already exists, skipping creation.');
    } else {
        print('Error creating admin user: ' + error.message);
    }
}

// Insert sample model metrics
print('Inserting sample model metrics...');

const sampleMetrics = {
    model_version: '1.0.0',
    accuracy: 0.962,
    precision: 0.978,
    recall: 0.954,
    f1_score: 0.966,
    roc_auc: 0.989,
    evaluation_date: new Date(),
    dataset_size: 2000,
    training_epochs: 50,
    validation_split: 0.2
};

try {
    db.model_metrics.insertOne(sampleMetrics);
    print('Sample model metrics inserted successfully!');
} catch (error) {
    print('Error inserting sample metrics: ' + error.message);
}

// Create aggregation views for common queries
print('Creating aggregation views...');

// Daily detection summary view
db.createView('daily_detection_summary', 'detection_logs', [
    {
        $group: {
            _id: {
                $dateToString: { format: "%Y-%m-%d", date: "$timestamp" }
            },
            total_detections: { $sum: 1 },
            tampered_count: {
                $sum: { $cond: [{ $eq: ["$is_tampered", true] }, 1, 0] }
            },
            valid_count: {
                $sum: { $cond: [{ $eq: ["$is_tampered", false] }, 1, 0] }
            },
            avg_confidence: { $avg: "$confidence" },
            avg_processing_time: { $avg: "$processing_time_ms" },
            date: { $first: "$timestamp" }
        }
    },
    {
        $sort: { "_id": -1 }
    }
]);

// API performance summary view
db.createView('api_performance_summary', 'api_stats', [
    {
        $group: {
            _id: "$endpoint",
            total_requests: { $sum: 1 },
            avg_response_time: { $avg: "$response_time_ms" },
            success_count: {
                $sum: { $cond: [{ $lt: ["$status_code", 400] }, 1, 0] }
            },
            error_count: {
                $sum: { $cond: [{ $gte: ["$status_code", 400] }, 1, 0] }
            },
            success_rate: {
                $multiply: [
                    { $divide: [
                        { $sum: { $cond: [{ $lt: ["$status_code", 400] }, 1, 0] }},
                        { $sum: 1 }
                    ]},
                    100
                ]
            }
        }
    },
    {
        $sort: { "total_requests": -1 }
    }
]);

print('Aggregation views created successfully!');

// Set up capped collections for real-time monitoring (optional)
print('Setting up capped collections for real-time monitoring...');

// Real-time detection events (limited to 1000 documents)
db.createCollection('realtime_detections', {
    capped: true,
    size: 1048576, // 1MB
    max: 1000
});

// Real-time API events (limited to 5000 documents)
db.createCollection('realtime_api_events', {
    capped: true,
    size: 5242880, // 5MB
    max: 5000
});

print('Capped collections created successfully!');

// Database configuration
print('Configuring database settings...');

// Set profiling level for slow operations (optional)
db.setProfilingLevel(1, { slowms: 100 });

print('Database configuration completed!');

// Final summary
print('\n=== SecurePayQR MongoDB Initialization Complete ===');
print('Collections created:');
print('  - users (with unique indexes on username and email)');
print('  - detection_logs (with performance indexes)');
print('  - api_stats (with analytics indexes)');
print('  - model_metrics (with version tracking)');
print('  - realtime_detections (capped collection)');
print('  - realtime_api_events (capped collection)');
print('');
print('Views created:');
print('  - daily_detection_summary');
print('  - api_performance_summary');
print('');
print('Default admin user:');
print('  Username: admin');
print('  Password: admin123');
print('  Email: admin@securepayqr.com');
print('');
print('Database is ready for SecurePayQR application!');
print('===============================================');