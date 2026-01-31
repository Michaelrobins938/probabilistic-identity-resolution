-- PostgreSQL Schema for Identity Resolution System
-- Database: identity_resolution

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Accounts table (households)
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

-- Persons table (individual household members)
CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id VARCHAR(255) UNIQUE NOT NULL,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    label VARCHAR(50) NOT NULL,
    persona_type VARCHAR(50),
    confidence_score FLOAT DEFAULT 0.0,
    cluster_features FLOAT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(account_id, label)
);

-- Devices table
CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id VARCHAR(255) UNIQUE NOT NULL,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    device_type VARCHAR(50) NOT NULL, -- 'tv', 'mobile', 'tablet', 'desktop'
    device_fingerprint VARCHAR(255),
    user_agent TEXT,
    screen_resolution VARCHAR(50),
    os_info VARCHAR(100),
    browser_info VARCHAR(100),
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Sessions table (viewing sessions)
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    device_id UUID REFERENCES devices(id) ON DELETE SET NULL,
    
    -- Temporal features
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER DEFAULT 0,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    
    -- Content features
    content_id VARCHAR(255),
    content_type VARCHAR(50),
    genre VARCHAR(100),
    
    -- Assignment confidence
    assignment_confidence FLOAT DEFAULT 0.0,
    assignment_method VARCHAR(50), -- 'clustering', 'fallback', 'manual'
    
    -- Raw features (JSON for flexibility)
    feature_vector JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Events table (individual streaming events)
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    
    event_type VARCHAR(50) NOT NULL, -- 'play', 'pause', 'stop', 'seek', 'complete'
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Event properties
    properties JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Attribution table (marketing touchpoints)
CREATE TABLE attributions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    
    -- Channel information
    channel VARCHAR(100) NOT NULL, -- 'email', 'social', 'search', 'organic', etc.
    campaign_id VARCHAR(255),
    
    -- Attribution models
    first_touch FLOAT DEFAULT 0.0,
    last_touch FLOAT DEFAULT 0.0,
    linear FLOAT DEFAULT 0.0,
    time_decay FLOAT DEFAULT 0.0,
    markov FLOAT DEFAULT 0.0,
    shapley FLOAT DEFAULT 0.0,
    hybrid FLOAT DEFAULT 0.0,
    
    -- Conversion details
    converted BOOLEAN DEFAULT FALSE,
    conversion_value DECIMAL(10,2) DEFAULT 0.00,
    conversion_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Touchpoint sequence
    touchpoint_order INTEGER DEFAULT 0,
    days_to_conversion INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Cluster models table (ML model snapshots)
CREATE TABLE cluster_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    
    model_version VARCHAR(50) NOT NULL,
    algorithm VARCHAR(50) NOT NULL, -- 'kmeans', 'gmm', 'hierarchical'
    
    -- Model parameters
    n_clusters INTEGER NOT NULL,
    feature_names TEXT[] DEFAULT '{}',
    centroids FLOAT[][] DEFAULT '{}',
    
    -- Performance metrics
    silhouette_score FLOAT,
    inertia FLOAT,
    brier_score FLOAT,
    
    -- Drift detection
    drift_score FLOAT DEFAULT 0.0,
    drift_detected BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Drift history table
CREATE TABLE drift_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES accounts(id) ON DELETE CASCADE,
    model_id UUID REFERENCES cluster_models(id) ON DELETE CASCADE,
    
    drift_type VARCHAR(50) NOT NULL, -- 'gradual', 'sudden', 'recurring', 'concept', 'feature'
    drift_score FLOAT NOT NULL,
    threshold FLOAT DEFAULT 2.0,
    
    -- Context
    baseline_distribution JSONB DEFAULT '{}',
    current_distribution JSONB DEFAULT '{}',
    kl_divergence FLOAT,
    
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- GDPR deletion audit log
CREATE TABLE gdpr_deletion_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    
    account_id UUID NOT NULL,
    scope VARCHAR(50) NOT NULL, -- 'device_only', 'person', 'household', 'partial'
    
    -- Deletion details
    deleted_sessions INTEGER DEFAULT 0,
    deleted_events INTEGER DEFAULT 0,
    deleted_attributions INTEGER DEFAULT 0,
    
    -- Cryptographic verification
    deletion_hash VARCHAR(64) NOT NULL, -- SHA-256
    previous_hash VARCHAR(64),
    
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'verified'
    
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    verified_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX idx_sessions_account_id ON sessions(account_id);
CREATE INDEX idx_sessions_person_id ON sessions(person_id);
CREATE INDEX idx_sessions_device_id ON sessions(device_id);
CREATE INDEX idx_sessions_start_time ON sessions(start_time);
CREATE INDEX idx_events_session_id ON events(session_id);
CREATE INDEX idx_events_account_id ON events(account_id);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_attributions_account_id ON attributions(account_id);
CREATE INDEX idx_attributions_person_id ON attributions(person_id);
CREATE INDEX idx_attributions_channel ON attributions(channel);
CREATE INDEX idx_persons_account_id ON persons(account_id);
CREATE INDEX idx_devices_account_id ON devices(account_id);

-- GIN indexes for JSONB queries
CREATE INDEX idx_sessions_feature_vector ON sessions USING GIN(feature_vector);
CREATE INDEX idx_events_properties ON events USING GIN(properties);

-- Comments
COMMENT ON TABLE accounts IS 'Top-level household/organization accounts';
COMMENT ON TABLE persons IS 'Individual household members identified by ML clustering';
COMMENT ON TABLE devices IS 'Physical devices used for streaming';
COMMENT ON TABLE sessions IS 'Streaming sessions with behavioral features and person assignments';
COMMENT ON TABLE events IS 'Granular streaming events (play, pause, etc.)';
COMMENT ON TABLE attributions IS 'Marketing attribution across multiple touchpoints and models';
COMMENT ON TABLE cluster_models IS 'ML model snapshots for household clustering';
COMMENT ON TABLE drift_history IS 'History of behavioral drift detection events';
COMMENT ON TABLE gdpr_deletion_log IS 'Immutable audit trail for GDPR deletion requests';
