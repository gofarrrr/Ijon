-- Migration: Create Graph Tables for Entity and Relationship Storage
-- Version: 001
-- Description: Creates tables for storing entities and relationships extracted from documents

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create entities table
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT,
    properties JSONB DEFAULT '{}',
    chunk_ids TEXT[] DEFAULT '{}',
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique entities per name and type
    UNIQUE(name, type)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_chunk_ids ON entities USING gin(chunk_ids);
CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence DESC);

-- Create relationships table
CREATE TABLE IF NOT EXISTS relationships (
    id SERIAL PRIMARY KEY,
    source_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    properties JSONB DEFAULT '{}',
    chunk_ids TEXT[] DEFAULT '{}',
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent duplicate relationships
    UNIQUE(source_entity_id, target_entity_id, type)
);

-- Create indexes for relationship queries
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(type);
CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON relationships(confidence DESC);

-- Create a view for easier querying of entity relationships
CREATE OR REPLACE VIEW entity_graph AS
SELECT 
    e1.id as source_id,
    e1.name as source_name,
    e1.type as source_type,
    r.type as relationship_type,
    e2.id as target_id,
    e2.name as target_name,
    e2.type as target_type,
    r.confidence as relationship_confidence
FROM relationships r
JOIN entities e1 ON r.source_entity_id = e1.id
JOIN entities e2 ON r.target_entity_id = e2.id;

-- Add comments for documentation
COMMENT ON TABLE entities IS 'Stores entities extracted from documents (concepts, people, techniques, etc.)';
COMMENT ON TABLE relationships IS 'Stores relationships between entities';
COMMENT ON VIEW entity_graph IS 'Simplified view of entity relationships for easier querying';