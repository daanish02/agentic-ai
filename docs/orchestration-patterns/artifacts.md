# Artifacts

## Table of Contents

- [Introduction](#introduction)
- [Artifact Types](#artifact-types)
- [Artifact Patterns](#artifact-patterns)
- [Versioning](#versioning)
- [Artifact Storage](#artifact-storage)
- [Metadata Management](#metadata-management)
- [Content Addressing](#content-addressing)
- [Artifact Dependencies](#artifact-dependencies)
- [Lifecycle Management](#lifecycle-management)
- [Sharing and Distribution](#sharing-and-distribution)
- [Caching Strategies](#caching-strategies)
- [Validation and Integrity](#validation-and-integrity)
- [Real-World Applications](#real-world-applications)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Artifacts are tangible outputs produced by agents during execution. Unlike ephemeral conversation state, artifacts are:

- **Persistent**: Survive beyond conversation sessions
- **Versioned**: Track changes over time
- **Shareable**: Can be distributed to other agents or systems
- **Reusable**: Can be consumed as inputs to future workflows
- **Trackable**: Maintain provenance and metadata

> "Artifacts turn agent reasoning into concrete, reusable work products."

Effective artifact management enables:
- **Incremental development**: Build on previous outputs
- **Collaboration**: Share work between agents
- **Audit trails**: Track what was produced and when
- **Reproducibility**: Recreate past results
- **Knowledge accumulation**: Build organizational memory

```
Agent Workflow with Artifacts:

Input → Agent → Artifact → Storage
         ↓
    Metadata
         ↓
    Version Control
         ↓
    Artifact Catalog

Later:
Storage → Artifact → Agent → New Artifact
```

## Artifact Types

Different categories of artifacts agents produce.

### Code Artifacts

Generated source code, scripts, and configurations.

```python
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class ArtifactType(Enum):
    """Types of artifacts"""
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    MODEL = "model"
    IMAGE = "image"
    CONFIG = "config"
    REPORT = "report"

@dataclass
class CodeArtifact:
    """Code artifact"""
    artifact_id: str
    language: str
    code: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)
    
    def save(self, path: str):
        """Save code to file"""
        with open(path, 'w') as f:
            f.write(self.code)
    
    def execute(self, context: dict = None) -> any:
        """Execute code artifact"""
        context = context or {}
        exec(self.code, context)
        return context

# Example: Agent generates a function
artifact = CodeArtifact(
    artifact_id="func_001",
    language="python",
    code="""
def calculate_metrics(data):
    return {
        'mean': sum(data) / len(data),
        'min': min(data),
        'max': max(data)
    }
""",
    description="Calculate basic statistics",
    dependencies=[]
)

# Save artifact
artifact.save("metrics.py")

# Execute artifact
result = artifact.execute()
print(result['calculate_metrics']([1, 2, 3, 4, 5]))
```

### Document Artifacts

Generated text, markdown, reports.

```python
from markdown import markdown
import json

@dataclass
class DocumentArtifact:
    """Document artifact"""
    artifact_id: str
    format: str  # markdown, html, pdf, text
    content: str
    title: str
    author: str
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)
    
    def to_html(self) -> str:
        """Convert to HTML"""
        if self.format == "markdown":
            return markdown(self.content)
        elif self.format == "html":
            return self.content
        else:
            return f"<pre>{self.content}</pre>"
    
    def save(self, path: str):
        """Save document"""
        with open(path, 'w') as f:
            f.write(self.content)
    
    def to_dict(self) -> dict:
        """Serialize to dict"""
        return {
            "artifact_id": self.artifact_id,
            "format": self.format,
            "content": self.content,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata
        }

# Example: Agent generates a report
report = DocumentArtifact(
    artifact_id="report_001",
    format="markdown",
    title="System Analysis Report",
    author="Agent-1",
    content="""
# System Analysis Report

## Executive Summary
Analysis completed successfully.

## Findings
- Finding 1
- Finding 2

## Recommendations
- Recommendation 1
- Recommendation 2
"""
)

report.save("report.md")
html = report.to_html()
```

### Data Artifacts

Generated datasets, tables, structured data.

```python
import pandas as pd
import pickle

@dataclass
class DataArtifact:
    """Data artifact"""
    artifact_id: str
    data: any  # DataFrame, dict, list, etc.
    schema: Optional[dict] = None
    format: str = "parquet"  # parquet, csv, json, pickle
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)
    
    def save(self, path: str):
        """Save data artifact"""
        if isinstance(self.data, pd.DataFrame):
            if self.format == "parquet":
                self.data.to_parquet(path)
            elif self.format == "csv":
                self.data.to_csv(path, index=False)
            elif self.format == "json":
                self.data.to_json(path)
        else:
            # Pickle for general objects
            with open(path, 'wb') as f:
                pickle.dump(self.data, f)
    
    @staticmethod
    def load(path: str, format: str = "parquet") -> "DataArtifact":
        """Load data artifact"""
        if format == "parquet":
            data = pd.read_parquet(path)
        elif format == "csv":
            data = pd.read_csv(path)
        elif format == "json":
            data = pd.read_json(path)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        return DataArtifact(
            artifact_id=path,
            data=data,
            format=format
        )
    
    def validate(self) -> bool:
        """Validate data against schema"""
        if not self.schema:
            return True
        
        if isinstance(self.data, pd.DataFrame):
            # Check columns
            required_cols = self.schema.get("columns", [])
            return all(col in self.data.columns for col in required_cols)
        
        return True

# Example: Agent generates processed data
df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'score': [85, 92, 78],
    'category': ['A', 'B', 'A']
})

artifact = DataArtifact(
    artifact_id="processed_data_001",
    data=df,
    schema={"columns": ["user_id", "score", "category"]},
    format="parquet",
    description="Processed user scores"
)

artifact.save("processed_data.parquet")

# Later: Load artifact
loaded = DataArtifact.load("processed_data.parquet")
print(loaded.data.head())
```

### Model Artifacts

Trained models, embeddings, weights.

```python
import joblib
import json

@dataclass
class ModelArtifact:
    """Machine learning model artifact"""
    artifact_id: str
    model: any
    model_type: str  # sklearn, pytorch, tensorflow, etc.
    parameters: dict
    metrics: dict
    training_data_ref: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)
    
    def save(self, model_path: str, metadata_path: str):
        """Save model and metadata"""
        # Save model
        if self.model_type == "sklearn":
            joblib.dump(self.model, model_path)
        else:
            # Custom serialization
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            "artifact_id": self.artifact_id,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "training_data_ref": self.training_data_ref,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load(model_path: str, metadata_path: str) -> "ModelArtifact":
        """Load model artifact"""
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        if metadata["model_type"] == "sklearn":
            model = joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        return ModelArtifact(
            artifact_id=metadata["artifact_id"],
            model=model,
            model_type=metadata["model_type"],
            parameters=metadata["parameters"],
            metrics=metadata["metrics"],
            training_data_ref=metadata.get("training_data_ref"),
            version=metadata["version"],
            metadata=metadata.get("metadata", {})
        )
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

# Example: Agent trains and saves model
from sklearn.linear_model import LinearRegression
import numpy as np

# Train model
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression().fit(X, y)

# Create artifact
artifact = ModelArtifact(
    artifact_id="model_001",
    model=model,
    model_type="sklearn",
    parameters={"fit_intercept": True},
    metrics={"r2_score": 1.0},
    training_data_ref="training_data_001"
)

artifact.save("model.pkl", "model_metadata.json")

# Later: Load and use
loaded_model = ModelArtifact.load("model.pkl", "model_metadata.json")
prediction = loaded_model.predict([[5]])
print(f"Prediction: {prediction}")
```

## Artifact Patterns

Common patterns for managing artifacts.

### Artifact Registry

Central catalog of all artifacts.

```python
from typing import Dict, List, Optional
import sqlite3

class ArtifactRegistry:
    """Central registry for all artifacts"""
    
    def __init__(self, db_path: str = "artifacts.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                description TEXT,
                metadata TEXT,
                checksum TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifact_versions (
                artifact_id TEXT,
                version TEXT,
                storage_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                changes TEXT,
                PRIMARY KEY (artifact_id, version)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register(
        self,
        artifact_id: str,
        artifact_type: str,
        storage_path: str,
        version: str = "1.0.0",
        created_by: str = "system",
        description: str = "",
        metadata: dict = None,
        checksum: str = ""
    ):
        """Register artifact"""
        conn = sqlite3.connect(self.db_path)
        
        metadata_json = json.dumps(metadata or {})
        
        conn.execute("""
            INSERT OR REPLACE INTO artifacts
            (artifact_id, artifact_type, storage_path, version, created_by, description, metadata, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (artifact_id, artifact_type, storage_path, version, created_by, description, metadata_json, checksum))
        
        # Also record version
        conn.execute("""
            INSERT INTO artifact_versions
            (artifact_id, version, storage_path, changes)
            VALUES (?, ?, ?, ?)
        """, (artifact_id, version, storage_path, "Initial version"))
        
        conn.commit()
        conn.close()
        
        print(f"Registered artifact: {artifact_id} v{version}")
    
    def get(self, artifact_id: str) -> Optional[dict]:
        """Get artifact metadata"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result = dict(row)
            result['metadata'] = json.loads(result['metadata'])
            return result
        return None
    
    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> List[dict]:
        """List artifacts with filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM artifacts WHERE 1=1"
        params = []
        
        if artifact_type:
            query += " AND artifact_type = ?"
            params.append(artifact_type)
        
        if created_by:
            query += " AND created_by = ?"
            params.append(created_by)
        
        query += " ORDER BY created_at DESC"
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_versions(self, artifact_id: str) -> List[dict]:
        """Get all versions of artifact"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT * FROM artifact_versions
            WHERE artifact_id = ?
            ORDER BY created_at DESC
        """, (artifact_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search(self, query: str) -> List[dict]:
        """Search artifacts by description"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT * FROM artifacts
            WHERE description LIKE ?
            ORDER BY created_at DESC
        """, (f"%{query}%",))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

# Usage
registry = ArtifactRegistry()

# Register artifacts
registry.register(
    artifact_id="code_001",
    artifact_type="code",
    storage_path="/artifacts/code_001.py",
    version="1.0.0",
    created_by="agent-1",
    description="User authentication function",
    metadata={"language": "python", "lines": 45}
)

# Search artifacts
results = registry.search("authentication")
print(f"Found {len(results)} artifacts")

# Get artifact
artifact = registry.get("code_001")
print(f"Artifact: {artifact['description']}")
```

### Builder Pattern

Fluent interface for creating artifacts.

```python
class ArtifactBuilder:
    """Builder for creating artifacts"""
    
    def __init__(self):
        self._artifact_type = None
        self._content = None
        self._metadata = {}
        self._version = "1.0.0"
        self._description = ""
    
    def of_type(self, artifact_type: ArtifactType) -> "ArtifactBuilder":
        """Set artifact type"""
        self._artifact_type = artifact_type
        return self
    
    def with_content(self, content: any) -> "ArtifactBuilder":
        """Set content"""
        self._content = content
        return self
    
    def with_description(self, description: str) -> "ArtifactBuilder":
        """Set description"""
        self._description = description
        return self
    
    def with_version(self, version: str) -> "ArtifactBuilder":
        """Set version"""
        self._version = version
        return self
    
    def with_metadata(self, key: str, value: any) -> "ArtifactBuilder":
        """Add metadata"""
        self._metadata[key] = value
        return self
    
    def build(self) -> dict:
        """Build artifact"""
        artifact_id = f"{self._artifact_type.value}_{uuid.uuid4().hex[:8]}"
        
        return {
            "artifact_id": artifact_id,
            "type": self._artifact_type,
            "content": self._content,
            "description": self._description,
            "version": self._version,
            "metadata": self._metadata,
            "created_at": datetime.now()
        }

# Usage
artifact = (
    ArtifactBuilder()
    .of_type(ArtifactType.CODE)
    .with_content("def hello(): return 'Hello'")
    .with_description("Simple greeting function")
    .with_version("1.0.0")
    .with_metadata("language", "python")
    .with_metadata("author", "agent-1")
    .build()
)

print(f"Created artifact: {artifact['artifact_id']}")
```

## Versioning

Managing artifact versions over time.

### Semantic Versioning

```python
from typing import Optional
import re

class Version:
    """Semantic version"""
    
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch
    
    @staticmethod
    def parse(version_str: str) -> "Version":
        """Parse version string"""
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
        if not match:
            raise ValueError(f"Invalid version: {version_str}")
        
        return Version(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3))
        )
    
    def bump_major(self) -> "Version":
        """Bump major version"""
        return Version(self.major + 1, 0, 0)
    
    def bump_minor(self) -> "Version":
        """Bump minor version"""
        return Version(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> "Version":
        """Bump patch version"""
        return Version(self.major, self.minor, self.patch + 1)
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

class VersionedArtifact:
    """Artifact with version management"""
    
    def __init__(self, artifact_id: str, content: any, version: Version = None):
        self.artifact_id = artifact_id
        self.content = content
        self.version = version or Version(1, 0, 0)
        self.history = [{"version": str(self.version), "content": content}]
    
    def update(self, new_content: any, change_type: str = "patch") -> Version:
        """Update artifact with new version"""
        # Determine version bump
        if change_type == "major":
            self.version = self.version.bump_major()
        elif change_type == "minor":
            self.version = self.version.bump_minor()
        else:
            self.version = self.version.bump_patch()
        
        # Update content
        self.content = new_content
        
        # Record history
        self.history.append({
            "version": str(self.version),
            "content": new_content,
            "timestamp": datetime.now()
        })
        
        print(f"Updated to version {self.version}")
        return self.version
    
    def get_version(self, version: str) -> Optional[any]:
        """Get specific version"""
        for entry in self.history:
            if entry["version"] == version:
                return entry["content"]
        return None
    
    def list_versions(self) -> List[str]:
        """List all versions"""
        return [entry["version"] for entry in self.history]

# Usage
artifact = VersionedArtifact(
    artifact_id="doc_001",
    content="Initial content"
)

# Update with patch
artifact.update("Minor fix", change_type="patch")

# Update with minor change
artifact.update("New feature", change_type="minor")

# Update with breaking change
artifact.update("Complete rewrite", change_type="major")

# List versions
print(f"Versions: {artifact.list_versions()}")

# Get old version
old_content = artifact.get_version("1.0.0")
```

## Artifact Storage

Strategies for storing artifacts.

### File-Based Storage

```python
import os
import hashlib
import shutil
from pathlib import Path

class FileArtifactStorage:
    """File-based artifact storage"""
    
    def __init__(self, base_path: str = "./artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, artifact_id: str, content: bytes, metadata: dict = None) -> str:
        """Save artifact"""
        # Create directory for artifact
        artifact_dir = self.base_path / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save content
        content_path = artifact_dir / "content"
        with open(content_path, 'wb') as f:
            f.write(content)
        
        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        
        # Save metadata
        metadata = metadata or {}
        metadata['checksum'] = checksum
        metadata['size'] = len(content)
        metadata['created_at'] = datetime.now().isoformat()
        
        metadata_path = artifact_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved artifact {artifact_id} ({len(content)} bytes)")
        return str(content_path)
    
    def load(self, artifact_id: str) -> tuple[bytes, dict]:
        """Load artifact"""
        artifact_dir = self.base_path / artifact_id
        
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        
        # Load content
        content_path = artifact_dir / "content"
        with open(content_path, 'rb') as f:
            content = f.read()
        
        # Load metadata
        metadata_path = artifact_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify checksum
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != metadata.get('checksum'):
            raise ValueError("Checksum mismatch - artifact corrupted")
        
        return content, metadata
    
    def delete(self, artifact_id: str):
        """Delete artifact"""
        artifact_dir = self.base_path / artifact_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
            print(f"Deleted artifact: {artifact_id}")
    
    def list(self) -> List[str]:
        """List all artifacts"""
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

# Usage
storage = FileArtifactStorage()

# Save artifact
artifact_id = "report_001"
content = b"# Report\n\nThis is a report."
storage.save(
    artifact_id,
    content,
    metadata={"type": "markdown", "author": "agent-1"}
)

# Load artifact
loaded_content, metadata = storage.load(artifact_id)
print(f"Loaded: {loaded_content.decode()}")
print(f"Metadata: {metadata}")

# List artifacts
artifacts = storage.list()
print(f"Artifacts: {artifacts}")
```

### Content-Addressed Storage

```python
class ContentAddressedStorage:
    """Content-addressed storage (like Git)"""
    
    def __init__(self, base_path: str = "./cas"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _hash_content(self, content: bytes) -> str:
        """Calculate content hash"""
        return hashlib.sha256(content).hexdigest()
    
    def _get_path(self, content_hash: str) -> Path:
        """Get storage path for hash"""
        # Use first 2 chars as directory (like Git)
        return self.base_path / content_hash[:2] / content_hash[2:]
    
    def put(self, content: bytes) -> str:
        """Store content and return hash"""
        content_hash = self._hash_content(content)
        path = self._get_path(content_hash)
        
        # Only write if doesn't exist
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            print(f"Stored content: {content_hash}")
        else:
            print(f"Content already exists: {content_hash}")
        
        return content_hash
    
    def get(self, content_hash: str) -> bytes:
        """Retrieve content by hash"""
        path = self._get_path(content_hash)
        
        if not path.exists():
            raise FileNotFoundError(f"Content not found: {content_hash}")
        
        with open(path, 'rb') as f:
            return f.read()
    
    def exists(self, content_hash: str) -> bool:
        """Check if content exists"""
        return self._get_path(content_hash).exists()

class ArtifactWithCAS:
    """Artifact using content-addressed storage"""
    
    def __init__(self, storage: ContentAddressedStorage):
        self.storage = storage
        self.artifacts = {}  # artifact_id -> content_hash
    
    def save(self, artifact_id: str, content: bytes) -> str:
        """Save artifact"""
        # Store content
        content_hash = self.storage.put(content)
        
        # Map artifact ID to hash
        self.artifacts[artifact_id] = content_hash
        
        return content_hash
    
    def load(self, artifact_id: str) -> bytes:
        """Load artifact"""
        content_hash = self.artifacts.get(artifact_id)
        if not content_hash:
            raise ValueError(f"Artifact not found: {artifact_id}")
        
        return self.storage.get(content_hash)

# Usage
storage = ContentAddressedStorage()
artifact_mgr = ArtifactWithCAS(storage)

# Save same content twice
content = b"Hello, World!"
hash1 = artifact_mgr.save("greeting_1", content)
hash2 = artifact_mgr.save("greeting_2", content)

# Both point to same content
print(f"Same content hash: {hash1 == hash2}")

# Load
loaded = artifact_mgr.load("greeting_1")
print(f"Loaded: {loaded.decode()}")
```

## Artifact Dependencies

Tracking dependencies between artifacts.

### Dependency Graph

```python
from collections import defaultdict
from typing import Set

class ArtifactDependencyGraph:
    """Track artifact dependencies"""
    
    def __init__(self):
        self.dependencies = defaultdict(set)  # artifact_id -> set of dependencies
        self.dependents = defaultdict(set)    # artifact_id -> set of dependents
    
    def add_dependency(self, artifact_id: str, depends_on: str):
        """Add dependency"""
        self.dependencies[artifact_id].add(depends_on)
        self.dependents[depends_on].add(artifact_id)
        print(f"{artifact_id} depends on {depends_on}")
    
    def get_dependencies(self, artifact_id: str) -> Set[str]:
        """Get direct dependencies"""
        return self.dependencies[artifact_id]
    
    def get_all_dependencies(self, artifact_id: str) -> Set[str]:
        """Get all transitive dependencies"""
        visited = set()
        to_visit = [artifact_id]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            for dep in self.dependencies[current]:
                if dep not in visited:
                    to_visit.append(dep)
        
        visited.discard(artifact_id)
        return visited
    
    def get_dependents(self, artifact_id: str) -> Set[str]:
        """Get artifacts that depend on this one"""
        return self.dependents[artifact_id]
    
    def topological_sort(self) -> List[str]:
        """Get artifacts in dependency order"""
        in_degree = {}
        all_artifacts = set(self.dependencies.keys()) | set(self.dependents.keys())
        
        # Calculate in-degrees
        for artifact in all_artifacts:
            in_degree[artifact] = len(self.dependencies[artifact])
        
        # Find artifacts with no dependencies
        queue = [a for a in all_artifacts if in_degree[a] == 0]
        result = []
        
        while queue:
            artifact = queue.pop(0)
            result.append(artifact)
            
            # Decrease in-degree of dependents
            for dependent in self.dependents[artifact]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(all_artifacts):
            raise ValueError("Circular dependency detected")
        
        return result
    
    def visualize(self) -> str:
        """Generate ASCII visualization"""
        lines = []
        for artifact, deps in self.dependencies.items():
            for dep in deps:
                lines.append(f"{artifact} -> {dep}")
        return "\n".join(lines) if lines else "No dependencies"

# Usage
graph = ArtifactDependencyGraph()

# Add dependencies
graph.add_dependency("report", "analysis")
graph.add_dependency("report", "charts")
graph.add_dependency("analysis", "data")
graph.add_dependency("charts", "data")

# Get dependencies
deps = graph.get_all_dependencies("report")
print(f"Report depends on: {deps}")

# Get build order
order = graph.topological_sort()
print(f"Build order: {order}")

# Visualize
print("\nDependency Graph:")
print(graph.visualize())
```

## Real-World Applications

### Pattern 1: Code Generation Pipeline

```python
class CodeGenerationPipeline:
    """Pipeline for generating and managing code artifacts"""
    
    def __init__(self):
        self.registry = ArtifactRegistry()
        self.storage = FileArtifactStorage()
        self.dependencies = ArtifactDependencyGraph()
    
    def generate_module(
        self,
        module_name: str,
        specifications: dict,
        dependencies: List[str] = None
    ) -> str:
        """Generate code module"""
        # Generate code
        code = self._generate_code(specifications)
        
        # Create artifact
        artifact_id = f"module_{module_name}"
        
        # Save to storage
        storage_path = self.storage.save(
            artifact_id,
            code.encode(),
            metadata={"type": "python_module", "module_name": module_name}
        )
        
        # Register in registry
        self.registry.register(
            artifact_id=artifact_id,
            artifact_type="code",
            storage_path=storage_path,
            description=f"Generated module: {module_name}",
            metadata=specifications
        )
        
        # Track dependencies
        if dependencies:
            for dep in dependencies:
                self.dependencies.add_dependency(artifact_id, dep)
        
        return artifact_id
    
    def _generate_code(self, specifications: dict) -> str:
        """Generate code from specifications"""
        # Simple code generation
        functions = specifications.get("functions", [])
        
        code_lines = ["# Generated Module\n\n"]
        
        for func_spec in functions:
            func_name = func_spec["name"]
            params = func_spec.get("parameters", [])
            return_type = func_spec.get("return_type", "None")
            
            code_lines.append(f"def {func_name}({', '.join(params)}) -> {return_type}:")
            code_lines.append(f'    """Generated function: {func_name}"""')
            code_lines.append("    pass")
            code_lines.append("\n")
        
        return "\n".join(code_lines)
    
    def build_project(self, artifacts: List[str]) -> str:
        """Build project from artifacts in dependency order"""
        # Get build order
        build_order = self.dependencies.topological_sort()
        
        # Filter to requested artifacts
        build_order = [a for a in build_order if a in artifacts]
        
        # Assemble project
        project_code = []
        
        for artifact_id in build_order:
            content, metadata = self.storage.load(artifact_id)
            project_code.append(f"# {artifact_id}")
            project_code.append(content.decode())
            project_code.append("\n")
        
        return "\n".join(project_code)

# Usage
pipeline = CodeGenerationPipeline()

# Generate utilities module
utils_id = pipeline.generate_module(
    "utils",
    specifications={
        "functions": [
            {"name": "calculate", "parameters": ["x", "y"], "return_type": "float"}
        ]
    }
)

# Generate main module that depends on utils
main_id = pipeline.generate_module(
    "main",
    specifications={
        "functions": [
            {"name": "run", "parameters": [], "return_type": "None"}
        ]
    },
    dependencies=[utils_id]
)

# Build project
project = pipeline.build_project([main_id, utils_id])
print(project)
```

### Pattern 2: Multi-Stage Data Processing

```python
class DataProcessingPipeline:
    """Pipeline for data processing with artifact tracking"""
    
    def __init__(self):
        self.artifacts = {}
        self.metadata = {}
    
    def ingest_data(self, source: str) -> str:
        """Stage 1: Ingest raw data"""
        artifact_id = f"raw_data_{uuid.uuid4().hex[:8]}"
        
        # Simulate data ingestion
        df = pd.DataFrame({
            'id': range(100),
            'value': [i * 2 for i in range(100)]
        })
        
        self.artifacts[artifact_id] = DataArtifact(
            artifact_id=artifact_id,
            data=df,
            description=f"Raw data from {source}"
        )
        
        return artifact_id
    
    def clean_data(self, input_artifact_id: str) -> str:
        """Stage 2: Clean data"""
        input_artifact = self.artifacts[input_artifact_id]
        
        # Clean data
        df = input_artifact.data.copy()
        df = df.dropna()
        df = df.drop_duplicates()
        
        artifact_id = f"cleaned_data_{uuid.uuid4().hex[:8]}"
        self.artifacts[artifact_id] = DataArtifact(
            artifact_id=artifact_id,
            data=df,
            description="Cleaned data",
            metadata={"source": input_artifact_id}
        )
        
        return artifact_id
    
    def transform_data(self, input_artifact_id: str) -> str:
        """Stage 3: Transform data"""
        input_artifact = self.artifacts[input_artifact_id]
        
        # Transform
        df = input_artifact.data.copy()
        df['transformed_value'] = df['value'] ** 2
        
        artifact_id = f"transformed_data_{uuid.uuid4().hex[:8]}"
        self.artifacts[artifact_id] = DataArtifact(
            artifact_id=artifact_id,
            data=df,
            description="Transformed data",
            metadata={"source": input_artifact_id}
        )
        
        return artifact_id
    
    def aggregate_data(self, input_artifact_id: str) -> str:
        """Stage 4: Aggregate data"""
        input_artifact = self.artifacts[input_artifact_id]
        
        # Aggregate
        df = input_artifact.data.copy()
        result = {
            'total': df['transformed_value'].sum(),
            'mean': df['transformed_value'].mean(),
            'count': len(df)
        }
        
        artifact_id = f"aggregated_data_{uuid.uuid4().hex[:8]}"
        self.artifacts[artifact_id] = DataArtifact(
            artifact_id=artifact_id,
            data=result,
            description="Aggregated results",
            metadata={"source": input_artifact_id}
        )
        
        return artifact_id
    
    def run_pipeline(self, source: str) -> str:
        """Run complete pipeline"""
        print("Starting pipeline...")
        
        # Stage 1: Ingest
        raw_id = self.ingest_data(source)
        print(f"✓ Ingested data: {raw_id}")
        
        # Stage 2: Clean
        cleaned_id = self.clean_data(raw_id)
        print(f"✓ Cleaned data: {cleaned_id}")
        
        # Stage 3: Transform
        transformed_id = self.transform_data(cleaned_id)
        print(f"✓ Transformed data: {transformed_id}")
        
        # Stage 4: Aggregate
        final_id = self.aggregate_data(transformed_id)
        print(f"✓ Aggregated data: {final_id}")
        
        print("Pipeline complete!")
        return final_id
    
    def get_artifact(self, artifact_id: str) -> DataArtifact:
        """Get artifact by ID"""
        return self.artifacts.get(artifact_id)

# Usage
pipeline = DataProcessingPipeline()

# Run pipeline
final_artifact_id = pipeline.run_pipeline("database")

# Get final results
final = pipeline.get_artifact(final_artifact_id)
print(f"\nFinal results: {final.data}")
```

## Summary

Artifacts are tangible outputs that persist beyond agent execution:

**Key Concepts**:
- **Types**: Code, documents, data, models, images
- **Versioning**: Semantic versioning, history tracking
- **Storage**: File-based, content-addressed
- **Registry**: Central catalog of artifacts
- **Dependencies**: Track relationships between artifacts

**Patterns**:
- **Builder**: Fluent interface for creation
- **Registry**: Central catalog
- **CAS**: Content-addressed storage
- **Dependency Graph**: Track relationships

**Management**:
- Version control and history
- Metadata tracking
- Checksum verification
- Lifecycle management
- Distribution and sharing

**Best Practices**:
1. Use semantic versioning
2. Track metadata and provenance
3. Verify integrity with checksums
4. Manage dependencies explicitly
5. Implement content addressing when appropriate
6. Maintain audit trails
7. Clean up obsolete artifacts

Artifacts transform ephemeral agent work into durable, reusable assets.

## Next Steps

Explore related topics:

- **[Filesystem Context](filesystem-context.md)**: Using files for persistence
- **[State Management](state-management.md)**: Managing runtime state
- **[Workflow Patterns](workflow-patterns.md)**: Orchestrating artifact generation

Related areas:
- **[Memory Systems](../memory-systems/memory-architecture.md)**: Long-term storage
- **[Tool Use](../tool-use/function-calling.md)**: Producing artifacts via tools
