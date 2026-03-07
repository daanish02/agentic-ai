# API Integration

## Table of Contents

- [Introduction](#introduction)
- [REST API Basics](#rest-api-basics)
- [Authentication Methods](#authentication-methods)
- [Request Construction](#request-construction)
- [Response Parsing](#response-parsing)
- [Pagination](#pagination)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Caching Strategies](#caching-strategies)
- [Webhooks and Callbacks](#webhooks-and-callbacks)
- [Streaming Responses](#streaming-responses)
- [GraphQL Integration](#graphql-integration)
- [API Versioning](#api-versioning)
- [Testing and Mocking](#testing-and-mocking)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Modern agents interact with the world through APIs. Whether fetching data, triggering actions, or integrating services, API integration is fundamental.

This guide covers **practical patterns** for integrating external APIs as agent tools:

- **Authentication**: API keys, OAuth, JWT tokens
- **Request handling**: Building requests, handling parameters
- **Response parsing**: JSON, XML, error handling
- **Pagination**: Fetching all results across multiple pages
- **Rate limiting**: Respecting API limits
- **Caching**: Reducing redundant calls
- **Testing**: Mocking APIs for development

> "The best API is one you never see breaking."  
> -- API Design wisdom

## REST API Basics

Understanding REST API fundamentals.

### HTTP Methods

```python
import requests
from typing import Any, Dict, Optional

class RESTClient:
    """
    Basic REST API client.
    """

    def __init__(self, base_url: str, default_headers: Dict[str, str] = None):
        self.base_url = base_url.rstrip('/')
        self.default_headers = default_headers or {}

    def get(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> requests.Response:
        """GET request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.default_headers, **(headers or {})}

        return requests.get(url, params=params, headers=headers)

    def post(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> requests.Response:
        """POST request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.default_headers, **(headers or {})}

        return requests.post(url, data=data, json=json_data, headers=headers)

    def put(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> requests.Response:
        """PUT request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.default_headers, **(headers or {})}

        return requests.put(url, data=data, json=json_data, headers=headers)

    def patch(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> requests.Response:
        """PATCH request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.default_headers, **(headers or {})}

        return requests.patch(url, data=data, json=json_data, headers=headers)

    def delete(
        self,
        endpoint: str,
        headers: Dict[str, str] = None
    ) -> requests.Response:
        """DELETE request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.default_headers, **(headers or {})}

        return requests.delete(url, headers=headers)


# Usage
client = RESTClient(
    base_url="https://api.example.com",
    default_headers={"User-Agent": "MyAgent/1.0"}
)

response = client.get("users", params={"page": 1, "limit": 10})
```

### Status Code Handling

```python
def handle_response(response: requests.Response) -> Dict[str, Any]:
    """
    Handle HTTP response with appropriate error handling.
    """
    # Success codes
    if 200 <= response.status_code < 300:
        try:
            return response.json()
        except ValueError:
            return {"data": response.text}

    # Client errors (4xx)
    elif 400 <= response.status_code < 500:
        if response.status_code == 400:
            raise ValueError(f"Bad request: {response.text}")
        elif response.status_code == 401:
            raise PermissionError("Unauthorized - check credentials")
        elif response.status_code == 403:
            raise PermissionError("Forbidden - insufficient permissions")
        elif response.status_code == 404:
            raise FileNotFoundError(f"Resource not found: {response.url}")
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 60)
            raise Exception(f"Rate limit exceeded. Retry after {retry_after}s")
        else:
            raise Exception(f"Client error {response.status_code}: {response.text}")

    # Server errors (5xx)
    elif 500 <= response.status_code < 600:
        raise Exception(f"Server error {response.status_code}: {response.text}")

    else:
        raise Exception(f"Unexpected status code {response.status_code}")
```

## Authentication Methods

Different ways to authenticate API requests.

### API Key Authentication

```python
class APIKeyAuth:
    """
    API Key authentication.
    """

    def __init__(self, api_key: str, key_name: str = "X-API-Key"):
        self.api_key = api_key
        self.key_name = key_name

    def get_headers(self) -> Dict[str, str]:
        """Get headers with API key."""
        return {self.key_name: self.api_key}


# Usage
auth = APIKeyAuth(api_key="sk-abc123", key_name="Authorization")

client = RESTClient(
    base_url="https://api.example.com",
    default_headers=auth.get_headers()
)
```

### Bearer Token Authentication

```python
class BearerTokenAuth:
    """
    Bearer token authentication (JWT, OAuth).
    """

    def __init__(self, token: str):
        self.token = token

    def get_headers(self) -> Dict[str, str]:
        """Get headers with bearer token."""
        return {"Authorization": f"Bearer {self.token}"}


# Usage
auth = BearerTokenAuth(token="eyJhbGciOiJIUzI1NiIsInR...")

client = RESTClient(
    base_url="https://api.example.com",
    default_headers=auth.get_headers()
)
```

### OAuth 2.0

```python
from datetime import datetime, timedelta
from typing import Optional

class OAuth2Client:
    """
    OAuth 2.0 client with automatic token refresh.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: str = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope

        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None

    def get_access_token(self) -> str:
        """
        Get access token, refreshing if necessary.
        """
        # Check if token is valid
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token

        # Get new token
        self._fetch_token()

        return self.access_token

    def _fetch_token(self):
        """Fetch new access token from OAuth server."""
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        if self.scope:
            data["scope"] = self.scope

        response = requests.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()

        self.access_token = token_data["access_token"]

        # Calculate expiration (with 5 minute buffer)
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.now() + timedelta(
            seconds=expires_in - 300
        )

    def get_headers(self) -> Dict[str, str]:
        """Get headers with current access token."""
        token = self.get_access_token()
        return {"Authorization": f"Bearer {token}"}


# Usage
oauth = OAuth2Client(
    client_id="my_client_id",
    client_secret="my_client_secret",
    token_url="https://oauth.example.com/token",
    scope="read write"
)

# Automatically handles token refresh
for i in range(100):
    headers = oauth.get_headers()
    response = requests.get("https://api.example.com/data", headers=headers)
```

### Basic Authentication

```python
import base64

class BasicAuth:
    """
    HTTP Basic Authentication.
    """

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def get_headers(self) -> Dict[str, str]:
        """Get headers with basic auth."""
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()

        return {"Authorization": f"Basic {encoded}"}


# Usage
auth = BasicAuth(username="user", password="pass")

# Or use requests built-in
response = requests.get(
    "https://api.example.com/data",
    auth=("user", "pass")
)
```

## Request Construction

Building API requests from agent parameters.

### Parameter Mapping

```python
from typing import Any, Dict, List

class ParameterMapper:
    """
    Map agent parameters to API parameters.
    """

    def __init__(self):
        self.mappings: Dict[str, str] = {}
        self.transformers: Dict[str, callable] = {}

    def add_mapping(
        self,
        agent_param: str,
        api_param: str,
        transformer: callable = None
    ):
        """
        Map agent parameter name to API parameter name.
        Optionally apply transformation.
        """
        self.mappings[agent_param] = api_param

        if transformer:
            self.transformers[agent_param] = transformer

    def map_parameters(
        self,
        agent_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert agent parameters to API parameters.
        """
        api_params = {}

        for agent_param, value in agent_params.items():
            # Get API parameter name
            api_param = self.mappings.get(agent_param, agent_param)

            # Apply transformation if defined
            if agent_param in self.transformers:
                value = self.transformers[agent_param](value)

            api_params[api_param] = value

        return api_params


# Example: Weather API
mapper = ParameterMapper()

mapper.add_mapping("location", "q")  # location -> q
mapper.add_mapping("units", "units", lambda u: u.lower())  # Celsius -> celsius
mapper.add_mapping(
    "date",
    "dt",
    lambda d: int(d.timestamp())  # datetime -> unix timestamp
)

agent_params = {
    "location": "London",
    "units": "Celsius",
    "date": datetime(2024, 1, 1)
}

api_params = mapper.map_parameters(agent_params)
# Result: {"q": "London", "units": "celsius", "dt": 1704067200}
```

### Query String Building

```python
from urllib.parse import urlencode

def build_query_string(params: Dict[str, Any]) -> str:
    """
    Build URL query string from parameters.
    """
    # Filter out None values
    filtered = {k: v for k, v in params.items() if v is not None}

    # Handle lists
    processed = {}
    for key, value in filtered.items():
        if isinstance(value, list):
            # Multiple values for same key
            processed[key] = value
        elif isinstance(value, bool):
            # Convert boolean to lowercase string
            processed[key] = str(value).lower()
        else:
            processed[key] = value

    return urlencode(processed, doseq=True)


# Usage
params = {
    "q": "search term",
    "categories": ["tech", "science"],
    "include_archived": False,
    "page": 1,
    "limit": None  # Will be filtered out
}

query = build_query_string(params)
# Result: "q=search+term&categories=tech&categories=science&include_archived=false&page=1"
```

### Request Body Construction

```python
import json

class RequestBodyBuilder:
    """
    Build request body for POST/PUT requests.
    """

    @staticmethod
    def build_json(data: Dict[str, Any]) -> str:
        """Build JSON request body."""
        return json.dumps(data)

    @staticmethod
    def build_form(data: Dict[str, Any]) -> str:
        """Build form-encoded request body."""
        return urlencode(data)

    @staticmethod
    def build_multipart(
        data: Dict[str, Any],
        files: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build multipart form data."""
        # Use requests' files parameter
        form_data = data.copy()

        if files:
            # Files are separate
            return {"data": form_data, "files": files}

        return {"data": form_data}


# JSON body
body = RequestBodyBuilder.build_json({
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30
})

response = requests.post(
    "https://api.example.com/users",
    data=body,
    headers={"Content-Type": "application/json"}
)

# Form body
body = RequestBodyBuilder.build_form({
    "username": "john",
    "password": "secret"
})

response = requests.post(
    "https://api.example.com/login",
    data=body,
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

# Multipart with file upload
multipart = RequestBodyBuilder.build_multipart(
    data={"description": "Profile picture"},
    files={"photo": open("photo.jpg", "rb")}
)

response = requests.post(
    "https://api.example.com/upload",
    **multipart
)
```

## Response Parsing

Extracting data from API responses.

### JSON Response Parser

```python
from typing import Any, List, Optional

class JSONParser:
    """
    Parse JSON API responses.
    """

    @staticmethod
    def parse(response: requests.Response) -> Dict[str, Any]:
        """Parse JSON response."""
        try:
            return response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    @staticmethod
    def extract_field(
        data: Dict[str, Any],
        path: str,
        default: Any = None
    ) -> Any:
        """
        Extract nested field using dot notation.

        Example: extract_field(data, "user.profile.name")
        """
        keys = path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    @staticmethod
    def extract_list(
        data: Dict[str, Any],
        path: str
    ) -> List[Any]:
        """Extract list from nested field."""
        result = JSONParser.extract_field(data, path, default=[])

        if not isinstance(result, list):
            raise ValueError(f"Field '{path}' is not a list")

        return result


# Usage
response_data = {
    "status": "success",
    "data": {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ],
        "metadata": {
            "total": 2,
            "page": 1
        }
    }
}

# Extract nested field
total = JSONParser.extract_field(response_data, "data.metadata.total")
# Result: 2

# Extract list
users = JSONParser.extract_list(response_data, "data.users")
# Result: [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
```

### XML Response Parser

```python
import xml.etree.ElementTree as ET

class XMLParser:
    """
    Parse XML API responses.
    """

    @staticmethod
    def parse(response: requests.Response) -> ET.Element:
        """Parse XML response."""
        try:
            return ET.fromstring(response.text)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML response: {e}")

    @staticmethod
    def to_dict(element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}

        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib

        # Add text content
        if element.text and element.text.strip():
            result['text'] = element.text.strip()

        # Add children
        for child in element:
            child_data = XMLParser.to_dict(child)

            # Handle multiple children with same tag
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result if result else element.text


# Usage
xml_response = """
<response>
    <status>success</status>
    <data>
        <user id="1">
            <name>Alice</name>
            <email>alice@example.com</email>
        </user>
        <user id="2">
            <name>Bob</name>
            <email>bob@example.com</email>
        </user>
    </data>
</response>
"""

root = XMLParser.parse(xml_response)
data = XMLParser.to_dict(root)
```

### Response Normalization

```python
class ResponseNormalizer:
    """
    Normalize responses from different APIs to common format.
    """

    def __init__(self, mapping: Dict[str, str]):
        """
        Args:
            mapping: Map API field names to normalized names
                     {"api_field": "normalized_field"}
        """
        self.mapping = mapping

    def normalize(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize API response."""
        normalized = {}

        for api_field, normalized_field in self.mapping.items():
            value = JSONParser.extract_field(response, api_field)

            if value is not None:
                normalized[normalized_field] = value

        return normalized


# Example: Normalize different weather APIs to common format
openweather_normalizer = ResponseNormalizer({
    "main.temp": "temperature",
    "main.humidity": "humidity",
    "weather.0.description": "condition",
    "name": "location"
})

weatherapi_normalizer = ResponseNormalizer({
    "current.temp_c": "temperature",
    "current.humidity": "humidity",
    "current.condition.text": "condition",
    "location.name": "location"
})

# Normalize different API responses to same format
openweather_data = openweather_normalizer.normalize(openweather_response)
weatherapi_data = weatherapi_normalizer.normalize(weatherapi_response)
# Both result in: {"temperature": 20, "humidity": 65, "condition": "Clear", ...}
```

## Pagination

Fetching all results across multiple pages.

### Offset-Based Pagination

```python
from typing import Iterator, List, Any

def fetch_all_offset_pagination(
    client: RESTClient,
    endpoint: str,
    limit: int = 100,
    max_pages: int = None
) -> List[Dict[str, Any]]:
    """
    Fetch all results using offset-based pagination.
    """
    all_results = []
    offset = 0
    page = 0

    while True:
        # Check max pages limit
        if max_pages and page >= max_pages:
            break

        # Fetch page
        response = client.get(
            endpoint,
            params={"limit": limit, "offset": offset}
        )

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        all_results.extend(results)
        offset += limit
        page += 1

        # Check if more pages available
        total = data.get("total")
        if total and offset >= total:
            break

    return all_results


# Usage
client = RESTClient("https://api.example.com")
all_users = fetch_all_offset_pagination(client, "users", limit=50)
```

### Page-Based Pagination

```python
def fetch_all_page_pagination(
    client: RESTClient,
    endpoint: str,
    page_size: int = 100,
    max_pages: int = None
) -> List[Dict[str, Any]]:
    """
    Fetch all results using page number pagination.
    """
    all_results = []
    page = 1

    while True:
        # Check max pages limit
        if max_pages and page > max_pages:
            break

        # Fetch page
        response = client.get(
            endpoint,
            params={"page": page, "per_page": page_size}
        )

        data = response.json()
        results = data.get("items", [])

        if not results:
            break

        all_results.extend(results)

        # Check if more pages
        if not data.get("has_next", True):
            break

        page += 1

    return all_results
```

### Cursor-Based Pagination

```python
def fetch_all_cursor_pagination(
    client: RESTClient,
    endpoint: str,
    limit: int = 100,
    max_pages: int = None
) -> List[Dict[str, Any]]:
    """
    Fetch all results using cursor-based pagination.
    """
    all_results = []
    cursor = None
    page = 0

    while True:
        # Check max pages limit
        if max_pages and page >= max_pages:
            break

        # Build params
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        # Fetch page
        response = client.get(endpoint, params=params)

        data = response.json()
        results = data.get("data", [])

        if not results:
            break

        all_results.extend(results)

        # Get next cursor
        cursor = data.get("next_cursor")
        if not cursor:
            break

        page += 1

    return all_results
```

### Iterator Pattern

```python
class PaginatedIterator:
    """
    Iterator for paginated API results.
    """

    def __init__(
        self,
        client: RESTClient,
        endpoint: str,
        limit: int = 100,
        pagination_type: str = "offset"
    ):
        self.client = client
        self.endpoint = endpoint
        self.limit = limit
        self.pagination_type = pagination_type

        self.offset = 0
        self.page = 1
        self.cursor = None
        self.buffer = []
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        """Get next item."""
        # Refill buffer if empty
        if not self.buffer and not self.exhausted:
            self._fetch_next_page()

        if not self.buffer:
            raise StopIteration

        return self.buffer.pop(0)

    def _fetch_next_page(self):
        """Fetch next page of results."""
        # Build params based on pagination type
        if self.pagination_type == "offset":
            params = {"limit": self.limit, "offset": self.offset}
        elif self.pagination_type == "page":
            params = {"page": self.page, "per_page": self.limit}
        elif self.pagination_type == "cursor":
            params = {"limit": self.limit}
            if self.cursor:
                params["cursor"] = self.cursor
        else:
            raise ValueError(f"Unknown pagination type: {self.pagination_type}")

        # Fetch
        response = self.client.get(self.endpoint, params=params)
        data = response.json()

        # Extract results
        results = data.get("data", data.get("results", data.get("items", [])))

        if not results:
            self.exhausted = True
            return

        self.buffer.extend(results)

        # Update pagination state
        if self.pagination_type == "offset":
            self.offset += len(results)
        elif self.pagination_type == "page":
            self.page += 1
        elif self.pagination_type == "cursor":
            self.cursor = data.get("next_cursor")
            if not self.cursor:
                self.exhausted = True


# Usage
client = RESTClient("https://api.example.com")
iterator = PaginatedIterator(
    client,
    "users",
    limit=50,
    pagination_type="cursor"
)

for user in iterator:
    print(user)
```

## Rate Limiting

Respecting API rate limits.

### Rate Limit Aware Client

```python
from datetime import datetime, timedelta
import time

class RateLimitedClient(RESTClient):
    """
    REST client with automatic rate limiting.
    """

    def __init__(
        self,
        base_url: str,
        max_requests: int,
        time_window: int,
        default_headers: Dict[str, str] = None
    ):
        super().__init__(base_url, default_headers)

        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = []  # List of request timestamps

    def _wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = datetime.now()

        # Remove old requests outside time window
        cutoff = now - timedelta(seconds=self.time_window)
        self.requests = [ts for ts in self.requests if ts > cutoff]

        # Check if at limit
        if len(self.requests) >= self.max_requests:
            # Calculate wait time
            oldest = self.requests[0]
            wait_until = oldest + timedelta(seconds=self.time_window)
            wait_seconds = (wait_until - now).total_seconds()

            if wait_seconds > 0:
                print(f"Rate limit: waiting {wait_seconds:.2f}s...")
                time.sleep(wait_seconds + 0.1)

    def _record_request(self):
        """Record a request."""
        self.requests.append(datetime.now())

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """GET with rate limiting."""
        self._wait_if_needed()
        response = super().get(endpoint, **kwargs)
        self._record_request()

        # Check for rate limit in response
        self._handle_rate_limit_response(response)

        return response

    def _handle_rate_limit_response(self, response: requests.Response):
        """Handle 429 Too Many Requests."""
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 60)

            try:
                retry_after = int(retry_after)
            except ValueError:
                retry_after = 60

            print(f"Rate limit exceeded. Waiting {retry_after}s...")
            time.sleep(retry_after)


# Usage
client = RateLimitedClient(
    base_url="https://api.example.com",
    max_requests=100,
    time_window=60  # 100 requests per minute
)

# Client automatically handles rate limiting
for i in range(200):
    response = client.get("data")
```

## Error Handling

Handling API errors gracefully (see also [Error Handling](error-handling.md)).

### Retry Wrapper

```python
from functools import wraps
import time

def api_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    retryable_statuses: List[int] = None
):
    """
    Decorator for retrying API calls.
    """
    if retryable_statuses is None:
        retryable_statuses = [429, 500, 502, 503, 504]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except requests.HTTPError as e:
                    last_exception = e
                    response = e.response

                    # Check if retryable
                    if response.status_code not in retryable_statuses:
                        raise

                    # Don't retry on last attempt
                    if attempt >= max_attempts - 1:
                        raise

                    # Calculate backoff
                    delay = backoff_base ** attempt
                    print(f"Attempt {attempt + 1} failed. Retrying in {delay}s...")
                    time.sleep(delay)

                except (requests.ConnectionError, requests.Timeout) as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt >= max_attempts - 1:
                        raise

                    delay = backoff_base ** attempt
                    print(f"Network error. Retrying in {delay}s...")
                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# Usage
@api_retry(max_attempts=3, backoff_base=2.0)
def fetch_user(user_id: int) -> Dict[str, Any]:
    """Fetch user with automatic retry."""
    response = requests.get(f"https://api.example.com/users/{user_id}")
    response.raise_for_status()
    return response.json()
```

### Comprehensive Error Handler

```python
class APIError(Exception):
    """Base API error."""

    def __init__(
        self,
        message: str,
        status_code: int = None,
        response: requests.Response = None
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class APIErrorHandler:
    """
    Handle API errors with proper classification.
    """

    @staticmethod
    def handle_response(response: requests.Response) -> Dict[str, Any]:
        """
        Handle response and raise appropriate errors.
        """
        try:
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            status = response.status_code

            # Extract error message from response
            try:
                error_data = response.json()
                error_msg = error_data.get('message', error_data.get('error', str(e)))
            except ValueError:
                error_msg = response.text or str(e)

            # Client errors
            if status == 400:
                raise APIError(f"Bad request: {error_msg}", status, response)
            elif status == 401:
                raise APIError("Authentication failed", status, response)
            elif status == 403:
                raise APIError("Permission denied", status, response)
            elif status == 404:
                raise APIError("Resource not found", status, response)
            elif status == 422:
                raise APIError(f"Validation error: {error_msg}", status, response)
            elif status == 429:
                retry_after = response.headers.get('Retry-After', 'unknown')
                raise APIError(
                    f"Rate limit exceeded. Retry after: {retry_after}",
                    status,
                    response
                )

            # Server errors
            elif status >= 500:
                raise APIError(f"Server error: {error_msg}", status, response)

            # Other errors
            else:
                raise APIError(f"HTTP error {status}: {error_msg}", status, response)

        except requests.ConnectionError as e:
            raise APIError(f"Connection error: {e}")

        except requests.Timeout as e:
            raise APIError(f"Request timeout: {e}")

        except ValueError as e:
            # JSON parsing error
            raise APIError(f"Invalid response format: {e}")
```

## Caching Strategies

Reducing redundant API calls.

### Simple Cache

```python
from typing import Any, Optional
from datetime import datetime, timedelta

class APICache:
    """
    Simple in-memory cache for API responses.
    """

    def __init__(self, ttl: int = 300):
        """
        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, datetime]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self.cache:
            return None

        value, timestamp = self.cache[key]

        # Check if expired
        if datetime.now() - timestamp > timedelta(seconds=self.ttl):
            del self.cache[key]
            return None

        return value

    def set(self, key: str, value: Any):
        """Cache a value."""
        self.cache[key] = (value, datetime.now())

    def clear(self):
        """Clear all cached values."""
        self.cache.clear()

    def _make_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Make cache key from endpoint and parameters."""
        import hashlib
        import json

        # Sort params for consistent keys
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{endpoint}:{sorted_params}"

        return hashlib.md5(key_string.encode()).hexdigest()


class CachedRESTClient(RESTClient):
    """
    REST client with response caching.
    """

    def __init__(
        self,
        base_url: str,
        cache_ttl: int = 300,
        default_headers: Dict[str, str] = None
    ):
        super().__init__(base_url, default_headers)
        self.cache = APICache(ttl=cache_ttl)

    def get(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        use_cache: bool = True,
        **kwargs
    ) -> requests.Response:
        """GET with caching."""
        if use_cache:
            # Check cache
            cache_key = self.cache._make_key(endpoint, params or {})
            cached = self.cache.get(cache_key)

            if cached is not None:
                print(f"Cache hit for {endpoint}")
                # Return mock response with cached data
                mock_response = requests.Response()
                mock_response._content = json.dumps(cached).encode()
                mock_response.status_code = 200
                return mock_response

        # Not in cache - fetch from API
        response = super().get(endpoint, params=params, **kwargs)

        if use_cache and response.status_code == 200:
            try:
                data = response.json()
                cache_key = self.cache._make_key(endpoint, params or {})
                self.cache.set(cache_key, data)
            except ValueError:
                pass  # Not JSON, don't cache

        return response


# Usage
client = CachedRESTClient(
    base_url="https://api.example.com",
    cache_ttl=300  # 5 minutes
)

# First call - fetches from API
response1 = client.get("users", params={"page": 1})

# Second call - returns cached result
response2 = client.get("users", params={"page": 1})
```

## Webhooks and Callbacks

Handling asynchronous API notifications.

### Webhook Server

```python
from flask import Flask, request, jsonify
from typing import Callable

app = Flask(__name__)

class WebhookHandler:
    """
    Handle incoming webhooks.
    """

    def __init__(self):
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """Register handler for webhook event."""
        self.handlers[event_type] = handler

    def handle(self, event_type: str, payload: Dict[str, Any]) -> Any:
        """Handle webhook event."""
        handler = self.handlers.get(event_type)

        if not handler:
            raise ValueError(f"No handler registered for event: {event_type}")

        return handler(payload)


webhook_handler = WebhookHandler()


@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint."""
    data = request.json

    event_type = data.get('event')
    payload = data.get('payload', {})

    try:
        result = webhook_handler.handle(event_type, payload)
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# Register handlers
def handle_user_created(payload):
    user_id = payload['user_id']
    print(f"User created: {user_id}")
    # Process user creation
    return {"processed": True}


def handle_payment_received(payload):
    amount = payload['amount']
    print(f"Payment received: ${amount}")
    # Process payment
    return {"processed": True}


webhook_handler.register_handler('user.created', handle_user_created)
webhook_handler.register_handler('payment.received', handle_payment_received)


# Run webhook server
# app.run(port=5000)
```

## Streaming Responses

Handling streaming API responses.

### SSE (Server-Sent Events)

```python
def stream_sse(url: str, headers: Dict[str, str] = None):
    """
    Stream Server-Sent Events.
    """
    response = requests.get(url, headers=headers, stream=True)

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')

            # Parse SSE format
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix

                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    yield data


# Usage
# for event in stream_sse("https://api.example.com/stream"):
#     print(event)
```

### Chunked Responses

```python
def stream_chunks(url: str, chunk_size: int = 8192):
    """
    Stream response in chunks.
    """
    response = requests.get(url, stream=True)

    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            yield chunk


# Usage - download large file
# with open('large_file.bin', 'wb') as f:
#     for chunk in stream_chunks("https://api.example.com/large-file"):
#         f.write(chunk)
```

## GraphQL Integration

Working with GraphQL APIs.

### GraphQL Client

```python
class GraphQLClient:
    """
    Simple GraphQL client.
    """

    def __init__(self, endpoint: str, headers: Dict[str, str] = None):
        self.endpoint = endpoint
        self.headers = headers or {}

    def execute(
        self,
        query: str,
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query.
        """
        payload = {"query": query}

        if variables:
            payload["variables"] = variables

        response = requests.post(
            self.endpoint,
            json=payload,
            headers=self.headers
        )

        response.raise_for_status()
        data = response.json()

        # Check for GraphQL errors
        if "errors" in data:
            errors = data["errors"]
            raise Exception(f"GraphQL errors: {errors}")

        return data.get("data", {})

    def query(
        self,
        query: str,
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute GraphQL query."""
        full_query = f"query {{ {query} }}"
        return self.execute(full_query, variables)

    def mutation(
        self,
        mutation: str,
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute GraphQL mutation."""
        full_mutation = f"mutation {{ {mutation} }}"
        return self.execute(full_mutation, variables)


# Usage
client = GraphQLClient(
    endpoint="https://api.example.com/graphql",
    headers={"Authorization": "Bearer token123"}
)

# Query
query = """
    user(id: $userId) {
        id
        name
        email
        posts {
            title
            content
        }
    }
"""

result = client.execute(query, variables={"userId": 123})
user = result["user"]

# Mutation
mutation = """
    createUser(input: $input) {
        user {
            id
            name
        }
    }
"""

result = client.mutation(
    mutation,
    variables={"input": {"name": "John", "email": "john@example.com"}}
)
```

## API Versioning

Handling API version changes.

### Version-Aware Client

```python
class VersionedAPIClient(RESTClient):
    """
    API client that handles versioning.
    """

    def __init__(
        self,
        base_url: str,
        version: str,
        version_format: str = "header",  # or "path", "query"
        default_headers: Dict[str, str] = None
    ):
        super().__init__(base_url, default_headers)
        self.version = version
        self.version_format = version_format

    def _add_version(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> tuple[str, Dict[str, Any], Dict[str, str]]:
        """Add version to request."""
        params = params or {}
        headers = headers or {}

        if self.version_format == "header":
            headers["API-Version"] = self.version

        elif self.version_format == "path":
            endpoint = f"v{self.version}/{endpoint}"

        elif self.version_format == "query":
            params["version"] = self.version

        return endpoint, params, headers

    def get(self, endpoint: str, params: Dict[str, Any] = None, **kwargs):
        """GET with version."""
        endpoint, params, headers = self._add_version(
            endpoint,
            params,
            kwargs.get('headers')
        )

        return super().get(endpoint, params=params, headers=headers)


# Usage
client = VersionedAPIClient(
    base_url="https://api.example.com",
    version="2",
    version_format="path"
)

# Makes request to https://api.example.com/v2/users
response = client.get("users")
```

## Testing and Mocking

Testing API integrations without hitting real APIs.

### Mock API Responses

```python
from unittest.mock import Mock, patch

def test_api_integration():
    """Test API integration with mocked responses."""

    # Create mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com"
    }

    # Patch requests.get
    with patch('requests.get', return_value=mock_response):
        client = RESTClient("https://api.example.com")
        response = client.get("users/1")

        data = response.json()

        assert data["id"] == 1
        assert data["name"] == "Test User"


def test_error_handling():
    """Test error handling with mocked errors."""

    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError()

    with patch('requests.get', return_value=mock_response):
        client = RESTClient("https://api.example.com")

        with pytest.raises(requests.HTTPError):
            response = client.get("users/999")
            response.raise_for_status()
```

### Mock Server

```python
from flask import Flask, jsonify, request

def create_mock_server():
    """Create mock API server for testing."""
    app = Flask(__name__)

    # Mock data
    users = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]

    @app.route('/users', methods=['GET'])
    def get_users():
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))

        start = (page - 1) * limit
        end = start + limit

        return jsonify({
            "users": users[start:end],
            "total": len(users),
            "page": page
        })

    @app.route('/users/<int:user_id>', methods=['GET'])
    def get_user(user_id):
        user = next((u for u in users if u["id"] == user_id), None)

        if user:
            return jsonify(user)
        else:
            return jsonify({"error": "User not found"}), 404

    return app


# Run mock server
# app = create_mock_server()
# app.run(port=5000)
```

## Real-World Examples

Complete examples of common API integrations.

### Weather API Integration

```python
class WeatherAPI:
    """
    Integration with weather API.
    """

    def __init__(self, api_key: str):
        self.client = RESTClient("https://api.openweathermap.org/data/2.5")
        self.api_key = api_key

    def get_current_weather(
        self,
        location: str,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get current weather for a location.
        """
        response = self.client.get(
            "weather",
            params={
                "q": location,
                "units": units,
                "appid": self.api_key
            }
        )

        data = response.json()

        # Normalize response
        return {
            "location": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }

    def get_forecast(
        self,
        location: str,
        days: int = 5,
        units: str = "metric"
    ) -> List[Dict[str, Any]]:
        """
        Get weather forecast.
        """
        response = self.client.get(
            "forecast",
            params={
                "q": location,
                "cnt": days * 8,  # 8 forecasts per day
                "units": units,
                "appid": self.api_key
            }
        )

        data = response.json()

        # Parse forecast
        forecast = []
        for item in data["list"]:
            forecast.append({
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "description": item["weather"][0]["description"],
                "humidity": item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"]
            })

        return forecast


# Usage
weather = WeatherAPI(api_key="your_api_key")

current = weather.get_current_weather("London", units="metric")
print(f"Temperature in {current['location']}: {current['temperature']}°C")

forecast = weather.get_forecast("London", days=3)
for day in forecast:
    print(f"{day['datetime']}: {day['temperature']}°C, {day['description']}")
```

### GitHub API Integration

```python
class GitHubAPI:
    """
    Integration with GitHub API.
    """

    def __init__(self, token: str):
        self.client = RESTClient(
            base_url="https://api.github.com",
            default_headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json"
            }
        )

    def get_user(self, username: str) -> Dict[str, Any]:
        """Get user information."""
        response = self.client.get(f"users/{username}")
        return response.json()

    def list_repos(
        self,
        username: str,
        sort: str = "updated"
    ) -> List[Dict[str, Any]]:
        """List user repositories."""
        response = self.client.get(
            f"users/{username}/repos",
            params={"sort": sort, "per_page": 100}
        )
        return response.json()

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str
    ) -> Dict[str, Any]:
        """Create an issue."""
        response = self.client.post(
            f"repos/{owner}/{repo}/issues",
            json_data={
                "title": title,
                "body": body
            }
        )
        return response.json()

    def search_code(
        self,
        query: str,
        max_results: int = 30
    ) -> List[Dict[str, Any]]:
        """Search code across GitHub."""
        all_results = []
        page = 1

        while len(all_results) < max_results:
            response = self.client.get(
                "search/code",
                params={
                    "q": query,
                    "per_page": min(100, max_results - len(all_results)),
                    "page": page
                }
            )

            data = response.json()
            items = data.get("items", [])

            if not items:
                break

            all_results.extend(items)
            page += 1

        return all_results[:max_results]


# Usage
github = GitHubAPI(token="ghp_...")

user = github.get_user("octocat")
print(f"User: {user['name']}, Repos: {user['public_repos']}")

repos = github.list_repos("octocat")
for repo in repos[:5]:
    print(f"- {repo['name']}: {repo['description']}")
```

## Best Practices

Guidelines for robust API integrations.

**1. Handle Rate Limits**: Implement client-side rate limiting and respect Retry-After headers

**2. Use Retries Wisely**: Retry transient errors (5xx, timeouts) but not client errors (4xx)

**3. Cache When Possible**: Cache responses to reduce API calls and improve performance

**4. Validate Inputs**: Validate parameters before making API calls

**5. Log Everything**: Log requests, responses, and errors for debugging

**6. Handle Pagination**: Fetch all results when needed, with limits to prevent runaway

**7. Normalize Responses**: Map different API formats to common internal format

**8. Test with Mocks**: Test integration logic without hitting real APIs

**9. Version Explicitly**: Specify API version to avoid breaking changes

**10. Secure Credentials**: Never hardcode API keys, use environment variables

## Summary

Integrating external APIs is essential for agent capabilities:

**Authentication**:
- API keys for simple services
- OAuth 2.0 for user-delegated access
- JWT tokens for stateless auth
- Bearer tokens for modern APIs

**Request Handling**:
- Parameter mapping from agent to API format
- Request body construction (JSON, form, multipart)
- Query string building with proper encoding

**Response Processing**:
- Parse JSON, XML, or other formats
- Extract nested fields
- Normalize to common format

**Pagination**:
- Offset-based (offset + limit)
- Page-based (page number)
- Cursor-based (next token)
- Iterator pattern for clean consumption

**Resilience**:
- Rate limiting to respect API quotas
- Retry with exponential backoff
- Caching to reduce calls
- Error handling with proper classification

**Testing**:
- Mock responses for unit tests
- Mock servers for integration tests
- Test error scenarios explicitly

Build robust agent tools by treating API integration as a first-class concern.

## Next Steps

Continue building:

- [Error Handling](error-handling.md) - Handling API failures
- [Tool Schemas](tool-schemas.md) - Describing API tools to agents
- [Tool Chaining](tool-chaining.md) - Combining multiple APIs
- [ReAct](../agent-architectures/react.md) - Reasoning about API usage
