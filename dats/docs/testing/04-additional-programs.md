# Additional Test Programs

Once the calculator test passes, use these progressively complex programs to stress test the system.

---

## Test Program 2: URL Shortener API (Medium Complexity)

**Submit:**
```bash
dats submit --project test-urlshortener --mode autonomous \
  "Create a Python Flask API for URL shortening. It should have endpoints to: 
   1) POST /shorten - accept a URL and return a short code
   2) GET /<code> - redirect to original URL
   3) GET /stats/<code> - return click count
   Use SQLite for storage. Include proper validation and tests."
```

**Tests Decomposition Because:**
- Multiple endpoints (parallel subtasks)
- Database integration (dependency chain)
- External library usage (Flask, SQLite)

**Validates:**
- [ ] Larger task decomposition
- [ ] Database schema design decisions
- [ ] API design consistency across endpoints
- [ ] Integration between components

**Functional Test:**
```bash
# Start the server
python app.py &

# Shorten a URL
curl -X POST http://localhost:5000/shorten -d '{"url": "https://example.com/very/long/url"}'
# Returns: {"code": "abc123", "short_url": "http://localhost:5000/abc123"}

# Access shortened URL
curl -I http://localhost:5000/abc123
# Returns: 302 redirect to original

# Check stats
curl http://localhost:5000/stats/abc123
# Returns: {"clicks": 1, "created": "..."}
```

---

## Test Program 3: File Watcher with Notifications (Multi-Domain)

**Submit:**
```bash
dats submit --project test-filewatcher --mode autonomous \
  "Create a Python program that watches a directory for file changes.
   When a file is created, modified, or deleted, log the event and 
   optionally send a desktop notification. Use watchdog library.
   Include a simple CLI to start/stop watching and configure the directory.
   Include tests using mock filesystem events."
```

**Tests Because:**
- Requires code-embedded thinking (filesystem, events)
- CLI design
- External library integration
- Testing with mocks

**Validates:**
- [ ] Domain-appropriate routing (embedded concepts)
- [ ] Async/event-driven code generation
- [ ] Mock-based test generation

---

## Test Program 4: Markdown Blog Generator (Documentation Domain)

**Submit:**
```bash
dats submit --project test-bloggen --mode autonomous \
  "Create a static blog generator that:
   1) Reads markdown files from a /posts directory
   2) Converts them to HTML with a template
   3) Generates an index page listing all posts
   4) Outputs to /public directory
   Include a sample template and CSS. Include tests."
```

**Tests Because:**
- Documentation/content focus
- Template handling
- File I/O patterns
- Multiple output artifacts

**Validates:**
- [ ] Documentation worker involvement
- [ ] UI design for templates/CSS
- [ ] Multi-file output coordination

---

## Test Program 5: Collaborative Design Session (Collaborative Mode)

**Submit:**
```bash
dats submit --project test-collaborative --mode collaborative \
  "I want to build a personal finance tracker. Help me design it.
   It should track income, expenses, and show spending by category.
   I'm not sure if it should be a CLI tool, web app, or mobile app."
```

**Expected Behavior:**
- System asks clarifying questions
- Presents architecture options
- Waits for human input at decision points
- Adapts based on responses

**Tests:**
- [ ] Collaborative mode detection
- [ ] Human checkpoint creation
- [ ] Response to human input
- [ ] Iterative refinement

---

## Test Program 6: Refactor Existing Code (Edit Mode)

**Setup:** Create a deliberately messy Python file:
```python
# /tmp/messy_code.py
def calc(a,b,op):
    if op=="add":
        return a+b
    elif op=="sub":
        return a-b
    elif op=="mul":
        return a*b
    elif op=="div":
        if b==0:
            return "error"
        return a/b
    else:
        return "unknown"
```

**Submit:**
```bash
dats submit --project test-refactor --mode autonomous \
  --input /tmp/messy_code.py \
  "Refactor this code to follow Python best practices:
   - Add type hints
   - Use proper exception handling
   - Add docstrings
   - Use an enum for operations
   - Add input validation
   Maintain the same functionality."
```

**Validates:**
- [ ] Existing code analysis
- [ ] LightRAG ingestion of existing code
- [ ] Refactoring without breaking functionality
- [ ] Style and convention enforcement

---

## Test Matrix Summary

| Test | Complexity | Domains | Mode | Key Validation |
|------|------------|---------|------|----------------|
| Calculator | Simple | code-general | Autonomous | Core pipeline |
| URL Shortener | Medium | code-general, architecture | Autonomous | Multi-component |
| File Watcher | Medium | code-embedded, code-general | Autonomous | Cross-domain routing |
| Blog Generator | Medium | code-general, documentation, ui | Autonomous | Multi-domain |
| Finance Tracker | Design | architecture | Collaborative | Human interaction |
| Refactor | Edit | code-general, analysis | Autonomous | Existing code handling |