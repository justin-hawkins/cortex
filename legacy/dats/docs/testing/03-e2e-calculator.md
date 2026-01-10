# End-to-End Test: Build a Simple CLI Calculator

## Overview

This test validates the entire system by having it build a complete, working program.

**Target Program:** A command-line calculator that:
- Accepts two numbers and an operation
- Supports add, subtract, multiply, divide
- Handles errors (division by zero, invalid input)
- Includes tests

**Why This Test:**
- Simple enough to complete quickly
- Complex enough to require decomposition
- Testable output (we can run the program)
- Multiple file outputs (code + tests)

---

## Test Execution

### Step 1: Submit the Request
```bash
dats submit --project test-calculator --mode autonomous \
  "Create a Python command-line calculator that takes two numbers and an operation (add, subtract, multiply, divide) as arguments. Include proper error handling for invalid inputs and division by zero. Include pytest tests for all operations and error cases."
```

**Expected Immediate Response:**
```
✓ Request submitted
  Task ID: calc-001
  Mode: autonomous
  Status: decomposing
```

---

### Step 2: Observe Decomposition
```bash
dats status calc-001 --tree
```

**Expected Decomposition (approximate):**
```
calc-001: Create CLI calculator
├── calc-001-1: Design module structure
├── calc-001-2: Implement calculator core
│   ├── calc-001-2a: Implement add operation
│   ├── calc-001-2b: Implement subtract operation
│   ├── calc-001-2c: Implement multiply operation
│   └── calc-001-2d: Implement divide operation (with zero check)
├── calc-001-3: Implement CLI interface
│   ├── calc-001-3a: Argument parsing
│   └── calc-001-3b: Error handling and output
├── calc-001-4: Write tests
│   ├── calc-001-4a: Tests for operations
│   └── calc-001-4b: Tests for error cases
└── calc-001-5: Integration and documentation
```

**Validation Points:**
- [ ] Decomposition creates logical subtasks
- [ ] Dependencies make sense (2 depends on 1, 4 depends on 2 and 3)
- [ ] Subtasks are atomic (single function/file each)

---

### Step 3: Monitor Execution
```bash
dats status calc-001 --watch
```

**Expected Progress:**
```
[00:00] calc-001-1: Design module structure → queued (small)
[00:05] calc-001-1: Design module structure → in_progress
[00:15] calc-001-1: Design module structure → validating
[00:20] calc-001-1: Design module structure → completed ✓
[00:20] calc-001-2a: Implement add operation → queued (tiny)
[00:20] calc-001-2b: Implement subtract operation → queued (tiny)
...
```

**Validation Points:**
- [ ] Tasks execute in dependency order
- [ ] Parallel tasks run concurrently where possible
- [ ] Each task goes through QA
- [ ] No tasks stuck in queue

---

### Step 4: Verify Outputs

After completion:
```bash
dats status calc-001
```

**Expected:**
```
Task: calc-001
Status: completed
Duration: ~5-10 minutes

Outputs:
  - calculator/
      - __init__.py
      - operations.py
      - cli.py
  - tests/
      - test_operations.py
      - test_cli.py
  - README.md

Files committed to: github.com/user/test-calculator
```

---

### Step 5: Functional Validation

Clone and test the output:
```bash
# Clone the output repository
cd /tmp
git clone <output-repo-url> test-calculator
cd test-calculator

# Install dependencies (if any)
pip install -e .

# Run the calculator
python -m calculator 5 3 add
# Expected: 8

python -m calculator 10 4 subtract  
# Expected: 6

python -m calculator 7 6 multiply
# Expected: 42

python -m calculator 15 3 divide
# Expected: 5.0

# Test error handling
python -m calculator 10 0 divide
# Expected: Error message about division by zero

python -m calculator abc 5 add
# Expected: Error message about invalid input

# Run the tests
pytest tests/ -v
# Expected: All tests pass
```

**Validation Points:**
- [ ] All operations produce correct results
- [ ] Division by zero handled gracefully
- [ ] Invalid input handled gracefully
- [ ] All tests pass
- [ ] Code is readable and follows conventions

---

### Step 6: Verify Provenance
```bash
dats provenance export calc-001 --format json > provenance.json
```

**Expected Provenance:**
- Every subtask has a record
- Inputs and outputs linked correctly
- Models used recorded
- QA results attached

**Validation Points:**
- [ ] All tasks have provenance records
- [ ] Parent-child relationships correct
- [ ] QA verdicts recorded
- [ ] Model versions recorded

---

### Step 7: Verify LightRAG Content
```bash
dats test lightrag --query "How does the calculator handle division by zero?"
```

**Expected:**
- Returns content about the divide operation
- References the error handling code
- Demonstrates project knowledge was embedded

---

## Success Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Request accepted and decomposed | Task tree visible | Error or no decomposition |
| All subtasks complete | Status: completed | Any task stuck or failed |
| Code compiles/runs | `python -m calculator` works | Import or runtime errors |
| Correct arithmetic results | All 4 operations correct | Wrong results |
| Error handling works | Graceful error messages | Crashes or stack traces |
| Tests pass | pytest exits 0 | Test failures |
| Provenance complete | All records present | Missing records |
| LightRAG updated | Relevant query results | Empty or unrelated results |

---

## Troubleshooting

### If decomposition fails:
- Check Coordinator logs
- Verify gpt-oss:120b accessible
- Check prompt template loading

### If tasks stuck in queue:
- Verify workers running: `dats workers`
- Check queue depths: `dats queues`
- Verify model endpoints accessible

### If QA keeps failing:
- Check QA reviewer logs
- Verify acceptance criteria reasonable
- May need to adjust QA thresholds

### If code doesn't work:
- Check worker output in provenance
- Review QA feedback if any
- May indicate prompt template issues

---

## Cleanup
```bash
dats project delete test-calculator --confirm