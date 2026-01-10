# Integration Test Cases

## Test Category 6: Agent Pipeline Integration

### Test 6.1: Coordinator → Decomposer Handoff
**Purpose:** Verify Coordinator correctly hands off to Decomposer

**Steps:**
1. Submit request: "Create a hello world Python script"
2. Verify Coordinator produces initial decomposition task
3. Verify Decomposer receives and processes

**Expected Result:**
- Coordinator output has correct schema
- Decomposer input matches Coordinator output
- Subtasks generated

**Validation Points:**
- [ ] Coordinator sets mode correctly
- [ ] Initial task description passed to Decomposer
- [ ] Decomposer creates at least one subtask

**Command:**
```bash
dats test integration --stage coordinator-decomposer --request "Create a hello world Python script"
```

---

### Test 6.2: Decomposer → Complexity Estimator Flow
**Purpose:** Verify subtasks get routed correctly

**Steps:**
1. Use subtask from Test 6.1
2. Pass through Complexity Estimator
3. Verify tier and model assignment

**Expected Result:**
- Each subtask gets tier assignment
- Context window estimate reasonable
- Capability domains assigned

**Validation Points:**
- [ ] Tier is one of: tiny, small, large, frontier
- [ ] Model matches tier
- [ ] Context estimate < tier's safe limit

**Command:**
```bash
dats test integration --stage decomposer-estimator
```

---

### Test 6.3: Full Routing to Queue
**Purpose:** Verify tasks land in correct queues

**Steps:**
1. Submit variety of tasks (simple, medium, complex)
2. Verify each lands in appropriate queue

**Expected Result:**
- Simple tasks → tiny or small queue
- Complex tasks → large queue
- Ambiguous tasks → frontier queue

**Validation Points:**
- [ ] Queue name matches assigned tier
- [ ] Task payload contains all required fields

**Command:**
```bash
dats test integration --stage full-routing
```

---

### Test 6.4: Worker Execution with Context
**Purpose:** Verify worker gets LightRAG context

**Steps:**
1. Seed LightRAG with project context: "This project uses pytest for testing"
2. Submit task: "Write a test for the hello function"
3. Verify worker prompt includes LightRAG context

**Expected Result:**
- Worker prompt contains seeded context
- Output references pytest (not unittest or other)

**Validation Points:**
- [ ] LightRAG query executed
- [ ] Context injected into prompt
- [ ] Output reflects context awareness

**Command:**
```bash
dats test integration --stage worker-context
```

---

### Test 6.5: Worker → QA Pipeline
**Purpose:** Verify outputs go through QA

**Steps:**
1. Configure task with qa_profile: consensus
2. Execute task
3. Verify QA runs after completion

**Expected Result:**
- QA task triggered
- Multiple reviewers invoked (consensus)
- QA result recorded

**Validation Points:**
- [ ] QA task created with correct profile
- [ ] At least 2 reviewers for consensus
- [ ] Verdict recorded (PASS/FAIL)

**Command:**
```bash
dats test integration --stage worker-qa
```

---

### Test 6.6: QA Pass → Embedding
**Purpose:** Verify passed outputs get embedded

**Steps:**
1. Execute task that passes QA
2. Verify embedding trigger fires
3. Query LightRAG for new content

**Expected Result:**
- Output embedded after QA pass
- Content queryable in LightRAG

**Validation Points:**
- [ ] Embedding queue received item
- [ ] After flush, content is queryable
- [ ] Provenance marked as passed

**Command:**
```bash
dats test integration --stage qa-embedding
```

---

### Test 6.7: QA Fail → Re-queue
**Purpose:** Verify failed outputs get re-queued

**Steps:**
1. Submit task designed to fail QA (intentionally flawed)
2. Verify QA fails
3. Verify task re-queued with revision guidance

**Expected Result:**
- QA returns NEEDS_REVISION
- Task re-queued with guidance attached
- Attempt counter incremented

**Validation Points:**
- [ ] Failure detected
- [ ] Revision guidance present in re-queued task
- [ ] attempts < max_attempts

**Command:**
```bash
dats test integration --stage qa-requeue
```

---

## Test Category 7: Failure Handling Integration

### Test 7.1: Worker Refuses Task
**Purpose:** Verify refuse/re-queue works

**Steps:**
1. Submit complex task to tiny tier (force mismatch)
2. Verify worker refuses
3. Verify task escalated to higher tier

**Expected Result:**
- Worker returns REFUSE status
- Task re-queued to higher tier
- Eventually completes at appropriate tier

**Validation Points:**
- [ ] Refuse reason captured
- [ ] New tier assignment correct
- [ ] Task eventually succeeds

**Command:**
```bash
dats test integration --stage worker-refuse
```

---

### Test 7.2: Cascade Failure Handling
**Purpose:** Verify taint propagation and revalidation

**Steps:**
1. Create task chain: parent → child1, child2, child3
2. Complete all tasks
3. Mark parent output as flawed
4. Verify children marked suspect
5. Verify revalidation tasks created

**Expected Result:**
- Parent marked tainted
- Children marked suspect
- Revalidation evaluates each child

**Validation Points:**
- [ ] Taint status correct
- [ ] Suspect status propagated
- [ ] Revalidation tasks queued
- [ ] Valid children cleared, invalid children tainted

**Command:**
```bash
dats test integration --stage cascade-failure
```

---

### Test 7.3: Human Review Checkpoint
**Purpose:** Verify human review pauses pipeline

**Steps:**
1. Submit task requiring human approval (architecture decision)
2. Verify system pauses at review point
3. Approve via API/CLI
4. Verify pipeline continues

**Expected Result:**
- Task reaches NEEDS_HUMAN status
- Review appears in review queue
- Approval allows continuation

**Validation Points:**
- [ ] Human checkpoint detected
- [ ] Review record created
- [ ] Approval updates status
- [ ] Downstream tasks proceed

**Command:**
```bash
# Submit
dats submit --require-approval "Design database schema for user management"

# Check review queue
dats review list

# Approve
dats review approve <review-id>

# Verify continuation
dats status <task-id>