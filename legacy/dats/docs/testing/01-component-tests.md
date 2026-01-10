# Component-Level Test Cases

## Test Category 1: Model Client Connectivity

### Test 1.1: Ollama Small Model Connection
**Purpose:** Verify connection to small model Ollama instance

**Steps:**
1. Initialize OllamaClient with endpoint http://192.168.1.12:11434
2. Send simple prompt: "Say hello"
3. Verify response received

**Expected Result:**
- Response contains text
- tokens_input > 0
- tokens_output > 0
- No connection errors

**Command:**
```bash
dats health --component ollama-small
```

---

### Test 1.2: Ollama Embedding Connection
**Purpose:** Verify embedding model works

**Steps:**
1. Initialize embedding client with endpoint http://192.168.1.11:11434
2. Request embedding for text: "This is a test sentence"
3. Verify embedding vector returned

**Expected Result:**
- Vector returned with expected dimensions
- No errors

**Command:**
```bash
dats health --component embedding
```

---

### Test 1.3: vLLM Connection
**Purpose:** Verify vLLM large model endpoint

**Steps:**
1. Initialize OpenAI-compatible client with endpoint http://192.168.1.11:8000/v1
2. Send prompt: "What is 2+2?"
3. Verify response

**Expected Result:**
- Response contains "4"
- OpenAI-compatible response format

**Command:**
```bash
dats health --component vllm
```

---

### Test 1.4: Anthropic Connection (if configured)
**Purpose:** Verify frontier model access

**Steps:**
1. Initialize Anthropic client
2. Send simple prompt
3. Verify response

**Expected Result:**
- Response received
- API key valid

**Command:**
```bash
dats health --component anthropic
```

---

## Test Category 2: Celery Queue Operations

### Test 2.1: Redis Connection
**Purpose:** Verify Celery broker accessible

**Steps:**
1. Attempt Redis connection
2. Set and get test key

**Expected Result:**
- Connection successful
- Read/write works

**Command:**
```bash
dats health --component redis
```

---

### Test 2.2: Queue Task Submission
**Purpose:** Verify tasks can be queued

**Steps:**
1. Submit dummy task to each queue (tiny, small, large, frontier)
2. Verify task appears in queue
3. Verify task can be retrieved

**Expected Result:**
- Task ID returned for each
- Tasks visible in queue inspection

**Command:**
```bash
dats queues --test-submit
```

---

### Test 2.3: Worker Task Pickup
**Purpose:** Verify workers pick up tasks

**Steps:**
1. Start worker for specific queue
2. Submit task to that queue
3. Verify worker picks up and processes

**Expected Result:**
- Task status changes from queued → in_progress → completed
- Result stored in backend

**Command:**
```bash
# Terminal 1: Start worker
celery -A src.queue.celery_app worker -Q small --loglevel=info

# Terminal 2: Submit test
dats test queue-pickup --tier small
```

---

## Test Category 3: Prompt System

### Test 3.1: Prompt Loading
**Purpose:** Verify all prompts load correctly

**Steps:**
1. Load each prompt template from /prompts directory
2. Verify no syntax errors
3. Verify all expected variables present

**Expected Result:**
- All prompts load
- Variable placeholders identified

**Command:**
```bash
dats test prompts --validate
```

---

### Test 3.2: Prompt Rendering
**Purpose:** Verify prompt templates render with variables

**Steps:**
1. Load worker prompt template
2. Provide sample variables
3. Render template
4. Verify placeholders replaced

**Expected Result:**
- No unreplaced {variable} patterns
- Valid prompt text produced

**Command:**
```bash
dats test prompts --render code_general --sample-data
```

---

## Test Category 4: LightRAG Integration

### Test 4.1: Document Insertion
**Purpose:** Verify content can be embedded

**Steps:**
1. Insert test document: "The ProjectX API uses REST endpoints for user management"
2. Verify insertion acknowledged

**Expected Result:**
- Document ID returned
- No errors

**Command:**
```bash
dats test lightrag --insert "Test document content"
```

---

### Test 4.2: Context Retrieval
**Purpose:** Verify semantic search works

**Steps:**
1. Insert known document (from 4.1)
2. Query: "How does ProjectX handle users?"
3. Verify relevant content returned

**Expected Result:**
- Query returns the inserted document
- Relevance score reasonable

**Command:**
```bash
dats test lightrag --query "How does ProjectX handle users?"
```

---

### Test 4.3: Batch Embedding Trigger
**Purpose:** Verify batch embedding process

**Steps:**
1. Add multiple items to embedding queue
2. Trigger flush (or wait for threshold)
3. Verify all items embedded

**Expected Result:**
- All queued items processed
- Items queryable after flush

**Command:**
```bash
dats test lightrag --batch-embed --count 5
```

---

## Test Category 5: Provenance System

### Test 5.1: Record Creation
**Purpose:** Verify provenance records can be created

**Steps:**
1. Create mock task completion
2. Record provenance with inputs, outputs, model used
3. Retrieve record by ID

**Expected Result:**
- Record persisted
- All fields retrievable

**Command:**
```bash
dats test provenance --create-record
```

---

### Test 5.2: Forward Traversal
**Purpose:** Verify dependent tracking works

**Steps:**
1. Create chain: A → B → C (A is input to B, B is input to C)
2. Query dependents of A
3. Verify B and C returned

**Expected Result:**
- Both B and C identified as dependents
- Correct depth levels

**Command:**
```bash
dats test provenance --traversal forward
```

---

### Test 5.3: Taint Propagation
**Purpose:** Verify taint spreads correctly

**Steps:**
1. Create chain: A → B → C
2. Mark A as tainted
3. Verify B and C marked as suspect

**Expected Result:**
- A marked tainted
- B and C marked suspect
- Revalidation tasks queued

**Command:**
```bash
dats test provenance --taint-test