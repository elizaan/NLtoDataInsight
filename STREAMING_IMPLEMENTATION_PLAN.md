# Real-Time Streaming Implementation Plan

## Current Behavior (What You're Seeing)
✅ **Messages DO appear in chat** - Intent Analysis, Generating Insight, Temperature result, Data Summary
❌ **Messages appear ALL AT ONCE** - only after the entire LLM process completes (can take 30+ seconds)
❌ **No real-time updates** - user sees nothing while LLM is thinking

## Root Cause
- `agent.process_query()` is **synchronous** and **blocking**
- IntentParser calls OpenAI → waits for response → then InsightExtractor calls OpenAI → waits
- Routes.py builds `assistant_messages` and returns JSON only AFTER everything completes
- Frontend receives one big response at the end

## Solution Options

### Option 1: Server-Sent Events (SSE) ⭐ **RECOMMENDED**
**Pros:**
- Native browser support (EventSource API)
- One-way server→client streaming (perfect for our use case)
- Automatic reconnection
- Simple to implement

**Cons:**
- Requires new endpoint
- Requires refactoring agent to yield updates

**Implementation:**
1. Create `/api/chat/stream` endpoint that returns `text/event-stream`
2. Modify core_agent to yield progress updates
3. Frontend uses EventSource to listen for updates
4. Display messages as they arrive

### Option 2: WebSockets
**Pros:**
- Full duplex (two-way) communication
- Very flexible

**Cons:**
- Overkill for one-way streaming
- More complex to implement
- Need WebSocket server setup

### Option 3: Aggressive Polling
**Pros:**
- No backend changes needed
- Simple to implement

**Cons:**
- Inefficient (constant HTTP requests)
- Not truly real-time
- High server load

### Option 4: Background Task + Status Polling ⭐ **EASIER ALTERNATIVE**
**Pros:**
- Minimal changes to existing code
- No new protocols needed
- Can show incremental updates

**Cons:**
- Not truly streaming
- Some latency (polling interval)

**Implementation:**
1. Start agent processing in background thread
2. Agent writes progress to shared state/cache
3. Frontend polls `/api/chat/status/{task_id}` every 500ms
4. Display updates as they become available

## Quick Fix: Option 4 (Recommended for MVP)

### Backend Changes

#### 1. Add task tracking to routes.py
```python
import uuid
import threading
from collections import defaultdict

# Global task storage
active_tasks = {}  # {task_id: {status, messages, result}}

@app.route('/api/chat', methods=['POST'])
def chat():
    if action == 'continue_conversation':
        # Create task ID
        task_id = str(uuid.uuid4())
        active_tasks[task_id] = {
            'status': 'processing',
            'messages': [],
            'result': None
        }
        
        # Start processing in background
        def process_in_background():
            try:
                # Call agent
                result = agent.process_query(user_message, context=context)
                
                # Store result
                active_tasks[task_id]['result'] = result
                active_tasks[task_id]['status'] = 'completed'
            except Exception as e:
                active_tasks[task_id]['status'] = 'error'
                active_tasks[task_id]['error'] = str(e)
        
        thread = threading.Thread(target=process_in_background)
        thread.start()
        
        # Return task ID immediately
        return jsonify({
            'type': 'task_started',
            'task_id': task_id,
            'status': 'processing'
        })

@app.route('/api/chat/status/<task_id>', methods=['GET'])
def chat_status(task_id):
    if task_id not in active_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = active_tasks[task_id]
    return jsonify({
        'status': task['status'],
        'messages': task['messages'],
        'result': task.get('result')
    })
```

#### 2. Modify core_agent to report progress
```python
def process_query_with_intent(self, user_message, context, progress_callback=None):
    # After intent parsing
    intent_result = self.intent_parser.parse_intent(user_message, context)
    if progress_callback:
        progress_callback('intent_parsed', intent_result)
    
    # After insight extraction
    if intent_type == 'PARTICULAR':
        result = self._handle_particular_exploration(...)
        if progress_callback:
            progress_callback('insight_generated', result.get('insight_result'))
    
    return result
```

### Frontend Changes

#### 1. Start task and poll for updates
```javascript
async sendMessage() {
    const message = this.chatInput.value.trim();
    if (!message) return;
    
    this.addMessage(message, 'user');
    
    // Start task
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: 'continue_conversation', message})
    });
    
    const data = await response.json();
    
    if (data.type === 'task_started') {
        // Start polling
        this.pollTaskStatus(data.task_id);
    }
}

async pollTaskStatus(taskId) {
    const interval = setInterval(async () => {
        const response = await fetch(`/api/chat/status/${taskId}`);
        const data = await response.json();
        
        // Display new messages
        data.messages.forEach(msg => {
            if (!this.displayedMessages.has(msg.id)) {
                this.displayMessage(msg);
                this.displayedMessages.add(msg.id);
            }
        });
        
        // Check if complete
        if (data.status === 'completed') {
            clearInterval(interval);
            this.handleFinalResult(data.result);
        } else if (data.status === 'error') {
            clearInterval(interval);
            this.addMessage('Error: ' + data.error, 'bot');
        }
    }, 500);  // Poll every 500ms
}
```

## Implementation Priority

### Phase 1: Minimum Viable Streaming (1-2 hours)
1. Add task tracking to routes.py
2. Add status endpoint
3. Add frontend polling
4. Test with simple progress messages

### Phase 2: Agent Progress Callbacks (2-3 hours)
1. Modify IntentParser to accept callback
2. Modify InsightExtractor to accept callback
3. Pass callbacks through from routes.py
4. Report progress at each LLM call

### Phase 3: Polish (1 hour)
1. Add loading spinners
2. Add progress indicators
3. Handle errors gracefully
4. Clean up completed tasks after 5 minutes

## Alternative: Keep Current Behavior, Add Progress Indicator

**Simplest fix if real-time isn't critical:**

```javascript
// Show "thinking" animation while waiting
async sendMessage() {
    this.addMessage(message, 'user');
    const thinkingMsg = this.addThinkingAnimation();
    
    const response = await fetch('/api/chat', {...});
    const data = await response.json();
    
    this.removeThinkingAnimation(thinkingMsg);
    this.handleApiResponse(data);
}
```

## Your Current Working State

✅ **What's Working:**
- Intent Analysis appears in chat
- Generating Insight message appears
- Actual insight text appears
- Data Summary appears
- All LLM outputs are captured and displayed

✅ **What's Missing:**
- Real-time updates (everything comes at once)
- Progress indication while LLM is thinking
- User sees blank screen for 30-60 seconds

## Recommendation

**For now, since messages ARE appearing correctly:**
1. Add a "thinking" animation/spinner while processing
2. Maybe add a progress message in system logs that shows "Intent parsing... Generating insight... Querying data..."
3. Consider Phase 1 implementation above for true streaming (if needed)

The core functionality is working - you just need to decide if real-time streaming is worth the implementation effort vs. showing a nice loading state.
