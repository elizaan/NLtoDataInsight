# Testing Guide: LLM Outputs in Chat UI

## Overview
This guide will help you test that LLM agent outputs (IntentParser and InsightExtractor results) now appear in the chat conversation area.

## Pre-Test Checklist
- [x] Flask server is running at http://localhost:5000
- [x] Browser opened at http://localhost:5000
- [ ] Browser Developer Console is open (F12 or Cmd+Option+I)
- [ ] Terminal with Flask server output is visible
- [ ] System Logs panel (right side) is visible in UI

## Test Steps

### Test 1: Basic Query with PARTICULAR Intent

1. **Enter this query in the chat input:**
   ```
   show me average ocean current velocity magnitude over 2 days
   ```

2. **Expected Outputs in Browser Console:**
   ```
   ====================================
   HANDLING API RESPONSE
   Response type: particular_response
   Has assistant_messages: true
   assistant_messages length: 2
   ====================================
   🔥 Processing assistant_messages: 2
   🔍 Adding intent parsing message
   ✅ Intent message added
   💡 Adding insight generation message
   ✅ Insight messages added
   ✅ All assistant_messages processed
   ```

3. **Expected Outputs in Chat Area (left panel):**
   ```
   User: show me average ocean current velocity magnitude over 2 days
   
   Bot: 🔍 Intent Analysis:
   {
     "intent_type": "PARTICULAR",
     "confidence": 0.95,
     "plot_hints": {...},
     "reasoning": "User is asking a specific question about ocean current velocity..."
   }
   
   Bot: 💡 Generating Insight...
   
   Bot: [Insight text about ocean currents]
   
   Bot: 📊 Data Summary:
   {
     "mean": ...,
     "std": ...,
     ...
   }
   ```

4. **Expected Outputs in Terminal (Flask server):**
   ```
   ============================================================
   [ROUTES] Calling agent.process_query
   [ROUTES] User message: show me average ocean current velocity magnitude over 2 days
   ============================================================
   
   [Agent] Processing query with intent classification: show me average ocean current velocity magnitude...
   [Agent] Intent: PARTICULAR (confidence: 0.95)
   
   ============================================================
   [ROUTES] Agent returned result:
   [ROUTES] Result keys: ['type', 'intent', 'intent_result', 'insight_result', 'insight', ...]
   [ROUTES] Has intent_result: True
   [ROUTES] Has insight_result: True
   [ROUTES] Context flags after agent call:
   [ROUTES]   is_particular: True
   [ROUTES]   is_not_particular: False
   ============================================================
   
   [ROUTES] Building assistant_messages...
   [ROUTES]   intent_result extracted: True
   [ROUTES]   insight_result extracted: True
   [ROUTES] Added intent_parsing message: {...}
   [ROUTES] Added insight_generation message with keys: ['status', 'insight', 'data_summary', ...]
   [ROUTES] Final assistant_messages count: 2
   [ROUTES] Returning particular_response
   [ROUTES] Response data keys: ['type', 'message', 'answer', 'insight', 'data_summary', 'visualization', 'plot_files', 'assistant_messages', 'status']
   [ROUTES] assistant_messages in response: 2
   ```

5. **Expected Outputs in System Logs (right panel):**
   ```
   [11:23:45 AM] [continue_conversation] Calling agent.process_query with context
   [11:23:45 AM] [continue_conversation] User message: show me average ocean current velocity magnitude over 2 days
   [11:23:46 AM] [IntentParser] Set context['is_particular'] = True
   [11:23:46 AM] [continue_conversation] Agent returned result type: success
   ```

### Test 2: General Exploration Query (NOT_PARTICULAR Intent)

1. **Enter this query:**
   ```
   tell me something interesting about this dataset
   ```

2. **Expected Response Type:** `exploration_response`

3. **Should See:**
   - 🔍 Intent Analysis with `"intent_type": "NOT_PARTICULAR"`
   - 💡 Generating Insight...
   - Insight text
   - Data summary (if available)

### Test 3: Help Request

1. **Enter:**
   ```
   help
   ```

2. **Expected:**
   - Response type: `help_response`
   - May or may not have intent_result (depending on implementation)
   - Should show help text in chat

### Test 4: Unrelated Query

1. **Enter:**
   ```
   what's the weather today?
   ```

2. **Expected:**
   - Response type: `clarification`
   - Should see message asking to rephrase in data exploration terms

## Debugging Checklist

### If You Don't See Intent Analysis in Chat:

- [ ] Check browser console for errors
- [ ] Look for message: "⚠️ No assistant_messages in response or not an array"
- [ ] Check network tab - does API response include `assistant_messages`?
- [ ] Check terminal - does routes.py show "Final assistant_messages count: 2"?

### If You See Intent But Not Insight:

- [ ] Check if `insight_result` has a `status: 'error'`
- [ ] Look for errors in InsightExtractor in terminal
- [ ] Check if `insight_result.insight` exists

### If Messages Appear in System Logs Instead of Chat:

- [ ] This should NOT happen anymore - system logs and chat messages are separate
- [ ] System logs go to right panel (System Logs)
- [ ] Chat messages go to left panel (Chat Messages)

### If Network Request Fails:

- [ ] Check Flask server is running: `curl http://localhost:5000/api/datasets`
- [ ] Check terminal for errors
- [ ] Verify OpenAI API key is present

## Key Differences to Verify

### System Logs Panel (Right Side):
```
[11:23:45 AM] [continue_conversation] Calling agent.process_query with context
[11:23:45 AM] [IntentParser] Set context['is_particular'] = True
[11:23:46 AM] Summary retrieved successfully
```
- Timestamped
- Technical/diagnostic
- Component labels in brackets

### Chat Conversation (Left Side):
```
Bot: 🔍 Intent Analysis:
{
  "intent_type": "PARTICULAR",
  ...
}

Bot: 💡 Generating Insight...

Bot: The average ocean current velocity shows...
```
- No timestamps
- User-friendly
- Emoji indicators
- Formatted JSON
- Natural language insights

## Success Criteria

✅ **Test passes if:**
1. Intent Analysis JSON appears in chat area (left panel)
2. Insight text appears in chat area (left panel)
3. System logs appear only in System Logs panel (right panel)
4. Browser console shows assistant_messages being processed
5. Terminal shows routes.py building assistant_messages
6. No JavaScript errors in browser console
7. No Python errors in terminal

❌ **Test fails if:**
1. Intent/Insight appear in System Logs instead of chat
2. No LLM outputs visible anywhere
3. JavaScript errors in console
4. Python errors in terminal
5. assistant_messages is empty or null

## Quick Verification Commands

### Check if server is running:
```bash
curl http://localhost:5000/api/datasets
```

### Watch server logs in real-time:
```bash
tail -f /Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app/server_debug.log
```

### Check for errors in code:
```bash
cd /Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight
python -m py_compile agent6-web-app/src/api/routes.py
python -m py_compile agent6-web-app/src/agents/core_agent.py
```

## Next Steps After Testing

1. If tests pass: Document any UI improvements needed (formatting, styling)
2. If tests fail: Share browser console output AND terminal output for debugging
3. Consider optional enhancements:
   - Collapsible JSON sections
   - Syntax highlighting for JSON
   - Loading spinners during LLM calls
   - Copy button for JSON output
