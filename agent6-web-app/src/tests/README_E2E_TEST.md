# End-to-End Workflow Test

## Purpose
This test exercises the complete pipeline from dataset summarization → intent parsing → insight generation → query execution → plot generation, with conversation context across multiple queries.

## Test Queries
The test includes 7 queries that demonstrate different capabilities:

1. **q1**: "show me temperature change for two months" - Basic temporal aggregation
2. **q2**: "what is the highest temperature in full dataset?" - Max value (likely timeout)
3. **q3**: "when was this highest temperature seen?" - References q2 result
4. **q4**: "where was this highest temperature seen?" - References q2/q3 results
5. **q5**: "what is the minimum and maximum salinity in data?" - Different variable
6. **q6**: "in which time steps was the salinity highest?" - References q5
7. **q7**: "any plot to reveal eddy formation in mediterranean sea?" - Geographic region + visualization

## Running the Test

### Prerequisites
```bash
# Ensure you have OpenAI API key set
export OPENAI_API_KEY="your-api-key-here"

# Navigate to the project root
cd /Users/ishratjahaneliza/Documents/PhD/Valerio/codes/NLQtoDataInsight/agent6-web-app
```

### Execute Test
```bash
python3 src/tests/test_workflow_e2e.py
```

### Expected Behavior
- **q1**: Should succeed with 1-2 plots showing temperature trends
- **q2**: May timeout (scanning 10,366 timesteps × large spatial grid)
  - If timeout, LLM will suggest smaller queries (e.g., monthly aggregation, specific region)
- **q3-q4**: Should reference q2's result (or suggest smaller scope if q2 timed out)
- **q5**: Should succeed with min/max salinity values
- **q6**: Should reference q5's max salinity and identify timesteps
- **q7**: Should use geographic mapping to find Mediterranean Sea coordinates and generate eddy visualization

### Output Structure
Each query produces:
- **Query code**: `ai_data/codes/dyamond_llc2160/query_YYYYMMDD_HHMMSS.py`
- **Plot code**: `ai_data/codes/dyamond_llc2160/plots_YYYYMMDD_HHMMSS.py`
- **Data cache**: `ai_data/data_cache/dyamond_llc2160/data_YYYYMMDD_HHMMSS.npz`
- **Plots**: `ai_data/plots/dyamond_llc2160/plot_*_YYYYMMDD_HHMMSS.png`
- **Insight**: `ai_data/insights/dyamond_llc2160/insight_YYYYMMDD_HHMMSS.txt`

### Conversation Context
The test maintains a `ConversationContext` object that:
- Tracks all previous query results
- Generates a text summary for each new query
- Passes this summary to the LLM so it can reference past results (e.g., "the max temperature from query #2 was 30.5°C")

## Debugging
If a query fails, check:
1. The query code file for syntax/API errors
2. System logs for detailed error messages
3. Whether the data cache file was created
4. OpenVisus API binding errors (common: wrong argument types for `ds.db.read()`)

## Timeout Handling
When a query times out (>5 minutes):
- The system aborts execution
- LLM generates 4 concrete smaller-query suggestions
- Suggestions include: short description, why it helps, estimated cost, code hint
- User can pick one suggestion and retry
