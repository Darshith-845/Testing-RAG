#!/bin/bash

# Endpoint
URL="http://127.0.0.1:8000/query"

# Video ID for testing
VIDEO_ID="video_001"

# Array of test questions
declare -a questions=(
    "Explain variables simply."
    "Where does the video talk about conditional statements?"
    "Summarize the video in 3 points."
    "Give me an analogy for loops."
    "What are data structures according to the video?"
    "Explain the whole video in detail."
    "Ask me a quiz question about variables."
)

echo "==============================="
echo "Running Evaluation on $URL"
echo "==============================="

for q in "${questions[@]}"
do
    echo -e "\n🔹 Question: $q"
    curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"$q\", \"video_id\":\"$VIDEO_ID\"}" \
        | jq '.'
done

echo -e "\n✅ Evaluation complete!"
