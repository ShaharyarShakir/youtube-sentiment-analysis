// popup.js

document.addEventListener("DOMContentLoaded", async () => {
    const outputDiv = document.getElementById("output");
    const API_KEY = 'AIzaSyAHxO8ZCKPLODbhgSQcDV49Bv8cgkOA8Z4';
    const API_URL = 'http://localhost:8000';

    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
        const url = tabs[0].url;
        const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
        const match = url.match(youtubeRegex);

        if (match && match[1]) {
            const videoId = match[1];
            outputDiv.innerHTML = `
                <div class="section">
                    <span class="section-title">Video ID</span>
                    <p style="font-family:monospace;font-size:11px;padding:8px 14px 12px;color:#4d9fff;">${videoId}</p>
                </div>
                <p>Fetching comments&hellip;</p>`;

            const comments = await fetchComments(videoId);
            if (comments.length === 0) {
                outputDiv.innerHTML += "<p>No comments found for this video.</p>";
                return;
            }

            outputDiv.innerHTML += `<p>Fetched ${comments.length} comments &mdash; running sentiment analysis&hellip;</p>`;
            const predictions = await getSentimentPredictions(comments);

            if (predictions) {
                const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
                const sentimentData = [];
                const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);

                predictions.forEach((item) => {
                    sentimentCounts[item.sentiment]++;
                    sentimentData.push({ timestamp: item.timestamp, sentiment: parseInt(item.sentiment) });
                });

                const totalComments = comments.length;
                const uniqueCommenters = new Set(comments.map(c => c.authorId)).size;
                const totalWords = comments.reduce((sum, c) => sum + c.text.split(/\s+/).filter(w => w.length > 0).length, 0);
                const avgWordLength = (totalWords / totalComments).toFixed(1);
                const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
                const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(1);

                // Metrics
                outputDiv.innerHTML += `
                <div class="section">
                    <span class="section-title">Summary</span>
                    <div class="metrics-container">
                        <div class="metric">
                            <div class="metric-title">Comments</div>
                            <div class="metric-value">${totalComments}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-title">Commenters</div>
                            <div class="metric-value">${uniqueCommenters}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-title">Avg length</div>
                            <div class="metric-value">${avgWordLength}<span style="font-size:11px;font-weight:400;color:#8888a0;margin-left:2px;">wds</span></div>
                        </div>
                        <div class="metric">
                            <div class="metric-title">Sentiment</div>
                            <div class="metric-value">${normalizedSentimentScore}<span style="font-size:11px;font-weight:400;color:#8888a0;margin-left:1px;">/10</span></div>
                        </div>
                    </div>
                </div>`;

                // Pie chart
                outputDiv.innerHTML += `
                <div class="section">
                    <span class="section-title">Sentiment Distribution</span>
                    <div id="chart-container"></div>
                </div>`;
                await fetchAndDisplayChart(sentimentCounts);

                // Trend graph
                outputDiv.innerHTML += `
                <div class="section">
                    <span class="section-title">Sentiment Trend Over Time</span>
                    <div id="trend-graph-container"></div>
                </div>`;
                await fetchAndDisplayTrendGraph(sentimentData);

                // Word cloud
                outputDiv.innerHTML += `
                <div class="section">
                    <span class="section-title">Comment Word Cloud</span>
                    <div id="wordcloud-container"></div>
                </div>`;
                await fetchAndDisplayWordCloud(comments.map(c => c.text));

                // Top comments
                const sentimentLabel = { "1": "Positive", "0": "Neutral", "-1": "Negative" };
                outputDiv.innerHTML += `
                <div class="section">
                    <span class="section-title">Top 25 Comments</span>
                    <ul class="comment-list">
                        ${predictions.slice(0, 25).map((item, index) => `
                        <li class="comment-item">
                            <span>${index + 1}. ${escapeHtml(item.comment)}</span>
                            <span class="comment-sentiment" data-sentiment="${item.sentiment}">${sentimentLabel[item.sentiment] || item.sentiment}</span>
                        </li>`).join('')}
                    </ul>
                </div>`;
            }
        } else {
            outputDiv.innerHTML = "<p>Open a YouTube video to see comment insights.</p>";
        }
    });

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    async function fetchComments(videoId) {
        let comments = [];
        let pageToken = "";
        try {
            while (comments.length < 500) {
                const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
                const data = await response.json();
                if (data.items) {
                    data.items.forEach(item => {
                        const snippet = item.snippet.topLevelComment.snippet;
                        comments.push({
                            text: snippet.textOriginal,
                            timestamp: snippet.publishedAt,
                            authorId: snippet.authorChannelId?.value || 'Unknown'
                        });
                    });
                }
                pageToken = data.nextPageToken;
                if (!pageToken) break;
            }
        } catch (error) {
            console.error("Error fetching comments:", error);
            outputDiv.innerHTML += "<p>Error fetching comments.</p>";
        }
        return comments;
    }

    async function getSentimentPredictions(comments) {
        try {
            const response = await fetch(`${API_URL}/predict_with_timestamps`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ comments })
            });
            const result = await response.json();
            if (response.ok) return result;
            throw new Error(result.error || 'Error fetching predictions');
        } catch (error) {
            console.error("Error fetching predictions:", error);
            outputDiv.innerHTML += "<p>Error fetching sentiment predictions.</p>";
            return null;
        }
    }

    async function fetchAndDisplayChart(sentimentCounts) {
        try {
            const response = await fetch(`${API_URL}/generate_chart`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sentiment_counts: sentimentCounts })
            });
            if (!response.ok) throw new Error('Failed to fetch chart image');
            const blob = await response.blob();
            appendImage('chart-container', blob);
        } catch (error) {
            console.error("Error fetching chart image:", error);
        }
    }

    async function fetchAndDisplayWordCloud(comments) {
        try {
            const response = await fetch(`${API_URL}/generate_wordcloud`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ comments })
            });
            if (!response.ok) throw new Error('Failed to fetch word cloud image');
            const blob = await response.blob();
            appendImage('wordcloud-container', blob);
        } catch (error) {
            console.error("Error fetching word cloud image:", error);
        }
    }

    async function fetchAndDisplayTrendGraph(sentimentData) {
        try {
            const response = await fetch(`${API_URL}/generate_trend_graph`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sentiment_data: sentimentData })
            });
            if (!response.ok) throw new Error('Failed to fetch trend graph image');
            const blob = await response.blob();
            appendImage('trend-graph-container', blob);
        } catch (error) {
            console.error("Error fetching trend graph image:", error);
        }
    }

    function appendImage(containerId, blob) {
        const imgURL = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = imgURL;
        document.getElementById(containerId)?.appendChild(img);
    }
});