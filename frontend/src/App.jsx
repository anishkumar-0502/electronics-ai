/* eslint-disable no-unused-vars */
import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("ask"); // 'ask' or 'stream'
  const chatRef = useRef(null);
  const textareaRef = useRef(null);
  const [messagesByMode, setMessagesByMode] = useState({
    ask: [],
    stream: [],
  });

  const handleAskSubmit = async () => {
    if (!query.trim()) return;

    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const userMsg = { type: "user", text: query, timestamp };
    const thinkingMsg = { type: "ai", text: "⚙️ Thinking...", timestamp, isThinking: true };
    const currentMessages = messagesByMode[mode];
    const currentIndex = currentMessages.length;

    const updatedMessages = [...currentMessages, userMsg, thinkingMsg];
    setMessagesByMode((prev) => ({ ...prev, [mode]: updatedMessages }));

    try {
      const res = await fetch(`http://localhost:8000/${mode === "ask" ? "ask" : "ask-stream"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error("Network response was not ok");

      if (mode === "ask") {
        const data = await res.json();
        setMessagesByMode((prev) => {
          const updated = { ...prev };
          updated[mode][currentIndex + 1] = { type: "ai", text: data.response, timestamp };
          return updated;
        });
      } else {
        const reader = res.body.getReader();
        let responseText = "";

        setMessagesByMode((prev) => {
          const updated = { ...prev };
          updated[mode][currentIndex + 1] = { type: "ai", text: "", timestamp, isStreaming: true };
          return updated;
        });

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          responseText += new TextDecoder().decode(value);
          setMessagesByMode((prev) => {
            const updated = { ...prev };
            updated[mode][currentIndex + 1] = {
              type: "ai",
              text: responseText,
              timestamp,
              isStreaming: true,
            };
            return updated;
          });
        }
      }
    } catch (error) {
      setMessagesByMode((prev) => {
        const updated = { ...prev };
        updated[mode][currentIndex + 1] = {
          type: "ai",
          text: "Unable to fetch response",
          timestamp,
        };
        return updated;
      });
    }

    setQuery("");
    textareaRef.current?.focus();
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    // No need to clear messages; they are mode-specific
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAskSubmit();
    }
  };

  const currentMessages = messagesByMode[mode];

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [currentMessages]);

  return (
    <div className="app-wrapper">
      <div className="mode-selector">
        <button
          className={`mode-btn ${mode === "ask" ? "active" : ""}`}
          onClick={() => handleModeChange("ask")}
        >
          Standard Ask
        </button>
        <button
          className={`mode-btn ${mode === "stream" ? "active" : ""}`}
          onClick={() => handleModeChange("stream")}
        >
          Stream Response
        </button>
      </div>

      <div className="chat-window" ref={chatRef} role="log" aria-live="polite">
        <div className="response-box">
          {messagesByMode[mode].map((msg, index) => (
            <div
              key={index}
              className={`chat-bubble ${msg.type} ${msg.isThinking ? "thinking" : msg.isStreaming ? "streaming" : ""}`}
            >
              <span>{msg.text}</span>
              <span className="message-time">{msg.timestamp}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="input-bar" role="form">
        <textarea
          ref={textareaRef}
          className="query-input"
          rows={2}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask Semicon's AI about electronics design..."
          aria-label="Enter your question about electronics design"
        />
        <button className="submit-button" onClick={handleAskSubmit} aria-label="Send message">
          ➤
          <span className="sr-only">Send</span>
        </button>
      </div>
    </div>
  );
}

export default App;