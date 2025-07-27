import React, { useState } from "react";

export default function ChatWithPDF() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");

  const API_URL = "http://127.0.0.1:10000";

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a PDF first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/api/upload-pdf`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      alert(`PDF uploaded! Chunks created: ${data.chunks_created}`);
    } catch (err) {
      console.error(err);
      alert("Error uploading PDF");
    }
  };

  const handleChat = async () => {
    if (!message) {
      alert("Please enter a question.");
      return;
    }
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      setResponse(data.response.html);
    } catch (err) {
      console.error(err);
      alert("Error getting response");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Pension Scheme Assistant</h1>

      {/* Upload PDF */}
      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleUpload}>Upload PDF</button>

      {/* Chat Input */}
      <div style={{ marginTop: "20px" }}>
        <input
          type="text"
          placeholder="Ask about a scheme..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          style={{ width: "60%", padding: "10px" }}
        />
        <button onClick={handleChat}>Ask</button>
      </div>

      {/* Response */}
      <div
        style={{
          marginTop: "20px",
          padding: "15px",
          background: "#f9f9f9",
          borderRadius: "8px",
        }}
        dangerouslySetInnerHTML={{ __html: response }}
      />
    </div>
  );
}
