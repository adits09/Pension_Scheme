import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import Header from './Header';
import './App.css';
import LoginPage from './Login_Page';
import SignupPage from './Signup_Page';
import Widget_1 from './Widget_1';
import Widget_2 from './Widget_2';
import Widget_3 from './Widget_3';

const ChatComponent = () => {
  const [conversations, setConversations] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleNewChat = () => {
    const newConversation = {
      id: Date.now(),
      title: `Chat ${conversations.length + 1}`,
      messages: []
    };
    setConversations([...conversations, newConversation]);
    setMessages([]);
  };

  const handleSendMessage = async () => {
    if (currentMessage.trim()) {
      const userMessage = {
        id: Date.now(),
        text: currentMessage,
        sender: 'user',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, userMessage]);
      setCurrentMessage('');
      setIsLoading(true);

      try {
        const response = await fetch('https://pension-scheme.onrender.com/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: currentMessage }),
        });

        const data = await response.json();

        const botMessage = {
          id: Date.now() + 1,
          text: data.response?.raw || data.response || data.error || 'Sorry, I encountered an error.',
          html: data.response?.html || null,
          structured: data.response?.structured || null,
          sender: 'bot',
          timestamp: new Date(),
          contextFound: data.context_found || 0
        };


        setMessages(prev => [...prev, botMessage]);
      } catch (error) {
        const errorMessage = {
          id: Date.now() + 1,
          text: 'Sorry, I could not connect to the server.',
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      const formData = new FormData();
      formData.append('file', file); 

      try {
        const response = await fetch('https://pension-scheme.onrender.com/api/upload-pdf', {
          method: 'POST',
          body: formData, 
        });

        const data = await response.json();
        if (data.status === 'success') {
          const uploadMessage = {
            id: Date.now(),
            text: `PDF uploaded successfully!`,
            sender: 'system',
            timestamp: new Date()
          };
          setMessages(prev => [...prev, uploadMessage]);
        } else {
          alert(data.message || data.error);
        }
      } catch (error) {
        alert('Error uploading file');
      }
    } else {
      alert('Please select a PDF file');
    }
  };

  const MessageContent = ({ message }) => {
  if (message.sender === 'bot' && message.html) {
    return (
      <div className="message-content">
        <div
          className="formatted-response"
          dangerouslySetInnerHTML={{ __html: message.html }}
        />
      </div>
    );
  }
  return (
    <div className="message-content">
      {message.text}
    </div>
  );
};


  return (
    <div className="main-container">
      <div className="chat-widget">
        <div className="widget-left">
          <button className="new-chat-button" onClick={handleNewChat}>
            <span className="plus-icon">+</span>
            New chat
          </button>
          <div className="faq-section">
            <h3 className="faq-title">Frequently Asked Questions</h3>
            <div className="faq-list">
              <button className="faq-question" onClick={() => setCurrentMessage('How can I apply for a government scheme?')}>How can I apply for a government scheme?</button>
              <button className="faq-question" onClick={() => setCurrentMessage("What documents are required for Rajasthan's pension schemes?")}>What documents are required for Rajasthan's pension schemes?</button>
              <button className="faq-question" onClick={() => setCurrentMessage('Where can I find the Rajasthan government calendar?')}>Where can I find the Rajasthan government calendar?</button>
              <button className="faq-question" onClick={() => setCurrentMessage('How do I contact support?')}>How do I contact support?</button>
            </div>
          </div>
        </div>
        <div className="widget-center">
          {messages.length === 0 ? (
            <div className="welcome-section">
              <div className="avatar">🙏</div>
              <h1 className="welcome-text">
                Namaste, How can<br />
                I help you today?
              </h1>
              <div className="widgets-grid">
                <Widget_1 />
                <Widget_2 />
                <Widget_3 />
              </div>
            </div>
          ) : (
            <>
              <div className="chatbot-header">
                <span className="chatbot-title">SevaSaathi</span>
              </div>
              <div className="chat-messages">
                {messages.map((message) => (
                  <div key={message.id} className={`message ${message.sender}`}>
                    <MessageContent message={message} />
                  </div>
                ))}
                {isLoading && (
                  <div className="message bot">
                    <div className="message-content loading-message">
                      Thinking
                      <div className="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
          <div className="input-container">
            <div className="message-input-wrapper">
              <input
                type="file"
                name="file" 
                accept=".pdf"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                id="pdf-upload"
              />
              <label htmlFor="pdf-upload" className="attach-button">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/>
                </svg>
              </label>
              <input
                type="text"
                placeholder="Send a message..."
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                className="message-input"
                disabled={isLoading}
              />
              <button 
                className="send-button"
                onClick={handleSendMessage}
                disabled={!currentMessage.trim() || isLoading}
              >
                <span className="send-icon">➤</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const AppContent = () => {
  const location = useLocation();
  const hideHeaderRoutes = ['/login', '/signup'];
  const shouldHideHeader = hideHeaderRoutes.includes(location.pathname);

  return (
    <div className="app">
      {!shouldHideHeader && <Header />}
      <Routes>
        <Route path="/" element={<ChatComponent />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
      </Routes>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
